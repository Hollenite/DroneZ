from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..enums import ActionType, DroneStatus, DroneType

_VALID_REROUTE_CORRIDORS = {"direct", "weather_avoid", "congestion_avoid", "safe"}


_ACTION_PARAMETERS: dict[ActionType, tuple[str, ...]] = {
    ActionType.ASSIGN_DELIVERY: ("drone_id", "order_id"),
    ActionType.REROUTE: ("drone_id", "corridor"),
    ActionType.RETURN_TO_CHARGE: ("drone_id", "station_id"),
    ActionType.RESERVE_CHARGER: ("drone_id", "station_id"),
    ActionType.DELAY_ORDER: ("order_id",),
    ActionType.PRIORITIZE_ORDER: ("order_id",),
    ActionType.SWAP_ASSIGNMENTS: ("drone_a", "drone_b"),
    ActionType.ATTEMPT_DELIVERY: ("drone_id", "mode"),
    ActionType.FALLBACK_TO_LOCKER: ("order_id", "locker_id"),
    ActionType.HOLD_FLEET: ("zone_id",),
    ActionType.RESUME_OPERATIONS: ("zone_id",),
}
SUPPORTED_ACTIONS = (
    ActionType.ASSIGN_DELIVERY,
    ActionType.REROUTE,
    ActionType.RETURN_TO_CHARGE,
    ActionType.RESERVE_CHARGER,
    ActionType.DELAY_ORDER,
    ActionType.PRIORITIZE_ORDER,
    ActionType.SWAP_ASSIGNMENTS,
    ActionType.ATTEMPT_DELIVERY,
    ActionType.FALLBACK_TO_LOCKER,
    ActionType.HOLD_FLEET,
    ActionType.RESUME_OPERATIONS,
)
_SUPPORTED_ACTIONS = set(SUPPORTED_ACTIONS)
_AVAILABLE_ASSIGNMENT_HOLD_REASONS = {None, "delivery_failure", "order_delayed", "charger_reserved"}
_AVAILABLE_CHARGE_STATUSES = {DroneStatus.IDLE, DroneStatus.HOLDING, DroneStatus.ASSIGNED}


@dataclass(frozen=True)
class RoutedAction:
    simulator_action: dict[str, Any] | None
    action_type: str
    normalized_params: dict[str, Any]
    is_valid: bool
    error_code: str | None = None
    error_message: str | None = None

    @classmethod
    def invalid(cls, action_type: str, params: dict[str, Any], error_code: str, error_message: str) -> "RoutedAction":
        return cls(
            simulator_action=None,
            action_type=action_type,
            normalized_params=params,
            is_valid=False,
            error_code=error_code,
            error_message=error_message,
        )


class ActionRouter:
    def route(self, state, action_payload: dict[str, Any] | None) -> RoutedAction:
        if not isinstance(action_payload, dict):
            return RoutedAction.invalid("unknown", {}, "invalid_payload", "Action must be a dictionary.")

        action_name = action_payload.get("action")
        params = action_payload.get("params") or {}
        if not isinstance(params, dict):
            return RoutedAction.invalid(str(action_name or "unknown"), {}, "invalid_params", "Action params must be a dictionary.")

        try:
            action_type = ActionType(action_name)
        except Exception:
            return RoutedAction.invalid(str(action_name or "unknown"), params, "unknown_action", f"Unsupported action '{action_name}'.")

        missing = [name for name in _ACTION_PARAMETERS[action_type] if name not in params]
        if missing:
            return RoutedAction.invalid(
                action_type.value,
                params,
                "missing_params",
                f"Missing required params: {', '.join(missing)}.",
            )

        if action_type not in _SUPPORTED_ACTIONS:
            return RoutedAction.invalid(
                action_type.value,
                params,
                "unsupported_action",
                f"Action '{action_type.value}' is recognized but not implemented yet.",
            )

        validator = getattr(self, f"_validate_{action_type.value}")
        return validator(state, params)

    def _validate_assign_delivery(self, state, params: dict[str, Any]) -> RoutedAction:
        drone = self._get_drone(state, params["drone_id"])
        if drone is None:
            return RoutedAction.invalid("assign_delivery", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.drone_type == DroneType.RELAY:
            return RoutedAction.invalid("assign_delivery", params, "invalid_drone_type", "Relay drones cannot take delivery assignments.")
        if not self._drone_available_for_assignment(drone):
            return RoutedAction.invalid("assign_delivery", params, "drone_unavailable", f"Drone '{drone.drone_id}' is not available for assignment.")

        order = self._get_open_order(state, params["order_id"], "assign_delivery", params)
        if isinstance(order, RoutedAction):
            return order
        if order.assigned_drone_id is not None:
            return RoutedAction.invalid("assign_delivery", params, "order_assigned", f"Order '{order.order_id}' is already assigned.")

        return self._valid(ActionType.ASSIGN_DELIVERY, {"drone_id": drone.drone_id, "order_id": order.order_id})

    def _validate_reroute(self, state, params: dict[str, Any]) -> RoutedAction:
        drone = self._get_drone(state, params["drone_id"])
        if drone is None:
            return RoutedAction.invalid("reroute", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.drone_type == DroneType.RELAY:
            return RoutedAction.invalid("reroute", params, "invalid_drone_type", "Relay drones cannot be rerouted for deliveries.")
        if drone.status not in {DroneStatus.ASSIGNED, DroneStatus.IN_FLIGHT}:
            return RoutedAction.invalid("reroute", params, "drone_unavailable", f"Drone '{drone.drone_id}' is not actively carrying an assignment.")
        if drone.assigned_order_id is None or drone.eta == 0:
            return RoutedAction.invalid("reroute", params, "reroute_window_closed", f"Drone '{drone.drone_id}' cannot be rerouted at the delivery-attempt boundary.")
        if params["corridor"] not in _VALID_REROUTE_CORRIDORS:
            return RoutedAction.invalid("reroute", params, "invalid_corridor", f"Unsupported corridor '{params['corridor']}'.")
        if drone.target_zone is None:
            return RoutedAction.invalid("reroute", params, "missing_target_zone", f"Drone '{drone.drone_id}' has no active destination.")
        sector = self._get_sector(state, drone.target_zone)
        if sector is not None and sector.operations_paused:
            return RoutedAction.invalid("reroute", params, "zone_paused", f"Zone '{sector.zone_id}' is paused for routing.")

        return self._valid(ActionType.REROUTE, {"drone_id": drone.drone_id, "corridor": params["corridor"]})

    def _validate_return_to_charge(self, state, params: dict[str, Any]) -> RoutedAction:
        drone = self._get_drone(state, params["drone_id"])
        if drone is None:
            return RoutedAction.invalid("return_to_charge", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.drone_type == DroneType.RELAY:
            return RoutedAction.invalid("return_to_charge", params, "invalid_drone_type", "Relay drones do not use charging stations.")
        if drone.status not in _AVAILABLE_CHARGE_STATUSES:
            return RoutedAction.invalid("return_to_charge", params, "drone_unavailable", f"Drone '{drone.drone_id}' cannot be sent to charge right now.")

        station = self._get_station(state, params["station_id"])
        if station is None:
            return RoutedAction.invalid("return_to_charge", params, "unknown_station", f"Charging station '{params['station_id']}' does not exist.")

        return self._valid(ActionType.RETURN_TO_CHARGE, {"drone_id": drone.drone_id, "station_id": station.station_id})

    def _validate_reserve_charger(self, state, params: dict[str, Any]) -> RoutedAction:
        drone = self._get_drone(state, params["drone_id"])
        if drone is None:
            return RoutedAction.invalid("reserve_charger", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.drone_type == DroneType.RELAY:
            return RoutedAction.invalid("reserve_charger", params, "invalid_drone_type", "Relay drones do not use charging stations.")
        if drone.status not in {DroneStatus.IDLE, DroneStatus.HOLDING, DroneStatus.ASSIGNED}:
            return RoutedAction.invalid("reserve_charger", params, "drone_unavailable", f"Drone '{drone.drone_id}' cannot reserve charging right now.")

        station = self._get_station(state, params["station_id"])
        if station is None:
            return RoutedAction.invalid("reserve_charger", params, "unknown_station", f"Charging station '{params['station_id']}' does not exist.")

        return self._valid(ActionType.RESERVE_CHARGER, {"drone_id": drone.drone_id, "station_id": station.station_id})

    def _validate_delay_order(self, state, params: dict[str, Any]) -> RoutedAction:
        order = self._get_open_order(state, params["order_id"], "delay_order", params)
        if isinstance(order, RoutedAction):
            return order
        return self._valid(ActionType.DELAY_ORDER, {"order_id": order.order_id})

    def _validate_prioritize_order(self, state, params: dict[str, Any]) -> RoutedAction:
        order = self._get_open_order(state, params["order_id"], "prioritize_order", params)
        if isinstance(order, RoutedAction):
            return order
        return self._valid(ActionType.PRIORITIZE_ORDER, {"order_id": order.order_id})

    def _validate_swap_assignments(self, state, params: dict[str, Any]) -> RoutedAction:
        drone_a = self._get_drone(state, params["drone_a"])
        drone_b = self._get_drone(state, params["drone_b"])
        if drone_a is None or drone_b is None:
            return RoutedAction.invalid("swap_assignments", params, "unknown_drone", "Both drones must exist for swap_assignments.")
        if drone_a.drone_id == drone_b.drone_id:
            return RoutedAction.invalid("swap_assignments", params, "same_drone", "swap_assignments requires two distinct drones.")
        if drone_a.drone_type == DroneType.RELAY or drone_b.drone_type == DroneType.RELAY:
            return RoutedAction.invalid("swap_assignments", params, "invalid_drone_type", "Relay drones cannot participate in swap_assignments.")
        if drone_a.status not in {DroneStatus.ASSIGNED, DroneStatus.IN_FLIGHT} or drone_b.status not in {DroneStatus.ASSIGNED, DroneStatus.IN_FLIGHT}:
            return RoutedAction.invalid("swap_assignments", params, "drone_unavailable", "Both drones must be actively assigned before swapping.")
        if drone_a.assigned_order_id is None or drone_b.assigned_order_id is None:
            return RoutedAction.invalid("swap_assignments", params, "missing_assignment", "Both drones must have active assignments.")
        if drone_a.status == DroneStatus.IN_FLIGHT or drone_b.status == DroneStatus.IN_FLIGHT:
            return RoutedAction.invalid("swap_assignments", params, "midflight_swap_unsupported", "swap_assignments only supports pre-delivery assigned drones in this pass.")
        order_a = self._get_open_order(state, drone_a.assigned_order_id, "swap_assignments", params)
        if isinstance(order_a, RoutedAction):
            return order_a
        order_b = self._get_open_order(state, drone_b.assigned_order_id, "swap_assignments", params)
        if isinstance(order_b, RoutedAction):
            return order_b
        if order_a.package_weight > drone_b.payload_capacity or order_b.package_weight > drone_a.payload_capacity:
            return RoutedAction.invalid("swap_assignments", params, "payload_incompatible", "The swapped assignments exceed at least one drone payload capacity.")
        if order_a.order_id in state.delivery_attempt_required or order_b.order_id in state.delivery_attempt_required:
            return RoutedAction.invalid("swap_assignments", params, "attempt_pending", "Assignments with pending explicit delivery attempts cannot be swapped.")

        return self._valid(ActionType.SWAP_ASSIGNMENTS, {"drone_a": drone_a.drone_id, "drone_b": drone_b.drone_id})

    def _validate_attempt_delivery(self, state, params: dict[str, Any]) -> RoutedAction:
        drone = self._get_drone(state, params["drone_id"])
        if drone is None:
            return RoutedAction.invalid("attempt_delivery", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.assigned_order_id is None:
            return RoutedAction.invalid("attempt_delivery", params, "no_assigned_order", f"Drone '{drone.drone_id}' has no assigned order.")
        if drone.status not in {DroneStatus.ASSIGNED, DroneStatus.IN_FLIGHT}:
            return RoutedAction.invalid("attempt_delivery", params, "drone_unavailable", f"Drone '{drone.drone_id}' cannot attempt delivery right now.")
        if params["mode"] not in {"doorstep", "locker", "handoff"}:
            return RoutedAction.invalid("attempt_delivery", params, "invalid_mode", f"Unsupported delivery mode '{params['mode']}'.")

        return self._valid(ActionType.ATTEMPT_DELIVERY, {"drone_id": drone.drone_id, "mode": params["mode"]})

    def _validate_fallback_to_locker(self, state, params: dict[str, Any]) -> RoutedAction:
        order = self._get_open_order(state, params["order_id"], "fallback_to_locker", params)
        if isinstance(order, RoutedAction):
            return order
        if "locker" not in order.fallback_options:
            return RoutedAction.invalid("fallback_to_locker", params, "locker_unavailable", f"Order '{order.order_id}' does not support locker fallback.")
        if str(order.drop_mode.value if hasattr(order.drop_mode, 'value') else order.drop_mode) == "locker" and order.status == "locker_fallback":
            return RoutedAction.invalid("fallback_to_locker", params, "locker_already_selected", f"Order '{order.order_id}' is already using locker fallback.")

        return self._valid(ActionType.FALLBACK_TO_LOCKER, {"order_id": order.order_id, "locker_id": str(params["locker_id"])})

    def _validate_hold_fleet(self, state, params: dict[str, Any]) -> RoutedAction:
        sector = self._get_sector(state, params["zone_id"])
        if sector is None:
            return RoutedAction.invalid("hold_fleet", params, "unknown_zone", f"Zone '{params['zone_id']}' does not exist.")
        if sector.operations_paused:
            return RoutedAction.invalid("hold_fleet", params, "already_held", f"Zone '{sector.zone_id}' is already on hold.")

        return self._valid(ActionType.HOLD_FLEET, {"zone_id": sector.zone_id})

    def _validate_resume_operations(self, state, params: dict[str, Any]) -> RoutedAction:
        sector = self._get_sector(state, params["zone_id"])
        if sector is None:
            return RoutedAction.invalid("resume_operations", params, "unknown_zone", f"Zone '{params['zone_id']}' does not exist.")
        if not sector.operations_paused:
            return RoutedAction.invalid("resume_operations", params, "zone_not_held", f"Zone '{sector.zone_id}' is not currently on hold.")

        return self._valid(ActionType.RESUME_OPERATIONS, {"zone_id": sector.zone_id})

    def _valid(self, action_type: ActionType, normalized_params: dict[str, Any]) -> RoutedAction:
        return RoutedAction(
            simulator_action={"action_type": action_type.value, **normalized_params},
            action_type=action_type.value,
            normalized_params=normalized_params,
            is_valid=True,
        )

    def _get_drone(self, state, drone_id: str):
        return next((item for item in state.fleet if item.drone_id == drone_id), None)

    def _get_station(self, state, station_id: str):
        return next((item for item in state.charging_stations if item.station_id == station_id), None)

    def _get_sector(self, state, zone_id: str):
        return next((item for item in state.sectors if item.zone_id == zone_id), None)

    def _get_open_order(self, state, order_id: str, action_name: str, params: dict[str, Any]):
        order = next((item for item in state.orders if item.order_id == order_id), None)
        if order is None:
            return RoutedAction.invalid(action_name, params, "unknown_order", f"Order '{order_id}' does not exist.")
        if order.order_id in state.resolved_order_ids or order.status in {"delivered", "canceled"}:
            return RoutedAction.invalid(action_name, params, "order_resolved", f"Order '{order.order_id}' is already resolved.")
        return order

    def _drone_available_for_assignment(self, drone) -> bool:
        if drone.status == DroneStatus.IDLE:
            return True
        return drone.status == DroneStatus.HOLDING and drone.hold_reason in _AVAILABLE_ASSIGNMENT_HOLD_REASONS
