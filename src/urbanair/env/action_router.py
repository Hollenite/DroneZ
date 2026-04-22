from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..enums import ActionType, DroneStatus, DroneType


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
    ActionType.RETURN_TO_CHARGE,
    ActionType.PRIORITIZE_ORDER,
)
_SUPPORTED_ACTIONS = set(SUPPORTED_ACTIONS)
_AVAILABLE_ASSIGNMENT_STATUSES = {DroneStatus.IDLE, DroneStatus.HOLDING}
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
        drone = next((item for item in state.fleet if item.drone_id == params["drone_id"]), None)
        if drone is None:
            return RoutedAction.invalid("assign_delivery", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.drone_type == DroneType.RELAY:
            return RoutedAction.invalid("assign_delivery", params, "invalid_drone_type", "Relay drones cannot take delivery assignments.")
        if drone.status not in _AVAILABLE_ASSIGNMENT_STATUSES:
            return RoutedAction.invalid(
                "assign_delivery",
                params,
                "drone_unavailable",
                f"Drone '{drone.drone_id}' is not available for assignment.",
            )

        order = next((item for item in state.orders if item.order_id == params["order_id"]), None)
        if order is None:
            return RoutedAction.invalid("assign_delivery", params, "unknown_order", f"Order '{params['order_id']}' does not exist.")
        if order.order_id in state.resolved_order_ids:
            return RoutedAction.invalid("assign_delivery", params, "order_resolved", f"Order '{order.order_id}' is already resolved.")
        if order.assigned_drone_id is not None:
            return RoutedAction.invalid("assign_delivery", params, "order_assigned", f"Order '{order.order_id}' is already assigned.")

        return RoutedAction(
            simulator_action={
                "action_type": ActionType.ASSIGN_DELIVERY.value,
                "drone_id": drone.drone_id,
                "order_id": order.order_id,
            },
            action_type=ActionType.ASSIGN_DELIVERY.value,
            normalized_params={"drone_id": drone.drone_id, "order_id": order.order_id},
            is_valid=True,
        )

    def _validate_return_to_charge(self, state, params: dict[str, Any]) -> RoutedAction:
        drone = next((item for item in state.fleet if item.drone_id == params["drone_id"]), None)
        if drone is None:
            return RoutedAction.invalid("return_to_charge", params, "unknown_drone", f"Drone '{params['drone_id']}' does not exist.")
        if drone.status not in _AVAILABLE_CHARGE_STATUSES:
            return RoutedAction.invalid(
                "return_to_charge",
                params,
                "drone_unavailable",
                f"Drone '{drone.drone_id}' cannot be sent to charge right now.",
            )

        station = next((item for item in state.charging_stations if item.station_id == params["station_id"]), None)
        if station is None:
            return RoutedAction.invalid(
                "return_to_charge",
                params,
                "unknown_station",
                f"Charging station '{params['station_id']}' does not exist.",
            )

        return RoutedAction(
            simulator_action={
                "action_type": ActionType.RETURN_TO_CHARGE.value,
                "drone_id": drone.drone_id,
                "station_id": station.station_id,
            },
            action_type=ActionType.RETURN_TO_CHARGE.value,
            normalized_params={"drone_id": drone.drone_id, "station_id": station.station_id},
            is_valid=True,
        )

    def _validate_prioritize_order(self, state, params: dict[str, Any]) -> RoutedAction:
        order = next((item for item in state.orders if item.order_id == params["order_id"]), None)
        if order is None:
            return RoutedAction.invalid("prioritize_order", params, "unknown_order", f"Order '{params['order_id']}' does not exist.")
        if order.order_id in state.resolved_order_ids:
            return RoutedAction.invalid("prioritize_order", params, "order_resolved", f"Order '{order.order_id}' is already resolved.")

        return RoutedAction(
            simulator_action={
                "action_type": ActionType.PRIORITIZE_ORDER.value,
                "order_id": order.order_id,
            },
            action_type=ActionType.PRIORITIZE_ORDER.value,
            normalized_params={"order_id": order.order_id},
            is_valid=True,
        )
