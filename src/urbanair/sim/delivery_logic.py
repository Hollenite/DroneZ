from __future__ import annotations

from ..enums import DroneStatus, DropMode, RecipientAvailability, RiskLevel
from ..models import DroneState, OrderState, SectorState


def resolve_delivery_attempts(
    fleet: list[DroneState],
    orders: list[OrderState],
    sectors: list[SectorState],
    failure_prone_zones: set[str],
    failed_drop_probability: float,
    rng,
    auto_attempt_enabled: bool = True,
    delivery_attempt_required: set[str] | None = None,
) -> tuple[list[str], set[str], dict[str, str]]:
    events: list[str] = []
    resolved_order_ids: set[str] = set()
    recovery_actions: dict[str, str] = {}
    delivery_attempt_required = delivery_attempt_required or set()
    orders_by_id = {order.order_id: order for order in orders}
    sector_by_id = {sector.zone_id: sector for sector in sectors}

    for drone in fleet:
        if drone.status != DroneStatus.IN_FLIGHT or not drone.assigned_order_id or drone.eta != 0:
            continue

        order = orders_by_id.get(drone.assigned_order_id)
        if order is None:
            continue
        if not auto_attempt_enabled and order.order_id not in delivery_attempt_required:
            events.append(f"{order.order_id} awaiting explicit delivery attempt.")
            continue

        sector = sector_by_id.get(order.zone_id)
        blocked = bool(sector and sector.is_no_fly)
        recipient_unavailable = order.recipient_availability == RecipientAvailability.UNAVAILABLE
        failure_zone = order.zone_id in failure_prone_zones
        stochastic_failure = rng.random() < failed_drop_probability

        if blocked or recipient_unavailable or failure_zone or stochastic_failure:
            order.retry_count += 1
            order.status = "failed"
            drone.failed_order_count += 1
            drone.status = DroneStatus.HOLDING
            drone.current_zone = order.zone_id
            drone.hold_reason = "delivery_failure"
            recovery_actions[order.order_id] = _choose_recovery(order)
            reason = _failure_reason(blocked, recipient_unavailable, failure_zone, stochastic_failure)
            events.append(f"Delivery attempt for {order.order_id} failed: {reason}.")
            order.assigned_drone_id = None
            drone.assigned_order_id = None
            drone.target_zone = None
            delivery_attempt_required.discard(order.order_id)
            if recovery_actions[order.order_id] == "locker_fallback":
                order.drop_mode = DropMode.LOCKER
                order.status = "locker_fallback"
                events.append(f"{order.order_id} switched to locker fallback.")
            elif recovery_actions[order.order_id] == "retry":
                order.status = "delivery_attempted"
                events.append(f"{order.order_id} marked for retry.")
            else:
                order.status = "deferred"
                events.append(f"{order.order_id} deferred for reassignment.")
        else:
            delivery_attempt_required.discard(order.order_id)
            order.status = "delivered"
            drone.delivered_order_count += 1
            drone.status = DroneStatus.IDLE
            drone.current_zone = order.zone_id
            drone.assigned_order_id = None
            drone.target_zone = None
            drone.hold_reason = None
            order.assigned_drone_id = None
            resolved_order_ids.add(order.order_id)
            events.append(f"{order.order_id} delivered successfully by {drone.drone_id}.")

        if drone.battery <= 15:
            drone.health_risk = RiskLevel.CRITICAL

    return events, resolved_order_ids, recovery_actions


def _choose_recovery(order: OrderState) -> str:
    if order.recipient_availability == RecipientAvailability.UNAVAILABLE:
        return "locker_fallback"
    if order.retry_count >= 2:
        return "reassign"
    return "retry"


def _failure_reason(blocked: bool, recipient_unavailable: bool, failure_zone: bool, stochastic_failure: bool) -> str:
    if blocked:
        return "temporary no-fly restriction"
    if recipient_unavailable:
        return "recipient unavailable"
    if failure_zone:
        return "sector reliability degradation"
    if stochastic_failure:
        return "operational drop failure"
    return "unknown"
