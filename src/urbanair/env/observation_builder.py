from __future__ import annotations

from ..enums import ActionType, OrderPriority
from ..models import RewardBreakdown
from .reward_engine import serialize_breakdown


def build_observation(state, reward_breakdown: RewardBreakdown | None = None, last_action: dict | None = None) -> dict[str, object]:
    breakdown = reward_breakdown or RewardBreakdown()
    warnings = _build_warnings(state)
    fleet = [_serialize_drone(drone) for drone in sorted(state.fleet, key=lambda item: item.drone_id)]
    orders = [_serialize_order(order) for order in sorted(state.orders, key=_order_sort_key)]
    sectors = [_serialize_sector(sector) for sector in sorted(state.sectors, key=lambda item: item.zone_id)]
    charging = [_serialize_station(station) for station in sorted(state.charging_stations, key=lambda item: item.station_id)]
    notices = [_serialize_notice(notice) for notice in sorted(state.policy_notices, key=lambda item: item.notice_id)]
    emergencies = [_serialize_event(event) for event in sorted(state.emergency_events, key=lambda item: item.event_id)]
    action_reminder = [action.value for action in ActionType]
    observation = {
        "task_id": state.task_config.task_id,
        "step": state.tick,
        "max_steps": state.task_config.horizon,
        "fleet": fleet,
        "orders": orders,
        "city": {
            "sectors": sectors,
            "policy_notices": notices,
            "emergency_events": emergencies,
            "active_no_fly_zones": [sector["zone_id"] for sector in sectors if sector["is_no_fly"]],
        },
        "charging": charging,
        "recent_events": list(state.recent_events),
        "warnings": warnings,
        "action_reminder": action_reminder,
        "reward": serialize_breakdown(breakdown),
        "summary": build_summary(state, fleet, orders, charging, warnings, breakdown, last_action),
    }
    if last_action is not None:
        observation["last_action"] = last_action
    return observation


def build_summary(state, fleet: list[dict[str, object]], orders: list[dict[str, object]], charging: list[dict[str, object]], warnings: list[str], breakdown: RewardBreakdown, last_action: dict | None) -> str:
    lines = [
        f"TIME_STEP: {state.tick} / {state.task_config.horizon}",
        f"TASK: {state.task_config.task_id}",
        f"PENDING_ORDERS: {sum(1 for order in state.orders if order.order_id not in state.resolved_order_ids and order.status != 'canceled')}",
    ]

    no_fly = [sector.zone_id for sector in state.sectors if sector.is_no_fly]
    if no_fly:
        lines.append(f"ACTIVE_NO_FLY_ZONES: {', '.join(no_fly)}")

    weather_alerts = [f"{sector.zone_id}={sector.weather.value}" for sector in state.sectors if sector.weather.value != "clear"]
    if weather_alerts:
        lines.append(f"WEATHER_ALERTS: {', '.join(weather_alerts)}")

    if last_action:
        lines.append(f"LAST_ACTION: {last_action}")

    lines.append("")
    lines.append("FLEET:")
    for drone in fleet:
        lines.append(
            f"- {drone['drone_id']} | type={drone['drone_type']} | battery={drone['battery']} | zone={drone['current_zone']} | assigned={drone['assigned_order_id']} | eta={drone['eta']} | risk={drone['health_risk']}"
        )

    lines.append("ORDERS:")
    for order in orders[:5]:
        lines.append(
            f"- {order['order_id']} | priority={order['priority']} | deadline={order['deadline']} | recipient={order['recipient_availability']} | retry={order['retry_count']} | status={order['status']}"
        )

    lines.append("CHARGING:")
    for station in charging:
        lines.append(
            f"- {station['station_id']} | occupancy={station['occupied_slots']}/{station['capacity']} | queue={station['queue_size']}"
        )

    if state.recent_events:
        lines.append("RECENT EVENTS:")
        for event in state.recent_events[-5:]:
            lines.append(f"- {event}")

    if warnings:
        lines.append("WARNINGS:")
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.append(f"STEP_REWARD: {breakdown.total:.2f}")
    return "\n".join(lines)


def _serialize_drone(drone) -> dict[str, object]:
    return {
        "drone_id": drone.drone_id,
        "drone_type": drone.drone_type.value,
        "status": drone.status.value,
        "battery": drone.battery,
        "payload_capacity": drone.payload_capacity,
        "current_zone": drone.current_zone,
        "assigned_order_id": drone.assigned_order_id,
        "eta": drone.eta,
        "health_risk": drone.health_risk.value,
        "communication_strength": drone.communication_strength,
        "reserved_station_id": drone.reserved_station_id,
    }


def _serialize_order(order) -> dict[str, object]:
    return {
        "order_id": order.order_id,
        "priority": order.priority.value,
        "deadline": order.deadline,
        "drop_mode": order.drop_mode.value,
        "recipient_availability": order.recipient_availability.value,
        "retry_count": order.retry_count,
        "late_penalty": order.late_penalty,
        "assigned_drone_id": order.assigned_drone_id,
        "zone_id": order.zone_id,
        "status": order.status,
        "package_weight": order.package_weight,
    }


def _serialize_sector(sector) -> dict[str, object]:
    return {
        "zone_id": sector.zone_id,
        "weather": sector.weather.value,
        "congestion_score": sector.congestion_score,
        "is_no_fly": sector.is_no_fly,
        "likely_failure": sector.likely_failure,
    }


def _serialize_station(station) -> dict[str, object]:
    return {
        "station_id": station.station_id,
        "zone_id": station.zone_id,
        "capacity": station.capacity,
        "occupied_slots": station.occupied_slots,
        "queue_size": station.queue_size,
    }


def _serialize_notice(notice) -> dict[str, object]:
    return {
        "notice_id": notice.notice_id,
        "zone_id": notice.zone_id,
        "message": notice.message,
        "severity": notice.severity.value,
    }


def _serialize_event(event) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "zone_id": event.zone_id,
        "summary": event.summary,
        "severity": event.severity.value,
        "active": event.active,
    }


def _build_warnings(state) -> list[str]:
    warnings: list[str] = []
    for drone in sorted(state.fleet, key=lambda item: item.drone_id):
        if drone.battery <= 25:
            warnings.append(f"{drone.drone_id} battery is low ({drone.battery}).")
    for order in sorted(state.orders, key=_order_sort_key):
        if order.order_id in state.resolved_order_ids:
            continue
        if order.deadline <= 2:
            warnings.append(f"{order.order_id} is close to deadline ({order.deadline}).")
    return warnings[:6]


def _order_sort_key(order) -> tuple[int, int, str]:
    priority_rank = {
        OrderPriority.MEDICAL: 0,
        OrderPriority.URGENT: 1,
        OrderPriority.NORMAL: 2,
    }
    return (priority_rank[order.priority], order.deadline, order.order_id)
