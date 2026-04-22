from __future__ import annotations

from ..enums import DroneStatus


def evaluate_termination(state, done: bool | None = None) -> tuple[bool, str]:
    unresolved_orders = [
        order
        for order in state.orders
        if order.order_id not in state.resolved_order_ids and order.status != "canceled"
    ]
    no_viable_drones = not any(
        drone.status != DroneStatus.OFFLINE and drone.battery > 10 for drone in state.fleet
    )

    if not unresolved_orders:
        return True, "all_orders_resolved"
    if state.tick >= state.task_config.horizon:
        return True, "horizon_reached"
    if no_viable_drones:
        return True, "no_viable_drones"
    if done:
        return True, "terminated"
    return False, "ongoing"
