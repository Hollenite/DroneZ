from __future__ import annotations

from ..enums import DropMode, OrderPriority, RecipientAvailability
from ..models import OrderState, SectorState, TaskConfig
from .city import non_hub_zones


def build_orders(task_config: TaskConfig, sectors: list[SectorState], rng) -> list[OrderState]:
    zones = non_hub_zones(sectors)
    orders: list[OrderState] = []

    for index in range(task_config.initial_orders):
        priority = OrderPriority.NORMAL
        if index < task_config.urgent_orders:
            priority = OrderPriority.URGENT if index % 2 == 0 else OrderPriority.MEDICAL

        availability = RecipientAvailability.AVAILABLE
        if task_config.difficulty.value in {"medium", "hard", "demo"} and rng.random() < 0.35:
            availability = RecipientAvailability.UNCERTAIN

        zone = zones[index % len(zones)]
        deadline = max(2, task_config.horizon // 3 + rng.randint(-2, 3) - index // 2)
        orders.append(
            OrderState(
                order_id=f"O{index + 1}",
                priority=priority,
                deadline=deadline,
                drop_mode=DropMode.DOORSTEP if index % 3 else DropMode.HANDOFF,
                recipient_availability=availability,
                retry_count=0,
                late_penalty="critical" if priority == OrderPriority.MEDICAL else "high" if priority == OrderPriority.URGENT else "medium",
                assigned_drone_id=None,
                zone_id=zone,
                status="queued",
                package_weight=1 if priority == OrderPriority.NORMAL else 2,
            )
        )

    return orders


def tick_order_deadlines(orders: list[OrderState], resolved_order_ids: set[str]) -> list[str]:
    events: list[str] = []
    for order in orders:
        if order.order_id in resolved_order_ids:
            continue
        order.deadline = max(0, order.deadline - 1)
        if order.deadline == 0:
            events.append(f"{order.order_id} reached its delivery deadline.")
    return events


def update_customer_availability(orders: list[OrderState], resolved_order_ids: set[str], hidden_no_show_zones: set[str], order_zone_map: dict[str, str], rng) -> list[str]:
    events: list[str] = []
    for order in orders:
        if order.order_id in resolved_order_ids:
            continue
        zone_id = order_zone_map[order.order_id]
        if zone_id in hidden_no_show_zones and rng.random() < 0.35:
            if order.recipient_availability != RecipientAvailability.UNAVAILABLE:
                order.recipient_availability = RecipientAvailability.UNAVAILABLE
                events.append(f"{order.order_id} recipient became unavailable in {zone_id}.")
        elif order.recipient_availability == RecipientAvailability.UNCERTAIN and rng.random() < 0.45:
            order.recipient_availability = RecipientAvailability.AVAILABLE
            events.append(f"{order.order_id} recipient availability clarified.")
    return events


def insert_urgent_order(next_index: int, zone_id: str) -> OrderState:
    return OrderState(
        order_id=f"O{next_index}",
        priority=OrderPriority.URGENT,
        deadline=3,
        drop_mode=DropMode.HANDOFF,
        recipient_availability=RecipientAvailability.AVAILABLE,
        retry_count=0,
        late_penalty="critical",
        assigned_drone_id=None,
        zone_id=zone_id,
        status="queued",
        package_weight=2,
    )
