from __future__ import annotations

from ..enums import RecipientAvailability
from ..models import OrderState, PolicyNotice, ScriptedEvent
from .city import get_sector
from .orders import insert_urgent_order


def apply_scripted_events(scripted_events: list[ScriptedEvent], tick: int, sectors, policy_notices, orders, next_order_index: int) -> tuple[list[str], int, list[str]]:
    events: list[str] = []
    triggered: list[str] = []

    for scripted_event in scripted_events:
        if scripted_event.tick != tick:
            continue

        if scripted_event.type == "urgent_insertion":
            zone_id = scripted_event.params.get("zone_id") or _first_open_zone(sectors)
            new_order = insert_urgent_order(next_order_index, zone_id)
            orders.append(new_order)
            next_order_index += 1
            events.append(f"Scripted urgent order {new_order.order_id} inserted for {zone_id}.")
            triggered.append("urgent_insertion")

        elif scripted_event.type == "no_fly_shift":
            previously_blocked = next((sector for sector in sectors if sector.zone_id != "hub" and sector.is_no_fly), None)
            if previously_blocked is not None:
                previously_blocked.is_no_fly = False
                policy_notices.append(
                    PolicyNotice(
                        notice_id=f"scripted-no-fly-clear-{tick}-{previously_blocked.zone_id}",
                        zone_id=previously_blocked.zone_id,
                        message=f"Scripted no-fly restriction cleared in {previously_blocked.zone_id}.",
                    )
                )
                events.append(f"Scripted no-fly restriction cleared in {previously_blocked.zone_id}.")

            zone_id = scripted_event.params.get("zone_id") or _first_open_zone(sectors, exclude_zone_id=previously_blocked.zone_id if previously_blocked else None)
            sector = get_sector(sectors, zone_id)
            if sector is not None:
                sector.is_no_fly = True
                policy_notices.append(
                    PolicyNotice(
                        notice_id=f"scripted-no-fly-activate-{tick}-{sector.zone_id}",
                        zone_id=sector.zone_id,
                        message=f"Scripted no-fly restriction activated in {sector.zone_id}.",
                    )
                )
                events.append(f"Scripted no-fly restriction activated in {sector.zone_id}.")
                if previously_blocked is None and sector.zone_id == zone_id:
                    events.append(f"Scripted no-fly zone anchored at {sector.zone_id}.")
                elif previously_blocked is not None:
                    events.append(f"Scripted no-fly restriction shifted from {previously_blocked.zone_id} to {sector.zone_id}.")
                else:
                    events.append(f"Scripted no-fly restriction shifted to {sector.zone_id}.")

                triggered.append("no_fly_shift")

        elif scripted_event.type == "failed_drop":
            target_order = _first_active_urgent_order(orders)
            if target_order is not None:
                target_order.recipient_availability = RecipientAvailability.UNAVAILABLE
                events.append(f"Scripted disruption made {target_order.order_id} unavailable for doorstep delivery.")
                triggered.append("failed_drop")

    return events, next_order_index, triggered


def _first_open_zone(sectors, exclude_zone_id: str | None = None) -> str:
    for sector in sectors:
        if sector.zone_id == "hub" or sector.zone_id == exclude_zone_id:
            continue
        if not sector.is_no_fly:
            return sector.zone_id
    for sector in sectors:
        if sector.zone_id != "hub" and sector.zone_id != exclude_zone_id:
            return sector.zone_id
    return "hub"


def _first_active_urgent_order(orders: list[OrderState]) -> OrderState | None:
    for order in orders:
        if order.priority.value == "urgent" and order.status not in {"delivered", "failed", "canceled"}:
            return order
    return None
