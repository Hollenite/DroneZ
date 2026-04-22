from __future__ import annotations

from ..enums import RecipientAvailability, WeatherSeverity
from ..models import ChargingStationState, EmergencyEvent, OrderState, PolicyNotice, SectorState, TaskConfig
from .orders import insert_urgent_order

_WEATHER_ORDER = [
    WeatherSeverity.CLEAR,
    WeatherSeverity.MODERATE_WIND,
    WeatherSeverity.HEAVY_RAIN,
    WeatherSeverity.STORM,
]


def evolve_disruptions(
    task_config: TaskConfig,
    tick: int,
    sectors: list[SectorState],
    charging_stations: list[ChargingStationState],
    policy_notices: list[PolicyNotice],
    emergencies: list[EmergencyEvent],
    orders: list[OrderState],
    next_order_index: int,
    hidden_weather_bias: dict[str, float],
    rng,
) -> tuple[list[str], int]:
    events: list[str] = []

    if task_config.dynamic_events.weather:
        for sector in sectors:
            if sector.zone_id == "hub":
                continue
            bias = hidden_weather_bias.get(sector.zone_id, 0.0)
            if rng.random() < 0.15 + bias:
                sector.weather = _next_weather(sector.weather)
                events.append(f"Weather shifted in {sector.zone_id} to {sector.weather.value}.")

    if task_config.dynamic_events.no_fly and tick in {3, 6, 8}:
        movable = [sector for sector in sectors if sector.zone_id != "hub"]
        sector = movable[tick % len(movable)]
        sector.is_no_fly = not sector.is_no_fly
        state = "activated" if sector.is_no_fly else "cleared"
        policy_notices.append(
            PolicyNotice(
                notice_id=f"policy-{tick}-{sector.zone_id}",
                zone_id=sector.zone_id,
                message=f"Temporary restriction {state} in {sector.zone_id}.",
            )
        )
        events.append(f"No-fly restriction {state} in {sector.zone_id}.")

    if task_config.dynamic_events.charging_congestion and tick % 4 == 0:
        station = charging_stations[-1]
        station.queue_size += 1
        events.append(f"Charging congestion increased at {station.station_id}.")

    if task_config.dynamic_events.emergency and tick in {4, 7}:
        inactive = next((event for event in emergencies if not event.active), None)
        if inactive is not None:
            inactive.active = True
            events.append(f"Emergency lane activated in {inactive.zone_id}.")

    if task_config.difficulty.value in {"hard", "demo"} and tick in {4, 8}:
        zone_id = next((sector.zone_id for sector in sectors if sector.zone_id != "hub" and not sector.is_no_fly), sectors[1].zone_id)
        new_order = insert_urgent_order(next_order_index, zone_id)
        orders.append(new_order)
        next_order_index += 1
        events.append(f"Urgent order {new_order.order_id} inserted for {zone_id}.")

    if task_config.deterministic_demo:
        for scripted_event in task_config.scripted_events:
            if scripted_event.tick == tick and scripted_event.type == "failed_drop":
                target = next((order for order in orders if order.priority.value == "urgent" and order.status not in {"delivered", "failed"}), None)
                if target is not None:
                    target.recipient_availability = RecipientAvailability.UNAVAILABLE
                    events.append(f"Scripted disruption made {target.order_id} unavailable for doorstep delivery.")

    return events, next_order_index


def _next_weather(weather: WeatherSeverity) -> WeatherSeverity:
    index = _WEATHER_ORDER.index(weather)
    return _WEATHER_ORDER[min(len(_WEATHER_ORDER) - 1, index + 1)]
