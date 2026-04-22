from __future__ import annotations

from math import ceil

from ..enums import DroneStatus, DroneType, RiskLevel, WeatherSeverity
from ..models import ChargingStationState, DroneState, FleetProfile, SectorState, TaskConfig


_BATTERY_DRAIN_BY_TYPE = {
    DroneType.FAST_LIGHT: 8,
    DroneType.HEAVY_CARRIER: 6,
    DroneType.LONG_RANGE_SENSITIVE: 7,
    DroneType.RELAY: 4,
}


def build_fleet(task_config: TaskConfig, fleet_profiles: dict[DroneType, FleetProfile], rng) -> list[DroneState]:
    fleet: list[DroneState] = []

    for drone_type, count in task_config.fleet_counts.items():
        profile = fleet_profiles[drone_type]
        for index in range(count):
            drone_id = f"{drone_type.value[:2].upper()}-{index + 1}"
            battery = 100 if task_config.deterministic_demo else rng.randint(72, 100)
            fleet.append(
                DroneState(
                    drone_id=drone_id,
                    drone_type=drone_type,
                    status=DroneStatus.IDLE,
                    battery=battery,
                    payload_capacity=profile.payload_capacity,
                    current_zone="hub",
                    assigned_order_id=None,
                    eta=None,
                    health_risk=RiskLevel.LOW,
                    communication_strength="boosted" if drone_type == DroneType.RELAY else "strong",
                )
            )

    return fleet


def estimate_eta(drone: DroneState, destination_zone: str, sectors: list[SectorState], corridor: str | None = None) -> int:
    sector = next((item for item in sectors if item.zone_id == destination_zone), None)
    congestion = sector.congestion_score if sector else 0.0
    weather_penalty = 1 if sector and sector.weather in {WeatherSeverity.HEAVY_RAIN, WeatherSeverity.STORM} else 0
    base = 1 + weather_penalty + ceil(congestion * 2)
    if drone.drone_type == DroneType.HEAVY_CARRIER:
        base += 1
    if drone.drone_type == DroneType.FAST_LIGHT:
        base = max(1, base - 1)

    corridor = corridor or drone.active_corridor or "direct"
    if corridor == "weather_avoid":
        base += 1
        if weather_penalty:
            base = max(1, base - 1)
    elif corridor == "congestion_avoid":
        base += 1
        if congestion >= 0.3:
            base = max(1, base - 1)
    elif corridor == "safe":
        base += 1
        if sector and not sector.is_no_fly and not sector.operations_paused:
            base = max(1, base - 1)
    return base


def assign_order(drone: DroneState, order_id: str, destination_zone: str, sectors: list[SectorState], corridor: str | None = None) -> list[str]:
    drone.assigned_order_id = order_id
    drone.status = DroneStatus.ASSIGNED
    drone.target_zone = destination_zone
    drone.active_corridor = corridor or "direct"
    drone.flight_path = [drone.current_zone, drone.active_corridor, destination_zone]
    drone.eta = estimate_eta(drone, destination_zone, sectors, drone.active_corridor)
    return [f"{drone.drone_id} assigned to {order_id} via {drone.active_corridor} with ETA {drone.eta}."]


def send_to_charge(drone: DroneState, station: ChargingStationState, reserve_only: bool = False) -> list[str]:
    events: list[str] = []
    if drone.reserved_station_id and drone.reserved_station_id != station.station_id:
        events.append(f"{drone.drone_id} switched charging reservation to {station.station_id}.")
    drone.reserved_station_id = station.station_id
    if drone.drone_id not in station.reserved_drone_ids:
        station.reserved_drone_ids.append(drone.drone_id)

    if reserve_only:
        drone.hold_reason = "charger_reserved"
        return events + [f"{drone.drone_id} reserved charging at {station.station_id}."]

    if station.occupied_slots < station.capacity:
        station.occupied_slots += 1
        drone.status = DroneStatus.CHARGING
        drone.hold_reason = None
        drone.eta = None
        return events + [f"{drone.drone_id} started charging at {station.station_id}."]

    station.queue_size += 1
    drone.status = DroneStatus.HOLDING
    drone.hold_reason = "charging_queue"
    drone.eta = None
    return events + [f"{drone.drone_id} queued for charging at {station.station_id}."]


def advance_fleet_tick(fleet: list[DroneState], sectors: list[SectorState]) -> list[str]:
    events: list[str] = []

    for drone in fleet:
        if drone.status in {DroneStatus.ASSIGNED, DroneStatus.IN_FLIGHT} and drone.assigned_order_id:
            drone.status = DroneStatus.IN_FLIGHT
            drone.battery = max(0, drone.battery - _BATTERY_DRAIN_BY_TYPE[drone.drone_type])
            if drone.eta is None:
                drone.eta = 1
            else:
                drone.eta = max(0, drone.eta - 1)

            if drone.eta == 0 and drone.current_zone == "hub":
                target_zone = drone.target_zone or _assigned_zone_hint(drone.assigned_order_id, sectors)
                if target_zone:
                    drone.current_zone = target_zone
                    events.append(f"{drone.drone_id} arrived in {target_zone} for {drone.assigned_order_id}.")

            drone.total_flight_ticks += 1

        elif drone.status == DroneStatus.CHARGING:
            drone.battery = min(100, drone.battery + 18)
            if drone.battery >= 92:
                drone.status = DroneStatus.IDLE
                drone.hold_reason = None
                drone.reserved_station_id = None
                events.append(f"{drone.drone_id} finished charging.")

        if drone.battery <= 12:
            drone.health_risk = RiskLevel.CRITICAL
            events.append(f"{drone.drone_id} entered critical battery state.")
        elif drone.battery <= 25:
            drone.health_risk = RiskLevel.HIGH
        elif drone.battery <= 45:
            drone.health_risk = RiskLevel.MEDIUM
        else:
            drone.health_risk = RiskLevel.LOW

    return events


def apply_relay_effect(fleet: list[DroneState], weak_zone_ids: set[str]) -> None:
    relay_present = any(drone.drone_type == DroneType.RELAY and drone.status != DroneStatus.OFFLINE for drone in fleet)
    for drone in fleet:
        if drone.drone_type == DroneType.RELAY:
            continue
        if relay_present and drone.current_zone in weak_zone_ids:
            drone.communication_strength = "boosted"
        elif drone.current_zone in weak_zone_ids:
            drone.communication_strength = "weak"
        else:
            drone.communication_strength = "strong"


def _assigned_zone_hint(order_id: str, sectors: list[SectorState]) -> str | None:
    if not order_id:
        return None
    non_hub = [sector.zone_id for sector in sectors if sector.zone_id != "hub"]
    if not non_hub:
        return None
    numeric_part = sum(ord(char) for char in order_id)
    return non_hub[numeric_part % len(non_hub)]
