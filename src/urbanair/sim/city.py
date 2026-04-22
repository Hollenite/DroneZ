from __future__ import annotations

from typing import Iterable

from ..enums import Difficulty, RiskLevel, WeatherSeverity
from ..models import ChargingStationState, EmergencyEvent, PolicyNotice, SectorState, TaskConfig

_ZONE_COUNT_BY_DIFFICULTY = {
    Difficulty.EASY: 4,
    Difficulty.MEDIUM: 5,
    Difficulty.HARD: 6,
    Difficulty.DEMO: 5,
}


def build_sector_ids(task_config: TaskConfig) -> list[str]:
    zone_count = _ZONE_COUNT_BY_DIFFICULTY[task_config.difficulty]
    return ["hub", *[f"Z{index}" for index in range(1, zone_count + 1)]]


def build_city(task_config: TaskConfig, rng) -> tuple[list[SectorState], list[ChargingStationState], list[PolicyNotice], list[EmergencyEvent]]:
    sector_ids = build_sector_ids(task_config)
    sectors: list[SectorState] = []

    for zone_id in sector_ids:
        if zone_id == "hub":
            sectors.append(
                SectorState(
                    zone_id=zone_id,
                    weather=WeatherSeverity.CLEAR,
                    congestion_score=0.0,
                    is_no_fly=False,
                    likely_failure=False,
                )
            )
            continue

        sectors.append(
            SectorState(
                zone_id=zone_id,
                weather=WeatherSeverity.CLEAR,
                congestion_score=round(rng.uniform(0.1, 0.45), 2),
                is_no_fly=False,
                likely_failure=False,
            )
        )

    if task_config.difficulty in {Difficulty.HARD, Difficulty.DEMO} and len(sectors) > 2:
        sectors[2].is_no_fly = True

    charging_stations = [
        ChargingStationState(station_id="C_HUB", zone_id="hub", capacity=3, occupied_slots=1, queue_size=0),
        ChargingStationState(
            station_id="C_FIELD",
            zone_id=sector_ids[-1],
            capacity=2,
            occupied_slots=1 if task_config.dynamic_events.charging_congestion else 0,
            queue_size=1 if task_config.dynamic_events.charging_congestion else 0,
        ),
    ]

    policy_notices: list[PolicyNotice] = []
    if task_config.dynamic_events.no_fly:
        policy_notices.append(
            PolicyNotice(
                notice_id="policy-initial-no-fly",
                zone_id=sector_ids[-1],
                message=f"Restricted operations in {sector_ids[-1]} until conditions improve.",
                severity=RiskLevel.MEDIUM,
            )
        )

    emergencies: list[EmergencyEvent] = []
    if task_config.dynamic_events.emergency and task_config.difficulty in {Difficulty.HARD, Difficulty.DEMO}:
        emergencies.append(
            EmergencyEvent(
                event_id="emergency-standby",
                zone_id=sector_ids[1],
                summary="Emergency lane reserved for potential priority insertion.",
                severity=RiskLevel.HIGH,
                active=False,
            )
        )

    return sectors, charging_stations, policy_notices, emergencies


def non_hub_zones(sectors: Iterable[SectorState]) -> list[str]:
    return [sector.zone_id for sector in sectors if sector.zone_id != "hub"]


def get_sector(sectors: Iterable[SectorState], zone_id: str) -> SectorState | None:
    for sector in sectors:
        if sector.zone_id == zone_id:
            return sector
    return None
