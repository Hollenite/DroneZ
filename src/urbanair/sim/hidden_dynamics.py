from __future__ import annotations

from dataclasses import dataclass

from ..models import SectorState, TaskConfig


@dataclass
class HiddenState:
    sector_weather_bias: dict[str, float]
    failure_prone_zones: set[str]
    no_show_zones: set[str]
    charging_growth_factor: float
    downstream_delay_pressure: dict[str, float]


def build_hidden_state(task_config: TaskConfig, sectors: list[SectorState], rng) -> HiddenState:
    non_hub = [sector.zone_id for sector in sectors if sector.zone_id != "hub"]
    weather_bias = {zone_id: round(rng.uniform(0.0, task_config.hidden_factor_intensity), 2) for zone_id in non_hub}
    failure_count = max(1, len(non_hub) // 3)
    failure_prone = set(rng.sample(non_hub, k=min(failure_count, len(non_hub))))
    no_show_count = max(1, len(non_hub) // 3)
    no_show = set(rng.sample(non_hub, k=min(no_show_count, len(non_hub))))
    downstream_delay_pressure = {zone_id: round(rng.uniform(0.0, 1.0), 2) for zone_id in non_hub}
    return HiddenState(
        sector_weather_bias=weather_bias,
        failure_prone_zones=failure_prone,
        no_show_zones=no_show,
        charging_growth_factor=round(0.15 + task_config.hidden_factor_intensity * 0.5, 2),
        downstream_delay_pressure=downstream_delay_pressure,
    )


def update_hidden_state(hidden_state: HiddenState, sectors: list[SectorState]) -> None:
    for sector in sectors:
        if sector.zone_id == "hub":
            continue
        sector.likely_failure = sector.zone_id in hidden_state.failure_prone_zones
        if sector.zone_id in hidden_state.failure_prone_zones and sector.congestion_score > 0.55:
            hidden_state.downstream_delay_pressure[sector.zone_id] = min(
                1.0,
                hidden_state.downstream_delay_pressure[sector.zone_id] + 0.1,
            )
