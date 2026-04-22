from __future__ import annotations

from urbanair.sim.disruptions import evolve_disruptions
from urbanair.sim.engine import SimulatorEngine
from urbanair.utils.seeding import create_seed_bundle, derive_seed


def test_disruptions_are_reproducible_for_same_seed() -> None:
    engine = SimulatorEngine.from_repo_configs()
    first = engine.reset("demo")
    second = engine.reset("demo")

    first_events, first_next = evolve_disruptions(
        first.task_config,
        4,
        first.sectors,
        first.charging_stations,
        first.policy_notices,
        first.emergency_events,
        first.orders,
        first.next_order_index,
        first.hidden_state.sector_weather_bias,
        create_seed_bundle(derive_seed(first.seed_bundle.seed, 64)).rng,
    )
    second_events, second_next = evolve_disruptions(
        second.task_config,
        4,
        second.sectors,
        second.charging_stations,
        second.policy_notices,
        second.emergency_events,
        second.orders,
        second.next_order_index,
        second.hidden_state.sector_weather_bias,
        create_seed_bundle(derive_seed(second.seed_bundle.seed, 64)).rng,
    )

    assert first_events == second_events
    assert first_next == second_next
    assert [sector.model_dump() for sector in first.sectors] == [sector.model_dump() for sector in second.sectors]
