from __future__ import annotations

from urbanair.sim.engine import SimulatorEngine


def test_demo_scripted_events_fire_on_expected_ticks() -> None:
    engine = SimulatorEngine.from_repo_configs()
    state = engine.reset("demo")

    triggered_by_tick: dict[int, list[str]] = {}
    for _ in range(8):
        result = engine.step(state, None)
        if result.info["triggered_scripted_events"]:
            triggered_by_tick[state.tick] = result.info["triggered_scripted_events"]

    assert triggered_by_tick[4] == ["urgent_insertion"]
    assert triggered_by_tick[6] == ["no_fly_shift"]
    assert triggered_by_tick[8] == ["failed_drop"]


def test_demo_scripted_events_are_reproducible() -> None:
    engine = SimulatorEngine.from_repo_configs()

    first = engine.reset("demo")
    second = engine.reset("demo")

    first_sequence = []
    second_sequence = []
    for _ in range(8):
        first_sequence.append(engine.step(first, None).info["triggered_scripted_events"])
        second_sequence.append(engine.step(second, None).info["triggered_scripted_events"])

    assert first_sequence == second_sequence


def test_demo_no_fly_shift_does_not_duplicate_generic_toggle() -> None:
    engine = SimulatorEngine.from_repo_configs()
    state = engine.reset("demo")

    for _ in range(6):
        result = engine.step(state, None)

    assert result.info["triggered_scripted_events"] == ["no_fly_shift"]
    assert sum(1 for sector in state.sectors if sector.zone_id != "hub" and sector.is_no_fly) == 1
