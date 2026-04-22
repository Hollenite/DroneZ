from __future__ import annotations

from urbanair.sim.engine import SimulatorEngine


def test_reset_is_deterministic_for_same_task() -> None:
    engine = SimulatorEngine.from_repo_configs()

    first = engine.reset("easy")
    second = engine.reset("easy")

    assert [drone.model_dump() for drone in first.fleet] == [drone.model_dump() for drone in second.fleet]
    assert [order.model_dump() for order in first.orders] == [order.model_dump() for order in second.orders]
    assert [sector.model_dump() for sector in first.sectors] == [sector.model_dump() for sector in second.sectors]
    assert first.hidden_state == second.hidden_state


def test_reset_builds_expected_world_components() -> None:
    engine = SimulatorEngine.from_repo_configs()
    state = engine.reset("demo")

    assert state.tick == 0
    assert len(state.fleet) == 4
    assert len(state.orders) >= state.task_config.initial_orders
    assert len(state.sectors) >= 5
    assert len(state.charging_stations) == 2
    assert state.hidden_state is not None
