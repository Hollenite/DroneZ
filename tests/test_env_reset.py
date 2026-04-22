from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment


def test_reset_returns_structured_observation_and_summary() -> None:
    env = DroneZEnvironment()

    observation, info = env.reset("easy")

    assert observation["task_id"] == "easy"
    assert observation["step"] == 0
    assert "fleet" in observation
    assert "orders" in observation
    assert "city" in observation
    assert isinstance(observation["summary"], str)
    assert "TIME_STEP: 0 / 20" in observation["summary"]
    assert info["task_id"] == "easy"
    assert info["invalid_action"] is False


def test_reset_is_deterministic_for_same_task() -> None:
    env = DroneZEnvironment()

    first_observation, first_info = env.reset("easy")
    second_observation, second_info = env.reset("easy")

    assert first_observation["fleet"] == second_observation["fleet"]
    assert first_observation["orders"] == second_observation["orders"]
    assert first_observation["city"] == second_observation["city"]
    assert first_info["reward_breakdown"] == second_info["reward_breakdown"]
