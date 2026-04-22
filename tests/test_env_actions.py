from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment


def test_valid_assign_delivery_advances_tick() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    drone = next(item for item in observation["fleet"] if item["drone_type"] != "relay" and item["status"] == "idle")
    order = next(item for item in observation["orders"] if item["assigned_drone_id"] is None)

    next_observation, reward, done, info = env.step(
        {
            "action": "assign_delivery",
            "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]},
        }
    )

    assert next_observation["step"] == 1
    assert reward != 0.0
    assert done is False
    assert info["invalid_action"] is False
    assert next_observation["last_action"]["action"] == "assign_delivery"


def test_invalid_action_returns_penalty_without_advancing_tick() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    next_observation, reward, done, info = env.step(
        {
            "action": "assign_delivery",
            "params": {"drone_id": "missing", "order_id": "O1"},
        }
    )

    assert observation["step"] == 0
    assert next_observation["step"] == 0
    assert reward == -5.0
    assert done is False
    assert info["invalid_action"] is True
    assert info["error_code"] == "unknown_drone"


def test_unsupported_known_action_fails_explicitly() -> None:
    env = DroneZEnvironment()
    env.reset("easy")

    _, reward, done, info = env.step(
        {
            "action": "delay_order",
            "params": {"order_id": "O1"},
        }
    )

    assert reward == -5.0
    assert done is False
    assert info["invalid_action"] is True
    assert info["error_code"] == "unsupported_action"
