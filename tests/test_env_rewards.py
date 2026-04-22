from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment


def test_valid_step_exposes_reward_breakdown() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    drone = next(item for item in observation["fleet"] if item["drone_type"] != "relay" and item["status"] == "idle")
    order = next(item for item in observation["orders"] if item["assigned_drone_id"] is None)

    _, reward, _, info = env.step(
        {
            "action": "assign_delivery",
            "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]},
        }
    )

    assert reward == info["reward_breakdown"]["total"]
    assert isinstance(info["reward_breakdown"]["positive"], dict)
    assert isinstance(info["reward_breakdown"]["negative"], dict)


def test_invalid_action_uses_configured_penalty() -> None:
    env = DroneZEnvironment()
    env.reset("easy")

    _, reward, _, info = env.step({"action": "unknown_action", "params": {}})

    assert reward == -5.0
    assert info["reward_breakdown"]["negative"]["invalid_action"] == -5.0
    assert info["cumulative_reward"]["negative"]["invalid_action"] == -5.0
