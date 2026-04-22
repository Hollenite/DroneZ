from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment
from urbanair.policies.baseline import HeuristicPolicy, NaivePolicy


def test_policies_return_valid_action_shapes() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")

    for policy in (NaivePolicy(), HeuristicPolicy()):
        action = policy.choose_action(observation, info)
        assert set(action.keys()) == {"action", "params"}
        assert isinstance(action["params"], dict)


def test_policies_never_select_relay_for_assignment() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")

    for policy in (NaivePolicy(), HeuristicPolicy()):
        action = policy.choose_action(observation, info)
        if action["action"] == "assign_delivery":
            relay_ids = {drone["drone_id"] for drone in observation["fleet"] if drone["drone_type"] == "relay"}
            assert action["params"]["drone_id"] not in relay_ids


def test_heuristic_prefers_charging_for_low_battery_drone() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")
    low_battery_drone = next(drone for drone in observation["fleet"] if drone["drone_type"] != "relay")
    low_battery_drone["battery"] = 20
    low_battery_drone["health_risk"] = "high"

    action = HeuristicPolicy().choose_action(observation, info)

    assert action["action"] == "return_to_charge"
    assert action["params"]["drone_id"] == low_battery_drone["drone_id"]


def test_policies_are_deterministic_for_same_observation() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")

    assert NaivePolicy().choose_action(observation, info) == NaivePolicy().choose_action(observation, info)
    assert HeuristicPolicy().choose_action(observation, info) == HeuristicPolicy().choose_action(observation, info)
