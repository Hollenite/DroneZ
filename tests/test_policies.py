from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment
from urbanair.policies.baseline import HeuristicPolicy, ImprovedPolicy, NaivePolicy, RandomPolicy


def test_policies_return_valid_action_shapes() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")

    for policy in (RandomPolicy(), NaivePolicy(), HeuristicPolicy(), ImprovedPolicy()):
        action = policy.choose_action(observation, info)
        assert set(action.keys()) == {"action", "params"}
        assert isinstance(action["params"], dict)


def test_policies_never_select_relay_for_assignment() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")

    for policy in (RandomPolicy(), NaivePolicy(), HeuristicPolicy(), ImprovedPolicy()):
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

    assert action["action"] in {"reserve_charger", "return_to_charge"}
    assert action["params"]["drone_id"] == low_battery_drone["drone_id"]


def test_policies_choose_explicit_delivery_attempt_when_ready() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")
    drone = next(drone for drone in observation["fleet"] if drone["drone_type"] != "relay")
    drone["assigned_order_id"] = "O1"
    drone["eta"] = 0
    action = HeuristicPolicy().choose_action(observation, info)

    assert action["action"] == "attempt_delivery"
    assert action["params"]["drone_id"] == drone["drone_id"]


def test_heuristic_chooses_reroute_for_no_fly_target() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")
    drone = next(drone for drone in observation["fleet"] if drone["drone_type"] != "relay")
    drone["status"] = "assigned"
    drone["assigned_order_id"] = "O1"
    drone["target_zone"] = "Z1"
    drone["eta"] = 2
    observation["city"]["sectors"][1]["is_no_fly"] = True
    observation["city"]["sectors"][1]["zone_id"] = "Z1"

    action = HeuristicPolicy().choose_action(observation, info)
    assert action["action"] == "reroute"


def test_improved_policy_beats_naive_on_demo() -> None:
    from urbanair.eval.benchmark import run_episode

    naive = run_episode(NaivePolicy(), "demo")["summary"]
    improved = run_episode(ImprovedPolicy(), "demo")["summary"]

    assert improved.total_reward > naive.total_reward
    assert improved.invalid_action_count <= naive.invalid_action_count


def test_improved_policy_beats_heuristic_on_demo_score() -> None:
    from urbanair.eval.benchmark import run_episode

    heuristic = run_episode(HeuristicPolicy(), "demo")["summary"]
    improved = run_episode(ImprovedPolicy(), "demo")["summary"]

    assert improved.total_reward > heuristic.total_reward
    assert improved.normalized_score > heuristic.normalized_score


def test_improved_policy_preserves_demo_safety() -> None:
    from urbanair.eval.benchmark import run_episode

    improved = run_episode(ImprovedPolicy(), "demo")["summary"]

    assert improved.safety_violations == 0
    assert improved.invalid_action_count == 0


def test_policies_are_deterministic_for_same_observation() -> None:
    env = DroneZEnvironment()
    observation, info = env.reset("easy")

    assert NaivePolicy().choose_action(observation, info) == NaivePolicy().choose_action(observation, info)
    assert HeuristicPolicy().choose_action(observation, info) == HeuristicPolicy().choose_action(observation, info)
    assert ImprovedPolicy().choose_action(observation, info) == ImprovedPolicy().choose_action(observation, info)
