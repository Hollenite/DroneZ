from __future__ import annotations

from urbanair.eval.benchmark import benchmark_task_sweep, compare_demo_policies, run_episode
from urbanair.policies.baseline import HeuristicPolicy, ImprovedPolicy, NaivePolicy, RandomPolicy


def test_run_episode_returns_expected_summary_fields() -> None:
    result = run_episode(NaivePolicy(), "easy")
    summary = result["summary"]

    assert summary.task_id == "easy"
    assert summary.policy_id == "naive"
    assert summary.steps_completed >= 0
    assert isinstance(result["trace"], list)


def test_demo_comparison_returns_aligned_traces() -> None:
    comparison = compare_demo_policies(NaivePolicy(), HeuristicPolicy())

    assert comparison["task_id"] == "demo"
    assert comparison["baseline_policy_id"] == "naive"
    assert comparison["candidate_policy_id"] == "heuristic"
    assert len(comparison["per_step_comparison"]) > 0


def test_task_sweep_covers_expected_tasks() -> None:
    results = benchmark_task_sweep([RandomPolicy(), NaivePolicy(), HeuristicPolicy(), ImprovedPolicy()])

    assert results["tasks"] == ["easy", "medium", "hard", "demo"]
    assert set(results["policy_results"].keys()) == {"random", "naive", "heuristic", "improved"}


def test_benchmark_runs_are_reproducible() -> None:
    first = benchmark_task_sweep([NaivePolicy(), HeuristicPolicy(), ImprovedPolicy()])
    second = benchmark_task_sweep([NaivePolicy(), HeuristicPolicy(), ImprovedPolicy()])

    assert first["ranking"] == second["ranking"]
    assert first["aggregate"] == second["aggregate"]


def test_run_episode_trace_captures_new_action_types() -> None:
    result = run_episode(HeuristicPolicy(), "demo", max_steps=10)

    assert any(step["action"]["action"] in {"attempt_delivery", "reserve_charger", "fallback_to_locker", "hold_fleet", "resume_operations", "reroute", "swap_assignments"} for step in result["trace"])


def test_run_episode_never_stalls_on_invalid_actions() -> None:
    result = run_episode(RandomPolicy(), "demo", max_actions=20, max_invalid_actions=3)
    summary = result["summary"]

    assert summary.actions_taken <= 20
    assert summary.done_reason in {"all_orders_resolved", "horizon_reached", "no_viable_drones", "invalid_action_cap_reached", "action_cap_reached", "runner_action_cap_reached", "runner_step_cap_reached", "runner_invalid_action_cap_reached"}
