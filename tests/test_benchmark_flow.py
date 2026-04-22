from __future__ import annotations

from urbanair.eval.benchmark import benchmark_task_sweep, compare_demo_policies, run_episode
from urbanair.policies.baseline import HeuristicPolicy, NaivePolicy


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
    results = benchmark_task_sweep([NaivePolicy(), HeuristicPolicy()])

    assert results["tasks"] == ["easy", "medium", "hard"]
    assert set(results["policy_results"].keys()) == {"naive", "heuristic"}


def test_benchmark_runs_are_reproducible() -> None:
    first = benchmark_task_sweep([NaivePolicy(), HeuristicPolicy()])
    second = benchmark_task_sweep([NaivePolicy(), HeuristicPolicy()])

    assert first["ranking"] == second["ranking"]
    assert first["aggregate"] == second["aggregate"]
