from __future__ import annotations

from collections import defaultdict

from ..env.environment import DroneZEnvironment
from ..models import EpisodeSummary, RewardBreakdown
from ..policies.base import Policy
from .metrics import build_episode_summary


def run_episode(policy: Policy, task_id: str, max_steps: int | None = None) -> dict[str, object]:
    env = DroneZEnvironment(default_task_id=task_id)
    observation, info = env.reset(task_id)
    trace: list[dict[str, object]] = []
    action_counts: dict[str, int] = defaultdict(int)
    invalid_action_count = 0
    triggered_scripted_events: list[str] = []

    reward = 0.0
    done = info["done"]
    while not done and (max_steps is None or len(trace) < max_steps):
        action = policy.choose_action(observation, info)
        action_counts[action["action"]] += 1
        observation, reward, done, info = env.step(action)
        if info["invalid_action"]:
            invalid_action_count += 1
        triggered_scripted_events.extend(info.get("triggered_scripted_events", []))
        trace.append(
            {
                "tick": observation["step"],
                "action": action,
                "step_reward": reward,
                "cumulative_reward": info["cumulative_reward"]["total"],
                "pending_orders": sum(1 for order in observation["orders"] if order["status"] not in {"delivered", "canceled"}),
                "triggered_scripted_events": list(info.get("triggered_scripted_events", [])),
            }
        )

    orders = observation["orders"]
    summary = build_episode_summary(
        task_id=task_id,
        policy_id=policy.policy_id,
        seed=env.state.task_config.seed,
        cumulative_reward=RewardBreakdown.model_validate(info["cumulative_reward"]),
        steps_completed=observation["step"],
        completed_deliveries=sum(1 for order in orders if order["status"] == "delivered"),
        failed_deliveries=sum(1 for order in orders if order["status"] == "failed"),
        urgent_successes=sum(1 for order in orders if order["status"] == "delivered" and order["priority"] in {"urgent", "medical"}),
        invalid_action_count=invalid_action_count,
        deadline_miss_count=sum(1 for event in observation["recent_events"] if "deadline" in event.lower()),
        critical_battery_events=sum(1 for event in observation["recent_events"] if "critical battery" in event.lower()),
        action_counts=dict(action_counts),
        triggered_scripted_events=triggered_scripted_events,
    )
    return {"summary": summary, "trace": trace}


def compare_demo_policies(baseline_policy: Policy, candidate_policy: Policy) -> dict[str, object]:
    baseline_result = run_episode(baseline_policy, "demo")
    candidate_result = run_episode(candidate_policy, "demo")
    baseline_summary: EpisodeSummary = baseline_result["summary"]
    candidate_summary: EpisodeSummary = candidate_result["summary"]

    per_step_comparison = []
    for baseline_step, candidate_step in zip(baseline_result["trace"], candidate_result["trace"]):
        per_step_comparison.append(
            {
                "tick": baseline_step["tick"],
                "triggered_scripted_events": candidate_step["triggered_scripted_events"],
                "baseline_action": baseline_step["action"],
                "heuristic_action": candidate_step["action"],
                "baseline_step_reward": baseline_step["step_reward"],
                "heuristic_step_reward": candidate_step["step_reward"],
                "baseline_cumulative_reward": baseline_step["cumulative_reward"],
                "heuristic_cumulative_reward": candidate_step["cumulative_reward"],
                "baseline_pending_orders": baseline_step["pending_orders"],
                "heuristic_pending_orders": candidate_step["pending_orders"],
            }
        )

    return {
        "task_id": "demo",
        "seed": baseline_summary.seed,
        "baseline_policy_id": baseline_policy.policy_id,
        "candidate_policy_id": candidate_policy.policy_id,
        "baseline_summary": baseline_summary,
        "candidate_summary": candidate_summary,
        "reward_delta": candidate_summary.total_reward - baseline_summary.total_reward,
        "delivery_delta": candidate_summary.completed_deliveries - baseline_summary.completed_deliveries,
        "urgent_success_delta": candidate_summary.urgent_successes - baseline_summary.urgent_successes,
        "per_step_comparison": per_step_comparison,
    }


def benchmark_task_sweep(policies: list[Policy]) -> dict[str, object]:
    tasks = ["easy", "medium", "hard"]
    policy_results: dict[str, dict[str, EpisodeSummary]] = {}
    aggregate: dict[str, dict[str, float]] = {}

    for policy in policies:
        task_results: dict[str, EpisodeSummary] = {}
        for task_id in tasks:
            task_results[task_id] = run_episode(policy, task_id)["summary"]
        policy_results[policy.policy_id] = task_results
        aggregate[policy.policy_id] = {
            "mean_total_reward": sum(summary.total_reward for summary in task_results.values()) / len(tasks),
            "completed_deliveries": sum(summary.completed_deliveries for summary in task_results.values()),
            "failed_deliveries": sum(summary.failed_deliveries for summary in task_results.values()),
            "urgent_successes": sum(summary.urgent_successes for summary in task_results.values()),
            "invalid_actions": sum(summary.invalid_action_count for summary in task_results.values()),
            "deadline_misses": sum(summary.deadline_miss_count for summary in task_results.values()),
        }

    ranking = sorted(aggregate, key=lambda policy_id: aggregate[policy_id]["mean_total_reward"], reverse=True)
    return {"tasks": tasks, "policy_results": policy_results, "aggregate": aggregate, "ranking": ranking}
