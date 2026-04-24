from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
from typing import Any

from ..env.environment import DroneZEnvironment
from ..models import EpisodeSummary, RewardBreakdown
from ..policies.base import Policy
from .metrics import build_episode_summary

DEFAULT_MAX_INVALID_ACTIONS = 6
DEFAULT_ACTION_CAP_MULTIPLIER = 3


def run_episode(
    policy: Policy,
    task_id: str,
    max_steps: int | None = None,
    max_actions: int | None = None,
    max_invalid_actions: int = DEFAULT_MAX_INVALID_ACTIONS,
    capture_full_trace: bool = False,
) -> dict[str, object]:
    env = DroneZEnvironment(
        default_task_id=task_id,
        max_invalid_actions_per_episode=max_invalid_actions,
        max_episode_actions=max_actions,
        action_cap_multiplier=DEFAULT_ACTION_CAP_MULTIPLIER,
    )
    observation, info = env.reset(task_id)
    initial_observation = observation
    initial_info = info
    trace: list[dict[str, object]] = []
    action_counts: dict[str, int] = defaultdict(int)
    invalid_action_count = 0
    invalid_action_streak = 0
    triggered_scripted_events: list[str] = []
    runner_done_reason = info["done_reason"]
    runner_terminated_by = info.get("terminated_by", "environment")
    max_steps = max_steps or env.state.task_config.horizon
    max_actions = max_actions or env.current_max_episode_actions

    done = info["done"]
    while not done:
        if env.episode_action_count >= max_actions:
            runner_done_reason = "runner_action_cap_reached"
            runner_terminated_by = "runner_guard"
            break
        if observation["step"] >= max_steps:
            runner_done_reason = "runner_step_cap_reached"
            runner_terminated_by = "runner_guard"
            break

        action = policy.choose_action(observation, info)
        action_counts[action["action"]] += 1
        observation, reward, done, info = env.step(action)
        if info["invalid_action"]:
            invalid_action_count += 1
            invalid_action_streak += 1
        else:
            invalid_action_streak = 0
        if invalid_action_streak >= max_invalid_actions:
            runner_done_reason = "runner_invalid_action_cap_reached"
            runner_terminated_by = "runner_guard"
            done = True

        triggered_scripted_events.extend(info.get("triggered_scripted_events", []))
        trace_item: dict[str, object] = {
            "action_index": env.episode_action_count,
            "tick": observation["step"],
            "action": action,
            "step_reward": reward,
            "cumulative_reward": info["cumulative_reward"]["total"],
            "pending_orders": sum(1 for order in observation["orders"] if order["status"] not in {"delivered", "canceled"}),
            "done": done,
            "done_reason": info.get("done_reason", runner_done_reason),
            "terminated_by": info.get("terminated_by", runner_terminated_by),
            "invalid_action": info["invalid_action"],
            "invalid_action_count": info.get("invalid_action_count", invalid_action_count),
            "triggered_scripted_events": list(info.get("triggered_scripted_events", [])),
            "reward_breakdown": info["reward_breakdown"],
            "summary": observation.get("summary", ""),
        }
        if capture_full_trace:
            trace_item["observation"] = observation
            trace_item["info"] = info
        trace.append(trace_item)

        runner_done_reason = info.get("done_reason", runner_done_reason)
        runner_terminated_by = info.get("terminated_by", runner_terminated_by)

    orders = observation["orders"]
    reward_breakdown = RewardBreakdown.model_validate(info["cumulative_reward"])
    summary = build_episode_summary(
        task_id=task_id,
        policy_id=policy.policy_id,
        seed=env.state.task_config.seed,
        cumulative_reward=reward_breakdown,
        steps_completed=observation["step"],
        actions_taken=env.episode_action_count,
        total_orders=len(orders),
        total_urgent_orders=sum(1 for order in orders if order["priority"] in {"urgent", "medical"}),
        completed_deliveries=sum(1 for order in orders if order["status"] == "delivered"),
        failed_deliveries=sum(1 for order in orders if order["status"] == "failed"),
        urgent_successes=sum(1 for order in orders if order["status"] == "delivered" and order["priority"] in {"urgent", "medical"}),
        invalid_action_count=invalid_action_count,
        deadline_miss_count=_metric_from_breakdown(reward_breakdown, "missed_deadline", env.engine.reward_weights.negative),
        critical_battery_events=_metric_from_breakdown(reward_breakdown, "battery_critical", env.engine.reward_weights.negative),
        safety_violations=_metric_from_breakdown(reward_breakdown, "unsafe_zone_entry", env.engine.reward_weights.negative),
        done_reason=runner_done_reason,
        terminated_by=runner_terminated_by,
        action_counts=dict(action_counts),
        triggered_scripted_events=triggered_scripted_events,
    )
    return {
        "summary": summary,
        "trace": trace,
        "initial_observation": initial_observation,
        "initial_info": initial_info,
    }


def compare_demo_policies(
    baseline_policy: Policy,
    candidate_policy: Policy,
    *,
    max_steps: int = 18,
    max_actions: int = 48,
) -> dict[str, object]:
    baseline_result = run_episode(baseline_policy, "demo", max_steps=max_steps, max_actions=max_actions)
    candidate_result = run_episode(candidate_policy, "demo", max_steps=max_steps, max_actions=max_actions)
    baseline_summary: EpisodeSummary = baseline_result["summary"]
    candidate_summary: EpisodeSummary = candidate_result["summary"]

    per_step_comparison = []
    for step_index, (baseline_step, candidate_step) in enumerate(
        zip_longest(baseline_result["trace"], candidate_result["trace"], fillvalue={}),
        start=1,
    ):
        per_step_comparison.append(
            {
                "index": step_index,
                "baseline_tick": baseline_step.get("tick"),
                "heuristic_tick": candidate_step.get("tick"),
                "triggered_scripted_events": candidate_step.get("triggered_scripted_events", []),
                "baseline_action": baseline_step.get("action"),
                "candidate_action": candidate_step.get("action"),
                "baseline_step_reward": baseline_step.get("step_reward"),
                "candidate_step_reward": candidate_step.get("step_reward"),
                "baseline_cumulative_reward": baseline_step.get("cumulative_reward"),
                "candidate_cumulative_reward": candidate_step.get("cumulative_reward"),
                "baseline_pending_orders": baseline_step.get("pending_orders"),
                "candidate_pending_orders": candidate_step.get("pending_orders"),
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
        "normalized_score_delta": candidate_summary.normalized_score - baseline_summary.normalized_score,
        "delivery_delta": candidate_summary.completed_deliveries - baseline_summary.completed_deliveries,
        "urgent_success_delta": candidate_summary.urgent_successes - baseline_summary.urgent_successes,
        "per_step_comparison": per_step_comparison,
    }


def benchmark_task_sweep(
    policies: list[Policy],
    tasks: list[str] | None = None,
    *,
    max_actions: int | None = None,
) -> dict[str, object]:
    tasks = tasks or ["easy", "medium", "hard", "demo"]
    policy_results: dict[str, dict[str, EpisodeSummary]] = {}
    aggregate: dict[str, dict[str, float]] = {}

    for policy in policies:
        task_results: dict[str, EpisodeSummary] = {}
        for task_id in tasks:
            task_results[task_id] = run_episode(policy, task_id, max_actions=max_actions)["summary"]
        policy_results[policy.policy_id] = task_results
        aggregate[policy.policy_id] = {
            "mean_total_reward": sum(summary.total_reward for summary in task_results.values()) / len(tasks),
            "mean_normalized_score": sum(summary.normalized_score for summary in task_results.values()) / len(tasks),
            "completed_deliveries": sum(summary.completed_deliveries for summary in task_results.values()),
            "failed_deliveries": sum(summary.failed_deliveries for summary in task_results.values()),
            "urgent_successes": sum(summary.urgent_successes for summary in task_results.values()),
            "invalid_actions": sum(summary.invalid_action_count for summary in task_results.values()),
            "deadline_misses": sum(summary.deadline_miss_count for summary in task_results.values()),
            "safety_violations": sum(summary.safety_violations for summary in task_results.values()),
            "actions_taken": sum(summary.actions_taken for summary in task_results.values()),
        }

    ranking = sorted(
        aggregate,
        key=lambda policy_id: (
            aggregate[policy_id]["mean_normalized_score"],
            aggregate[policy_id]["mean_total_reward"],
        ),
        reverse=True,
    )
    return {"tasks": tasks, "policy_results": policy_results, "aggregate": aggregate, "ranking": ranking}


def _metric_from_breakdown(
    breakdown: RewardBreakdown,
    key: str,
    weights: dict[str, float],
) -> int:
    weight = abs(weights.get(key, 0.0))
    if not weight:
        return 0
    value = abs(breakdown.negative.get(key, 0.0))
    return int(round(value / weight))
