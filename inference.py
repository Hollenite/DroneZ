from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urbanair.eval.benchmark import run_episode
from urbanair.models import EpisodeSummary
from urbanair.policies.baseline import HeuristicPolicy, ImprovedPolicy, NaivePolicy, RandomPolicy
from urbanair.policies.base import Policy


def resolve_policy(policy_id: str) -> Policy:
    policies = {
        "random": RandomPolicy,
        "naive": NaivePolicy,
        "heuristic": HeuristicPolicy,
        "improved": ImprovedPolicy,
    }
    try:
        return policies[policy_id]()
    except KeyError as exc:
        raise ValueError(f"Unsupported policy '{policy_id}'.") from exc


def run_inference_episode(task_id: str, policy: Policy, max_steps: int | None = None, max_actions: int | None = None) -> dict[str, object]:
    result = run_episode(policy, task_id, max_steps=max_steps, max_actions=max_actions)
    trace = [
        {
            "step": step["tick"],
            "action_index": step["action_index"],
            "action": step["action"],
            "reward": step["step_reward"],
            "done": step["done"],
            "done_reason": step["done_reason"],
            "terminated_by": step["terminated_by"],
            "invalid_action": step["invalid_action"],
            "pending_orders": step["pending_orders"],
            "triggered_scripted_events": step["triggered_scripted_events"],
        }
        for step in result["trace"]
    ]
    return {"summary": result["summary"], "trace": trace}


def format_step_trace(trace: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    for step in trace:
        lines.append(
            "STEP {step} (action {action_index}) | action={action} | reward={reward:.2f} | pending_orders={pending_orders} | invalid={invalid_action} | done={done}".format(
                step=step["step"],
                action_index=step.get("action_index", step["step"]),
                action=step["action"],
                reward=step["reward"],
                pending_orders=step["pending_orders"],
                invalid_action=step.get("invalid_action", False),
                done=step["done"],
            )
        )
        events = step.get("triggered_scripted_events") or []
        if events:
            lines.append(f"  scripted_events={events}")
    return lines


def format_summary(summary: EpisodeSummary) -> list[str]:
    return [
        "EPISODE SUMMARY",
        f"task_id={summary.task_id}",
        f"policy_id={summary.policy_id}",
        f"seed={summary.seed}",
        f"steps_completed={summary.steps_completed}",
        f"actions_taken={summary.actions_taken}",
        f"total_reward={summary.total_reward:.2f}",
        f"normalized_score={summary.normalized_score:.3f}",
        f"average_step_reward={summary.average_step_reward:.2f}",
        f"done_reason={summary.done_reason}",
        f"terminated_by={summary.terminated_by}",
        f"completed_deliveries={summary.completed_deliveries}",
        f"failed_deliveries={summary.failed_deliveries}",
        f"urgent_successes={summary.urgent_successes}",
        f"invalid_action_count={summary.invalid_action_count}",
        f"deadline_miss_count={summary.deadline_miss_count}",
        f"critical_battery_events={summary.critical_battery_events}",
        f"safety_violations={summary.safety_violations}",
        f"action_counts={json.dumps(summary.action_counts, sort_keys=True)}",
        f"triggered_scripted_events={json.dumps(summary.triggered_scripted_events)}",
        f"reward_breakdown={summary.reward_breakdown.model_dump_json()}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DroneZ local inference episode.")
    parser.add_argument("--task", default="easy", help="Task id to run (easy, medium, hard, demo).")
    parser.add_argument("--policy", default="heuristic", choices=["random", "naive", "heuristic", "improved"], help="Policy to run.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on environment ticks.")
    parser.add_argument("--max-actions", type=int, default=None, help="Optional cap on total policy actions.")
    parser.add_argument("--summary-only", action="store_true", help="Print only the episode summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_inference_episode(task_id=args.task, policy=resolve_policy(args.policy), max_steps=args.max_steps, max_actions=args.max_actions)
    trace = result["trace"]
    summary = result["summary"]

    if not args.summary_only:
        for line in format_step_trace(trace):
            print(line)
    for line in format_summary(summary):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
