from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urbanair.eval.benchmark import benchmark_task_sweep
from urbanair.policies.baseline import HeuristicPolicy, ImprovedPolicy, NaivePolicy, RandomPolicy

RESULTS_DIR = ROOT / "artifacts" / "results"


def build_payload() -> dict[str, object]:
    results = benchmark_task_sweep([RandomPolicy(), NaivePolicy(), HeuristicPolicy(), ImprovedPolicy()])
    return {
        "tasks": results["tasks"],
        "ranking": results["ranking"],
        "aggregate": results["aggregate"],
        "policy_results": {
            policy_id: {task_id: summary.model_dump() for task_id, summary in summaries.items()}
            for policy_id, summaries in results["policy_results"].items()
        },
    }


def write_csv(payload: dict[str, object], destination: Path) -> None:
    fieldnames = [
        "policy_id",
        "task_id",
        "total_reward",
        "normalized_score",
        "completed_deliveries",
        "urgent_successes",
        "failed_deliveries",
        "deadline_miss_count",
        "critical_battery_events",
        "safety_violations",
        "invalid_action_count",
        "average_step_reward",
        "steps_completed",
        "actions_taken",
        "done_reason",
        "terminated_by",
    ]
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy_id, summaries in payload["policy_results"].items():
            for task_id, summary in summaries.items():
                writer.writerow(
                    {
                        "policy_id": policy_id,
                        "task_id": task_id,
                        "total_reward": summary["total_reward"],
                        "normalized_score": summary["normalized_score"],
                        "completed_deliveries": summary["completed_deliveries"],
                        "urgent_successes": summary["urgent_successes"],
                        "failed_deliveries": summary["failed_deliveries"],
                        "deadline_miss_count": summary["deadline_miss_count"],
                        "critical_battery_events": summary["critical_battery_events"],
                        "safety_violations": summary["safety_violations"],
                        "invalid_action_count": summary["invalid_action_count"],
                        "average_step_reward": summary["average_step_reward"],
                        "steps_completed": summary["steps_completed"],
                        "actions_taken": summary["actions_taken"],
                        "done_reason": summary["done_reason"],
                        "terminated_by": summary["terminated_by"],
                    }
                )


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_path = RESULTS_DIR / "policy_comparison.json"
    csv_path = RESULTS_DIR / "policy_comparison.csv"
    json_path.write_text(json.dumps(payload, indent=2))
    write_csv(payload, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Ranking: {payload['ranking']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
