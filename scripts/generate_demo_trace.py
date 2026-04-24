from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urbanair.eval.benchmark import run_episode
from urbanair.policies.baseline import HeuristicPolicy, ImprovedPolicy, NaivePolicy, RandomPolicy

TRACES_DIR = ROOT / "artifacts" / "traces"


def resolve_policy(policy_id: str):
    policies = {
        "random": RandomPolicy,
        "naive": NaivePolicy,
        "heuristic": HeuristicPolicy,
        "improved": ImprovedPolicy,
    }
    return policies[policy_id]()


def generate_trace(task_id: str, policy_id: str) -> Path:
    result = run_episode(resolve_policy(policy_id), task_id, capture_full_trace=True)
    payload = {
        "schema_version": "1.0",
        "task_id": task_id,
        "policy_id": policy_id,
        "summary": result["summary"].model_dump(),
        "initial_observation": result["initial_observation"],
        "initial_info": result["initial_info"],
        "frames": result["trace"],
    }
    destination = TRACES_DIR / f"{task_id}_{policy_id}_trace.json"
    destination.write_text(json.dumps(payload, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic DroneZ replay traces.")
    parser.add_argument("--task", default="demo", help="Task to serialize.")
    parser.add_argument("--policy", default="all", choices=["all", "random", "naive", "heuristic", "improved"], help="Policy to serialize.")
    args = parser.parse_args(argv)

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    policies = ["random", "naive", "heuristic", "improved"] if args.policy == "all" else [args.policy]
    for policy_id in policies:
        destination = generate_trace(args.task, policy_id)
        print(f"Wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
