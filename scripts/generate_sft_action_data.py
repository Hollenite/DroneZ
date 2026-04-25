from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from urbanair.env.environment import DroneZEnvironment
from urbanair.policies.baseline import ImprovedPolicy
from urbanair.training.action_format import build_action_prompt, build_candidate_actions, compact_observation_summary


DEFAULT_TASKS = ["easy", "medium", "demo", "hard"]
DEFAULT_OUTPUT = ROOT / "artifacts" / "training" / "sft_action_data.jsonl"


def generate_examples(tasks: list[str], max_actions: int | None = None) -> list[dict[str, Any]]:
    policy = ImprovedPolicy()
    examples: list[dict[str, Any]] = []
    for task_id in tasks:
        env = DroneZEnvironment(default_task_id=task_id, max_episode_actions=max_actions)
        observation, info = env.reset(task_id)
        done = bool(info.get("done", False))
        while not done:
            action = policy.choose_action(observation, info)
            prompt = build_action_prompt(observation, candidate_choice=True)
            candidates = build_candidate_actions(observation)
            next_observation, reward, done, info = env.step(action)
            if not info.get("invalid_action", False):
                examples.append(
                    {
                        "task_id": task_id,
                        "step": observation.get("step"),
                        "observation_summary": compact_observation_summary(observation),
                        "candidate_actions": candidates,
                        "prompt": prompt,
                        "action_json": action,
                        "completion": json.dumps(action, sort_keys=True),
                        "reward_after_action": reward,
                        "done_after_action": done,
                    }
                )
            observation = next_observation
    return examples


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate SFT action-format data from ImprovedPolicy traces.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-actions", type=int, default=None)
    args = parser.parse_args(argv)

    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    examples = generate_examples(tasks or DEFAULT_TASKS, max_actions=args.max_actions)
    destination = Path(args.output)
    write_jsonl(destination, examples)
    print(json.dumps({"output": str(destination), "examples": len(examples), "tasks": tasks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
