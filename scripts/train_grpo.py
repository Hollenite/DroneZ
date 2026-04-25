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
from urbanair.eval.benchmark import benchmark_task_sweep, run_episode
from urbanair.policies.baseline import HeuristicPolicy, ImprovedPolicy, RandomPolicy
from urbanair.training.action_format import build_action_prompt, parse_llm_action

RESULTS_DIR = ROOT / "artifacts" / "results"
TRAINING_DIR = ROOT / "artifacts" / "training"
PLOTS_DIR = ROOT / "artifacts" / "plots"


def build_prompt(observation: dict[str, Any]) -> str:
    return build_action_prompt(observation, candidate_choice=False)


def parse_action_text(text: str) -> dict[str, Any]:
    result = parse_llm_action(text)
    if not result.valid_action_shape:
        raise ValueError(result.error_code or "invalid_action")
    return result.action


def dependency_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for module_name in ("datasets", "transformers", "trl", "accelerate"):
        try:
            __import__(module_name)
            status[module_name] = True
        except Exception:
            status[module_name] = False
    return status


def reference_policy_evaluation() -> dict[str, object]:
    comparison = benchmark_task_sweep([RandomPolicy(), HeuristicPolicy(), ImprovedPolicy()])
    return {
        "status": "reference_only",
        "note": (
            "These are deterministic scripted-policy reference runs against the real DroneZ "
            "environment. They are useful for smoke testing and colab setup, but they are "
            "not a trained-model evaluation."
        ),
        "ranking": comparison["ranking"],
        "aggregate": comparison["aggregate"],
        "demo_improved": comparison["policy_results"]["improved"]["demo"].model_dump(),
        "demo_heuristic": comparison["policy_results"]["heuristic"]["demo"].model_dump(),
    }


def write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_smoke_curriculum() -> dict[str, object]:
    curriculum = [
        ("easy", RandomPolicy()),
        ("medium", HeuristicPolicy()),
        ("demo", ImprovedPolicy()),
    ]
    phases = []
    for task_id, policy in curriculum:
        result = run_episode(policy, task_id)
        phases.append(
            {
                "task_id": task_id,
                "policy_id": policy.policy_id,
                "summary": result["summary"].model_dump(),
            }
        )
    payload = {
        "mode": "smoke",
        "note": "This is a local environment-connected smoke harness, not a finished GRPO training run.",
        "phases": phases,
    }
    destination = RESULTS_DIR / "training_smoke_metrics.json"
    destination.write_text(json.dumps(payload, indent=2))
    return {"payload": payload, "destination": destination}


def run_dry_run(output_dir: Path, model: str, tasks: list[str]) -> dict[str, object]:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    env = DroneZEnvironment(default_task_id="easy")
    observation, info = env.reset("easy")
    prompt = build_prompt(observation)
    sample_action = RandomPolicy().choose_action(observation, info)
    reference_eval = reference_policy_evaluation()

    training_metrics = {
        "mode": "dry-run",
        "training_executed": False,
        "note": (
            "Dry-run mode prepares the real DroneZ prompt/action interface and environment "
            "evaluation metadata without claiming a GRPO run happened."
        ),
        "recommended_models": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "google/gemma-2-2b-it",
        ],
        "selected_model": model,
        "curriculum": tasks,
        "dependency_status": dependency_status(),
        "prompt_example": prompt,
        "sample_action_json": sample_action,
        "action_parser_example": parse_action_text(json.dumps(sample_action)),
        "output_dir": str(output_dir),
    }
    eval_before = {
        "status": "reference_only",
        "note": "Reference scripted baselines before any model training.",
        "payload": reference_eval,
    }
    eval_after = {
        "status": "not_run",
        "note": "No GRPO training was executed in dry-run mode, so there is no after-training evaluation yet.",
    }

    destinations = [
        write_json(TRAINING_DIR / "training_metrics.json", training_metrics),
        write_json(TRAINING_DIR / "eval_before.json", eval_before),
        write_json(TRAINING_DIR / "eval_after.json", eval_after),
    ]
    return {
        "payload": {
            "training_metrics": training_metrics,
            "eval_before": eval_before,
            "eval_after": eval_after,
        },
        "destination": destinations[-1],
        "destinations": [str(path) for path in destinations],
    }


def run_trl_template(output_dir: Path, model: str, tasks: list[str]) -> dict[str, object]:
    try:
        from datasets import Dataset  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from trl import GRPOConfig  # noqa: F401
    except Exception as exc:  # pragma: no cover - dependency-gated path
        raise RuntimeError(
            "TRL dependencies are not installed. Use `pip install -e .[train]` or run this script in Colab."
        ) from exc

    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:
        gpu_available = False

    env = DroneZEnvironment(default_task_id="easy")
    observation, _ = env.reset("easy")
    template = {
        "mode": "trl_template",
        "note": (
            "Dependency check passed. This is a Colab-ready template plan for a real GRPO run "
            "against the DroneZ environment. It does not fabricate training outputs."
        ),
        "selected_model": model,
        "selected_curriculum": tasks,
        "gpu_available": gpu_available,
        "hf_token_expected": "Set HF_TOKEN in the Colab or shell environment. Do not paste it into the notebook.",
        "recommended_curriculum": ["easy", "medium", "demo", "hard"],
        "recommended_models": [
            "Qwen/Qwen2.5-0.5B-Instruct",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "google/gemma-2-2b-it",
        ],
        "prompt_example": build_prompt(observation),
        "sample_action_json": {"action": "assign_delivery", "params": {"drone_id": "FA-1", "order_id": "O1"}},
        "environment_reward_source": "Use the real reward returned by DroneZEnvironment.step(...) rollouts.",
        "output_dir": str(output_dir),
    }
    destination = output_dir / "grpo_template.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(template, indent=2))
    return {"payload": template, "destination": destination}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DroneZ smoke harness and GRPO template.")
    parser.add_argument("--mode", choices=["smoke", "dry-run", "trl-template"], default="smoke")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "training"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tasks", default="easy,medium,demo,hard")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]

    if args.mode == "smoke":
        result = run_smoke_curriculum()
    elif args.mode == "dry-run":
        result = run_dry_run(output_dir, args.model, tasks)
    else:
        try:
            result = run_trl_template(output_dir, args.model, tasks)
        except RuntimeError as exc:
            print(f"GRPO setup is not ready: {exc}", file=sys.stderr)
            return 2

    print(f"Wrote {result['destination']}")
    print(json.dumps(result["payload"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
