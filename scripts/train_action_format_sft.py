from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "artifacts" / "training" / "sft_action_data.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "training" / "action_format_sft"


def dependency_status() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "trl", "peft", "datasets", "unsloth")}


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_template(args: argparse.Namespace) -> Path:
    data_path = Path(args.data)
    rows = load_jsonl(data_path)
    payload = {
        "mode": "action-format-sft-template",
        "training_executed": False,
        "note": (
            "Dry-run/template for teaching DroneZ JSON action format before GRPO. "
            "Run with --real-train on a GPU machine after installing train dependencies."
        ),
        "model": args.model,
        "data": str(data_path),
        "example_count": len(rows),
        "dependency_status": dependency_status(),
        "recommended_next_step": (
            "Run this SFT warm start first, then run train_grpo_local.py --candidate-choice --real-train."
        ),
        "output_dir": args.output_dir,
        "sample": rows[0] if rows else None,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "sft_template.json"
    destination.write_text(json.dumps(payload, indent=2))
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DroneZ action-format SFT warm-start template.")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--real-train", action="store_true")
    args = parser.parse_args(argv)

    if args.real_train:
        status = dependency_status()
        missing = [name for name in ("torch", "transformers", "datasets", "peft") if not status.get(name)]
        if missing:
            print(f"Action-format SFT is not ready: missing {', '.join(missing)}")
            return 2
        print("Action-format SFT dependencies are present, but this lightweight repo path intentionally writes a template.")

    destination = write_template(args)
    print(f"Wrote {destination}")
    print(destination.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
