from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_grpo import main as shared_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Colab-friendly entrypoint for DroneZ GRPO setup.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "training"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--tasks", default="easy,medium,demo")
    parser.add_argument("--dry-run", action="store_true", help="Prepare the real prompt/action interface without requiring TRL.")
    args = parser.parse_args(argv)

    mode = "dry-run" if args.dry_run else "trl-template"
    return shared_main(
        [
            "--mode",
            mode,
            "--output-dir",
            args.output_dir,
            "--model",
            args.model,
            "--tasks",
            args.tasks,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
