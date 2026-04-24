from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:."
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def test_evaluation_and_trace_scripts_generate_outputs() -> None:
    _run([sys.executable, "scripts/evaluate_policies.py"])
    _run([sys.executable, "scripts/generate_demo_trace.py", "--task", "demo", "--policy", "improved"])

    assert (ROOT / "artifacts" / "results" / "policy_comparison.json").exists()
    assert (ROOT / "artifacts" / "results" / "policy_comparison.csv").exists()
    trace_path = ROOT / "artifacts" / "traces" / "demo_improved_trace.json"
    assert trace_path.exists()

    payload = json.loads(trace_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["policy_id"] == "improved"
    assert "frames" in payload and payload["frames"]
    assert "initial_observation" in payload


def test_training_smoke_and_dry_run_generate_outputs() -> None:
    _run([sys.executable, "scripts/train_grpo.py", "--mode", "smoke"])
    _run([sys.executable, "scripts/train_grpo.py", "--mode", "dry-run"])

    assert (ROOT / "artifacts" / "results" / "training_smoke_metrics.json").exists()
    assert (ROOT / "artifacts" / "training" / "training_metrics.json").exists()
    assert (ROOT / "artifacts" / "training" / "eval_before.json").exists()
    assert (ROOT / "artifacts" / "training" / "eval_after.json").exists()

    payload = json.loads((ROOT / "artifacts" / "training" / "training_metrics.json").read_text())
    assert payload["mode"] == "dry-run"
    assert payload["training_executed"] is False
