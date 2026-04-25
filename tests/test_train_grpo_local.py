from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}:."
    return subprocess.run(command, cwd=ROOT, env=env, check=False, capture_output=True, text=True)


def test_sanity_check_writes_honest_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "local_sanity"
    result = _run([sys.executable, "scripts/train_grpo_local.py", "--sanity-check", "--output-dir", str(output_dir)])

    assert result.returncode == 0
    payload = json.loads((output_dir / "sanity_check.json").read_text())
    assert payload["mode"] == "local-grpo-sanity"
    assert payload["training_executed"] is False
    assert payload["selected_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert payload["curriculum"] == ["easy", "medium", "demo"]
    assert "gpu_available" in payload["device"]


def test_training_fails_honestly_without_gpu(tmp_path: Path) -> None:
    output_dir = tmp_path / "local_train"
    result = _run([sys.executable, "scripts/train_grpo_local.py", "--output-dir", str(output_dir), "--episodes", "1"])

    assert result.returncode == 2
    assert "Local GRPO training is not ready:" in result.stderr
    assert not (output_dir / "training_metrics.json").exists()
