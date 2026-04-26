"""Minimal remote DroneZ agent loop.

Usage:
    python examples/remote_agent_quickstart.py

Override target:
    BASE_URL=http://127.0.0.1:8000 python examples/remote_agent_quickstart.py
"""

from __future__ import annotations

import os
from typing import Any

import requests


BASE_URL = os.environ.get("BASE_URL", "https://krishna2521-dronez-openenv.hf.space").rstrip("/")
TASK_ID = os.environ.get("TASK_ID", "easy")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def get(path: str) -> dict[str, Any]:
    response = requests.get(f"{BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    print("DroneZ remote runtime:", BASE_URL)
    print("Tasks:", get("/tasks"))

    reset = post("/reset", {"task_id": TASK_ID})
    print("Session:", reset["session_id"])
    print("Initial task:", reset["observation"]["task_id"])

    total_reward = 0.0
    for step in range(MAX_STEPS):
        candidates = post("/valid-actions", {})["actions"]
        if not candidates:
            print("No valid actions returned; stopping.")
            break
        action = candidates[0]["action"]
        result = post("/step", {"action": action})
        total_reward += float(result["reward"])
        print(
            f"step={step + 1:02d}",
            f"action={action['action']}",
            f"reward={result['reward']:.2f}",
            f"done={result['done']}",
            f"reason={result['info'].get('done_reason')}",
        )
        if result["done"]:
            break

    print("Total reward:", round(total_reward, 2))


if __name__ == "__main__":
    main()
