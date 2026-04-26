"""Random valid-action remote agent for the DroneZ runtime."""

from __future__ import annotations

import os
import random
from typing import Any

import requests


BASE_URL = os.environ.get("BASE_URL", "https://krishna2521-dronez-openenv.hf.space").rstrip("/")
TASK_ID = os.environ.get("TASK_ID", "easy")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "20"))
SEED = int(os.environ.get("SEED", "42"))


def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main() -> None:
    random.seed(SEED)
    reset = post("/reset", {"task_id": TASK_ID})
    print(f"Started session {reset['session_id']} on task {TASK_ID}")

    total_reward = 0.0
    for step in range(MAX_STEPS):
        candidates = post("/valid-actions", {})["actions"]
        if not candidates:
            print("No candidate actions available.")
            break
        chosen = random.choice(candidates)["action"]
        result = post("/step", {"action": chosen})
        total_reward += float(result["reward"])
        print(
            {
                "step": step + 1,
                "action": chosen,
                "reward": result["reward"],
                "done": result["done"],
                "done_reason": result["info"].get("done_reason"),
            }
        )
        if result["done"]:
            break

    print({"total_reward": round(total_reward, 2)})


if __name__ == "__main__":
    main()
