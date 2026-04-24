from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "results"
PLOTS_DIR = ROOT / "artifacts" / "plots"


def load_results() -> dict[str, object]:
    path = RESULTS_DIR / "policy_comparison.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/evaluate_policies.py first.")
    return json.loads(path.read_text())


def plot_metric(payload: dict[str, object], metric: str, title: str, destination: Path) -> None:
    ranking = payload["ranking"]
    aggregate = payload["aggregate"]
    values = [aggregate[policy_id][metric] for policy_id in ranking]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(ranking, values, color=["#304C89", "#4AA96C", "#F4B400", "#D1495B"][: len(ranking)])
    plt.title(title)
    plt.ylabel(metric.replace("_", " ").title())
    plt.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}" if isinstance(value, float) else str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def main() -> int:
    payload = load_results()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric(payload, "mean_total_reward", "DroneZ Mean Reward by Policy", PLOTS_DIR / "reward_comparison.png")
    plot_metric(payload, "completed_deliveries", "DroneZ Completed Deliveries by Policy", PLOTS_DIR / "delivery_success_comparison.png")
    plot_metric(payload, "invalid_actions", "DroneZ Invalid Actions by Policy", PLOTS_DIR / "invalid_actions_comparison.png")

    print(f"Wrote plots to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
