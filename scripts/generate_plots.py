from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "artifacts" / "results"
TRAINING_DIR = ROOT / "artifacts" / "training"
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


def plot_training_series(values: list[float], title: str, ylabel: str, destination: Path, *, empty_note: str) -> None:
    plt.figure(figsize=(8, 4.5))
    if values:
        plt.plot(range(1, len(values) + 1), values, marker="o", color="#245C8D", linewidth=2.2)
        if len(set(values)) <= 1:
            plt.text(0.5, 0.85, "flat curve: no measured improvement", transform=plt.gca().transAxes, color="#C84C3F", ha="center")
    else:
        plt.text(0.5, 0.5, empty_note, transform=plt.gca().transAxes, ha="center", va="center", color="#647466")
    plt.title(title)
    plt.xlabel("Episode / Check")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def plot_eval_before_after(training_metrics: dict[str, object], destination: Path) -> None:
    before = training_metrics.get("pre_training_mean_reward")
    after = training_metrics.get("post_training_mean_reward")
    labels = ["before", "after"]
    values = [before, after] if before is not None and after is not None else []

    plt.figure(figsize=(7, 4.5))
    if values:
        bars = plt.bar(labels, values, color=["#647466", "#1F7A5D"])
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom")
        if float(after) <= float(before):
            plt.text(0.5, 0.86, "no trained-model improvement yet", transform=plt.gca().transAxes, color="#C84C3F", ha="center")
    else:
        plt.text(0.5, 0.5, "No real before/after training evaluation available", transform=plt.gca().transAxes, ha="center", va="center", color="#647466")
    plt.title("DroneZ Training Evaluation Before / After")
    plt.ylabel("Mean Reward")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(destination, dpi=180)
    plt.close()


def plot_training_artifacts() -> None:
    metrics_path = TRAINING_DIR / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    reward_history = [float(item) for item in metrics.get("reward_history", [])]
    loss_history = [float(item) for item in metrics.get("loss_history", [])]

    format_path = TRAINING_DIR / "format_check" / "format_check.json"
    if not format_path.exists():
        format_path = TRAINING_DIR / "format_check.json"
    format_payload = json.loads(format_path.read_text()) if format_path.exists() else {}
    valid_action_values = []
    if "valid_action_rate" in metrics:
        valid_action_values.append(float(metrics["valid_action_rate"]))
    if "valid_action_rate" in format_payload:
        valid_action_values.append(float(format_payload["valid_action_rate"]))

    plot_training_series(
        reward_history,
        "DroneZ Real Training Reward Curve",
        "Episode Reward",
        PLOTS_DIR / "training_reward_curve.png",
        empty_note="No real training reward history available yet",
    )
    plot_training_series(
        loss_history,
        "DroneZ Training Loss Curve",
        "Loss",
        PLOTS_DIR / "training_loss_curve.png",
        empty_note="No real training loss history available yet",
    )
    plot_training_series(
        valid_action_values,
        "DroneZ Valid Action Rate",
        "Valid Action Rate",
        PLOTS_DIR / "valid_action_rate.png",
        empty_note="Run --format-check to generate valid action rate",
    )
    plot_eval_before_after(metrics, PLOTS_DIR / "eval_before_after.png")


def main() -> int:
    payload = load_results()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_metric(payload, "mean_total_reward", "DroneZ Mean Reward by Policy", PLOTS_DIR / "reward_comparison.png")
    plot_metric(payload, "completed_deliveries", "DroneZ Completed Deliveries by Policy", PLOTS_DIR / "delivery_success_comparison.png")
    plot_metric(payload, "invalid_actions", "DroneZ Invalid Actions by Policy", PLOTS_DIR / "invalid_actions_comparison.png")
    plot_training_artifacts()

    print(f"Wrote plots to {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
