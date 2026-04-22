from __future__ import annotations


def render_demo_comparison_markdown(comparison: dict[str, object]) -> str:
    lines = [
        f"# Demo comparison: {comparison['baseline_policy_id']} vs {comparison['candidate_policy_id']}",
        f"Seed: {comparison['seed']}",
        f"Reward delta: {comparison['reward_delta']:.2f}",
        f"Delivery delta: {comparison['delivery_delta']}",
        f"Urgent success delta: {comparison['urgent_success_delta']}",
        "",
        "## Step comparison",
    ]
    for step in comparison["per_step_comparison"]:
        lines.append(
            f"- tick {step['tick']}: baseline={step['baseline_action']} heuristic={step['heuristic_action']} reward_delta={step['heuristic_step_reward'] - step['baseline_step_reward']:.2f} events={step['triggered_scripted_events']}"
        )
    return "\n".join(lines)


def render_task_sweep_markdown(results: dict[str, object]) -> str:
    lines = ["# Task sweep", ""]
    for policy_id in results["ranking"]:
        aggregate = results["aggregate"][policy_id]
        lines.append(
            f"- {policy_id}: mean_reward={aggregate['mean_total_reward']:.2f}, deliveries={aggregate['completed_deliveries']}, urgent_successes={aggregate['urgent_successes']}, invalid_actions={aggregate['invalid_actions']}"
        )
    return "\n".join(lines)
