from __future__ import annotations

from ..models import EpisodeSummary, RewardBreakdown


def clamp_normalized_score(value: float) -> float:
    return min(0.99, max(0.01, value))


def compute_normalized_score(
    *,
    total_reward: float,
    total_orders: int,
    total_urgent_orders: int,
    completed_deliveries: int,
    urgent_successes: int,
    deadline_miss_count: int,
    critical_battery_events: int,
    invalid_action_count: int,
    safety_violations: int,
    actions_taken: int,
) -> float:
    total_orders = max(1, total_orders)
    total_urgent_orders = max(1, total_urgent_orders)
    actions_taken = max(1, actions_taken)

    delivery_rate = completed_deliveries / total_orders
    urgent_rate = urgent_successes / total_urgent_orders
    deadline_rate = deadline_miss_count / total_orders
    invalid_rate = invalid_action_count / actions_taken
    battery_rate = critical_battery_events / actions_taken
    safety_rate = safety_violations / actions_taken
    reward_term = 0.5 + (total_reward / max(40.0, float(total_orders) * 18.0))

    raw_score = (
        reward_term * 0.42
        + delivery_rate * 0.18
        + urgent_rate * 0.16
        - deadline_rate * 0.10
        - invalid_rate * 0.10
        - battery_rate * 0.07
        - safety_rate * 0.17
    )
    return clamp_normalized_score(raw_score)


def build_episode_summary(
    task_id: str,
    policy_id: str,
    seed: int,
    cumulative_reward: RewardBreakdown,
    steps_completed: int,
    actions_taken: int,
    total_orders: int,
    total_urgent_orders: int,
    completed_deliveries: int,
    failed_deliveries: int,
    urgent_successes: int,
    invalid_action_count: int,
    deadline_miss_count: int,
    critical_battery_events: int,
    safety_violations: int,
    done_reason: str,
    terminated_by: str,
    action_counts: dict[str, int],
    triggered_scripted_events: list[str],
) -> EpisodeSummary:
    normalized_score = compute_normalized_score(
        total_reward=cumulative_reward.total,
        total_orders=total_orders,
        total_urgent_orders=total_urgent_orders,
        completed_deliveries=completed_deliveries,
        urgent_successes=urgent_successes,
        deadline_miss_count=deadline_miss_count,
        critical_battery_events=critical_battery_events,
        invalid_action_count=invalid_action_count,
        safety_violations=safety_violations,
        actions_taken=actions_taken,
    )
    return EpisodeSummary(
        task_id=task_id,
        policy_id=policy_id,
        seed=seed,
        total_reward=cumulative_reward.total,
        normalized_score=normalized_score,
        steps_completed=steps_completed,
        actions_taken=actions_taken,
        average_step_reward=cumulative_reward.total / max(1, actions_taken),
        total_orders=total_orders,
        total_urgent_orders=total_urgent_orders,
        completed_deliveries=completed_deliveries,
        failed_deliveries=failed_deliveries,
        urgent_successes=urgent_successes,
        invalid_action_count=invalid_action_count,
        deadline_miss_count=deadline_miss_count,
        critical_battery_events=critical_battery_events,
        safety_violations=safety_violations,
        done_reason=done_reason,
        terminated_by=terminated_by,
        action_counts=action_counts,
        triggered_scripted_events=triggered_scripted_events,
        reward_breakdown=cumulative_reward,
    )
