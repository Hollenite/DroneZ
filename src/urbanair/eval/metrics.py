from __future__ import annotations

from ..models import EpisodeSummary, RewardBreakdown


def build_episode_summary(task_id: str, policy_id: str, seed: int, cumulative_reward: RewardBreakdown, steps_completed: int, completed_deliveries: int, failed_deliveries: int, urgent_successes: int, invalid_action_count: int, deadline_miss_count: int, critical_battery_events: int, action_counts: dict[str, int], triggered_scripted_events: list[str]) -> EpisodeSummary:
    return EpisodeSummary(
        task_id=task_id,
        policy_id=policy_id,
        seed=seed,
        total_reward=cumulative_reward.total,
        steps_completed=steps_completed,
        completed_deliveries=completed_deliveries,
        failed_deliveries=failed_deliveries,
        urgent_successes=urgent_successes,
        invalid_action_count=invalid_action_count,
        deadline_miss_count=deadline_miss_count,
        critical_battery_events=critical_battery_events,
        action_counts=action_counts,
        triggered_scripted_events=triggered_scripted_events,
        reward_breakdown=cumulative_reward,
    )
