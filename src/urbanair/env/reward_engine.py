from __future__ import annotations

from ..models import RewardBreakdown


def merge_breakdowns(current: RewardBreakdown, delta: RewardBreakdown) -> RewardBreakdown:
    positive = {
        key: current.positive.get(key, 0.0) + delta.positive.get(key, 0.0)
        for key in set(current.positive) | set(delta.positive)
    }
    negative = {
        key: current.negative.get(key, 0.0) + delta.negative.get(key, 0.0)
        for key in set(current.negative) | set(delta.negative)
    }
    return RewardBreakdown.from_components(positive=positive, negative=negative)


def make_invalid_action_breakdown(penalty: float, loop_penalty: float = 0.0) -> RewardBreakdown:
    negative = {"invalid_action": penalty}
    if loop_penalty:
        negative["loop_or_no_progress"] = loop_penalty
    return RewardBreakdown.from_components(negative=negative)


def serialize_breakdown(breakdown: RewardBreakdown) -> dict[str, object]:
    return {
        "positive": dict(sorted(breakdown.positive.items())),
        "negative": dict(sorted(breakdown.negative.items())),
        "total": breakdown.total,
    }
