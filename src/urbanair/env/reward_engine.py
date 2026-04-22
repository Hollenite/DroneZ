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


def make_invalid_action_breakdown(penalty: float) -> RewardBreakdown:
    return RewardBreakdown.from_components(negative={"invalid_action": penalty})


def serialize_breakdown(breakdown: RewardBreakdown) -> dict[str, object]:
    return {
        "positive": dict(sorted(breakdown.positive.items())),
        "negative": dict(sorted(breakdown.negative.items())),
        "total": breakdown.total,
    }
