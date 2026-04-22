from __future__ import annotations

from typing import Any

from .action_router import ActionRouter
from .observation_builder import build_observation
from .reward_engine import make_invalid_action_breakdown, merge_breakdowns, serialize_breakdown
from .termination import evaluate_termination
from ..models import RewardBreakdown
from ..sim.engine import SimulatorEngine


class DroneZEnvironment:
    def __init__(self, engine: SimulatorEngine | None = None, default_task_id: str = "easy") -> None:
        self.engine = engine or SimulatorEngine.from_repo_configs()
        self.default_task_id = default_task_id
        self.router = ActionRouter()
        self.state = None
        self.current_task_id: str | None = None
        self.cumulative_breakdown: RewardBreakdown = RewardBreakdown()
        self.last_info: dict[str, Any] = {}

    def tasks(self) -> list[str]:
        return sorted(self.engine.task_configs.keys())

    def reset(self, task_id: str | None = None) -> tuple[dict[str, object], dict[str, Any]]:
        selected_task = task_id or self.default_task_id
        self.state = self.engine.reset(selected_task)
        self.current_task_id = selected_task
        self.cumulative_breakdown = self.state.cumulative_reward.model_copy(deep=True)
        done, done_reason = evaluate_termination(self.state)
        observation = build_observation(self.state, reward_breakdown=self.cumulative_breakdown)
        info = {
            "task_id": selected_task,
            "done": done,
            "done_reason": done_reason,
            "reward_breakdown": serialize_breakdown(self.cumulative_breakdown),
            "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
            "invalid_action": False,
            "recent_events": list(self.state.recent_events),
        }
        self.last_info = info
        return observation, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, object], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step().")

        routed = self.router.route(self.state, action)
        action_record = {"action": routed.action_type, "params": routed.normalized_params}
        if not routed.is_valid:
            breakdown = make_invalid_action_breakdown(self.engine.reward_weights.negative["invalid_action"])
            self.state.recent_events = (
                self.state.recent_events
                + [f"Invalid action rejected: {routed.error_message}"]
            )[-12:]
            self.cumulative_breakdown = merge_breakdowns(self.cumulative_breakdown, breakdown)
            done, done_reason = evaluate_termination(self.state)
            observation = build_observation(self.state, reward_breakdown=breakdown, last_action=action_record)
            info = {
                "task_id": self.current_task_id,
                "invalid_action": True,
                "error_code": routed.error_code,
                "error_message": routed.error_message,
                "reward_breakdown": serialize_breakdown(breakdown),
                "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
                "done_reason": done_reason,
                "recent_events": list(self.state.recent_events),
            }
            self.last_info = info
            return observation, breakdown.total, done, info

        result = self.engine.step(self.state, routed.simulator_action)
        self.cumulative_breakdown = self.state.cumulative_reward.model_copy(deep=True)
        done, done_reason = evaluate_termination(self.state, done=result.done)
        observation = build_observation(self.state, reward_breakdown=result.reward_breakdown, last_action=action_record)
        info = {
            "task_id": self.current_task_id,
            "invalid_action": False,
            "error_code": None,
            "error_message": None,
            "reward_breakdown": serialize_breakdown(result.reward_breakdown),
            "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
            "done_reason": done_reason,
            "resolved_order_ids": result.info.get("resolved_order_ids", []),
            "reward_inputs": result.info.get("reward_inputs", {}),
            "recovery_actions": result.info.get("recovery_actions", {}),
            "recent_events": list(self.state.recent_events),
        }
        self.last_info = info
        return observation, result.reward, done, info
