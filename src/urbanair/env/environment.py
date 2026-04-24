from __future__ import annotations

from typing import Any

from .action_router import ActionRouter
from .observation_builder import build_observation
from .reward_engine import make_invalid_action_breakdown, merge_breakdowns, serialize_breakdown
from .termination import evaluate_termination
from ..models import RewardBreakdown
from ..sim.engine import SimulatorEngine


class DroneZEnvironment:
    def __init__(
        self,
        engine: SimulatorEngine | None = None,
        default_task_id: str = "easy",
        max_invalid_actions_per_episode: int = 6,
        max_episode_actions: int | None = None,
        action_cap_multiplier: int = 3,
    ) -> None:
        self.engine = engine or SimulatorEngine.from_repo_configs()
        self.default_task_id = default_task_id
        self.router = ActionRouter()
        self.max_invalid_actions_per_episode = max_invalid_actions_per_episode
        self.max_episode_actions = max_episode_actions
        self.action_cap_multiplier = max(1, action_cap_multiplier)
        self.state = None
        self.current_task_id: str | None = None
        self.cumulative_breakdown: RewardBreakdown = RewardBreakdown()
        self.last_info: dict[str, Any] = {}
        self.episode_action_count = 0
        self.invalid_action_count = 0
        self.invalid_action_streak = 0
        self.current_max_episode_actions = 0

    def tasks(self) -> list[str]:
        return sorted(self.engine.task_configs.keys())

    def reset(self, task_id: str | None = None) -> tuple[dict[str, object], dict[str, Any]]:
        selected_task = task_id or self.default_task_id
        self.state = self.engine.reset(selected_task)
        self.current_task_id = selected_task
        self.cumulative_breakdown = self.state.cumulative_reward.model_copy(deep=True)
        self.episode_action_count = 0
        self.invalid_action_count = 0
        self.invalid_action_streak = 0
        self.current_max_episode_actions = self.max_episode_actions or (self.state.task_config.horizon * self.action_cap_multiplier)
        done, done_reason = evaluate_termination(self.state)
        observation = build_observation(self.state, reward_breakdown=self.cumulative_breakdown)
        info = {
            "task_id": selected_task,
            "done": done,
            "done_reason": done_reason,
            "terminated_by": "environment",
            "reward_breakdown": serialize_breakdown(self.cumulative_breakdown),
            "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
            "invalid_action": False,
            "episode_action_count": self.episode_action_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_streak": self.invalid_action_streak,
            "max_episode_actions": self.current_max_episode_actions,
            "triggered_scripted_events": [],
            "recent_events": list(self.state.recent_events),
        }
        self.last_info = info
        return observation, info

    def state_snapshot(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before state().")

        observation = build_observation(self.state, reward_breakdown=self.cumulative_breakdown)
        info = {
            "task_id": self.current_task_id,
            "done": self.last_info.get("done_reason", "ongoing") != "ongoing",
            "done_reason": self.last_info.get("done_reason", "ongoing"),
            "terminated_by": self.last_info.get("terminated_by", "environment"),
            "reward_breakdown": serialize_breakdown(self.cumulative_breakdown),
            "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
            "invalid_action": False,
            "episode_action_count": self.episode_action_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_streak": self.invalid_action_streak,
            "max_episode_actions": self.current_max_episode_actions,
            "triggered_scripted_events": self.last_info.get("triggered_scripted_events", []),
            "recent_events": list(self.state.recent_events),
        }
        return {
            "task_id": self.current_task_id,
            "observation": observation,
            "info": info,
            "episode_action_count": self.episode_action_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_streak": self.invalid_action_streak,
            "max_episode_actions": self.current_max_episode_actions,
            "done_reason": self.last_info.get("done_reason", "ongoing"),
            "terminated_by": self.last_info.get("terminated_by", "environment"),
        }

    def step(self, action: dict[str, Any]) -> tuple[dict[str, object], float, bool, dict[str, Any]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step().")

        self.episode_action_count += 1
        routed = self.router.route(self.state, action)
        action_record = {"action": routed.action_type, "params": routed.normalized_params}
        if not routed.is_valid:
            self.invalid_action_count += 1
            self.invalid_action_streak += 1
            loop_penalty = 0.0
            scaled_penalty = self.engine.reward_weights.negative["invalid_action"] * (1.0 + 0.5 * max(0, self.invalid_action_streak - 1))
            if self.invalid_action_streak >= 2:
                loop_penalty = self.engine.reward_weights.negative.get("loop_or_no_progress", 0.0) * max(1, self.invalid_action_streak - 1)
            breakdown = make_invalid_action_breakdown(scaled_penalty, loop_penalty)
            self.state.recent_events = (
                self.state.recent_events
                + [f"Invalid action rejected: {routed.error_message}"]
            )[-12:]
            self.cumulative_breakdown = merge_breakdowns(self.cumulative_breakdown, breakdown)
            done, done_reason, terminated_by = self._evaluate_episode_status()
            observation = build_observation(self.state, reward_breakdown=breakdown, last_action=action_record)
            info = {
                "task_id": self.current_task_id,
                "invalid_action": True,
                "error_code": routed.error_code,
                "error_message": routed.error_message,
                "reward_breakdown": serialize_breakdown(breakdown),
                "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
                "done_reason": done_reason,
                "terminated_by": terminated_by,
                "episode_action_count": self.episode_action_count,
                "invalid_action_count": self.invalid_action_count,
                "invalid_action_streak": self.invalid_action_streak,
                "max_episode_actions": self.current_max_episode_actions,
                "triggered_scripted_events": [],
                "recent_events": list(self.state.recent_events),
            }
            self.last_info = info
            return observation, breakdown.total, done, info

        result = self.engine.step(self.state, routed.simulator_action)
        self.invalid_action_streak = 0
        self.cumulative_breakdown = self.state.cumulative_reward.model_copy(deep=True)
        done, done_reason, terminated_by = self._evaluate_episode_status(engine_done=result.done)
        observation = build_observation(self.state, reward_breakdown=result.reward_breakdown, last_action=action_record)
        info = {
            "task_id": self.current_task_id,
            "invalid_action": False,
            "error_code": None,
            "error_message": None,
            "reward_breakdown": serialize_breakdown(result.reward_breakdown),
            "cumulative_reward": serialize_breakdown(self.cumulative_breakdown),
            "done_reason": done_reason,
            "terminated_by": terminated_by,
            "resolved_order_ids": result.info.get("resolved_order_ids", []),
            "reward_inputs": result.info.get("reward_inputs", {}),
            "recovery_actions": result.info.get("recovery_actions", {}),
            "pending_recovery_orders": result.info.get("pending_recovery_orders", []),
            "delivery_attempt_required": result.info.get("delivery_attempt_required", []),
            "zone_holds": result.info.get("zone_holds", {}),
            "episode_action_count": self.episode_action_count,
            "invalid_action_count": self.invalid_action_count,
            "invalid_action_streak": self.invalid_action_streak,
            "max_episode_actions": self.current_max_episode_actions,
            "triggered_scripted_events": result.info.get("triggered_scripted_events", []),
            "recent_events": list(self.state.recent_events),
        }
        self.last_info = info
        return observation, result.reward, done, info

    def _evaluate_episode_status(self, engine_done: bool | None = None) -> tuple[bool, str, str]:
        if self.invalid_action_count >= self.max_invalid_actions_per_episode:
            return True, "invalid_action_cap_reached", "environment_guard"
        if self.episode_action_count >= self.current_max_episode_actions:
            return True, "action_cap_reached", "environment_guard"

        done, done_reason = evaluate_termination(self.state, done=engine_done)
        terminated_by = "environment" if done else "ongoing"
        return done, done_reason, terminated_by
