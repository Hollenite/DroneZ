from __future__ import annotations

from uuid import uuid4

from ..env.environment import DroneZEnvironment


class EnvironmentRegistry:
    def __init__(self) -> None:
        self._sessions: dict[str, DroneZEnvironment] = {}

    def create_session(self, task_id: str | None = None) -> tuple[str, DroneZEnvironment, dict[str, object], dict[str, object]]:
        environment = DroneZEnvironment()
        session_id = uuid4().hex
        observation, info = environment.reset(task_id)
        self._sessions[session_id] = environment
        return session_id, environment, observation, info

    def get(self, session_id: str) -> DroneZEnvironment | None:
        return self._sessions.get(session_id)

    def reset_session(self, session_id: str, task_id: str | None = None) -> tuple[dict[str, object], dict[str, object]]:
        environment = self.require(session_id)
        return environment.reset(task_id)

    def step_session(self, session_id: str, action: dict[str, object]) -> tuple[dict[str, object], float, bool, dict[str, object]]:
        environment = self.require(session_id)
        return environment.step(action)

    def state_session(self, session_id: str) -> dict[str, object]:
        environment = self.require(session_id)
        return environment.state_snapshot()

    def require(self, session_id: str) -> DroneZEnvironment:
        environment = self.get(session_id)
        if environment is None:
            raise KeyError(session_id)
        return environment


_registry: EnvironmentRegistry | None = None


def get_registry() -> EnvironmentRegistry:
    global _registry
    if _registry is None:
        _registry = EnvironmentRegistry()
    return _registry


def list_tasks() -> list[str]:
    return DroneZEnvironment().tasks()
