from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class DroneZClient:
    base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 30.0
    session_id: str | None = None

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def tasks(self) -> dict[str, Any]:
        return self._request("GET", "/tasks")

    def create_session(self, task_id: str | None = None) -> dict[str, Any]:
        payload = {"task_id": task_id}
        response = self._request("POST", "/sessions", json=payload)
        self.session_id = response["session_id"]
        return response

    def reset(self, task_id: str | None = None) -> dict[str, Any]:
        session_id = self._require_session()
        return self._request("POST", f"/sessions/{session_id}/reset", json={"task_id": task_id})

    def step(self, action: dict[str, Any]) -> dict[str, Any]:
        session_id = self._require_session()
        return self._request("POST", f"/sessions/{session_id}/step", json={"action": action})

    def state(self) -> dict[str, Any]:
        session_id = self._require_session()
        return self._request("GET", f"/sessions/{session_id}/state")

    def _require_session(self) -> str:
        if self.session_id is None:
            created = self.create_session()
            return created["session_id"]
        return self.session_id

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(method, f"{self.base_url.rstrip('/')}{path}", timeout=self.timeout_seconds, **kwargs)
        response.raise_for_status()
        return response.json()
