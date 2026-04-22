from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .env_factory import get_registry, list_tasks


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="DroneZ Runtime", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, object]:
    available_tasks = list_tasks()
    default_task_id = "easy" if "easy" in available_tasks else (available_tasks[0] if available_tasks else None)
    return {"tasks": available_tasks, "default_task_id": default_task_id}


@app.post("/sessions")
def create_session(payload: ResetRequest) -> dict[str, object]:
    task_id = payload.task_id or None
    try:
        session_id, _, observation, info = get_registry().create_session(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'.") from exc
    return {"session_id": session_id, "observation": observation, "info": info}


@app.post("/sessions/{session_id}/reset")
def reset_session(session_id: str, payload: ResetRequest) -> dict[str, object]:
    try:
        observation, info = get_registry().reset_session(session_id, payload.task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.") from exc
    return {"session_id": session_id, "observation": observation, "info": info}


@app.post("/sessions/{session_id}/step")
def step_session(session_id: str, payload: StepRequest) -> dict[str, object]:
    try:
        observation, reward, done, info = get_registry().step_session(session_id, payload.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.") from exc
    return {
        "session_id": session_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }
