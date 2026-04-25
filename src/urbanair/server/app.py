from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .env_factory import get_registry, list_tasks


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="DroneZ Runtime", version="0.1.0")
_DEFAULT_SESSION_ID: str | None = None
ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "demo_ui"
ARTIFACTS_DIR = ROOT / "artifacts"


def _ensure_default_session(task_id: str | None = None) -> tuple[str, dict[str, object], dict[str, object]]:
    global _DEFAULT_SESSION_ID

    registry = get_registry()
    if _DEFAULT_SESSION_ID is None or registry.get(_DEFAULT_SESSION_ID) is None:
        session_id, _, observation, info = registry.create_session(task_id)
        _DEFAULT_SESSION_ID = session_id
        return session_id, observation, info

    if task_id is not None:
        observation, info = registry.reset_session(_DEFAULT_SESSION_ID, task_id)
        return _DEFAULT_SESSION_ID, observation, info

    state = registry.state_session(_DEFAULT_SESSION_ID)
    return _DEFAULT_SESSION_ID, state["observation"], state["info"]


if DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=DEMO_DIR, html=True), name="demo")

if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")


def runtime_manifest() -> dict[str, object]:
    return {
        "name": "DroneZ OpenEnv Runtime",
        "status": "ok",
        "root": "/",
        "docs": "/docs",
        "health": "/health",
        "tasks": "/tasks",
        "default_reset": "/reset",
        "default_step": "/step",
        "default_state": "/state",
        "demo": "/demo/index.html" if DEMO_DIR.exists() else None,
        "artifacts": "/artifacts" if ARTIFACTS_DIR.exists() else None,
        "space": "https://huggingface.co/spaces/Krishna2521/dronez-openenv",
        "github": "https://github.com/SAICHAITU2012/Meta-Drone-Environment",
    }


@app.get("/", response_model=None)
def root():
    if DEMO_DIR.exists():
        return RedirectResponse(url="/demo/index.html", status_code=307)
    return runtime_manifest()


@app.get("/api")
def api_manifest() -> dict[str, object]:
    return runtime_manifest()


@app.get("/runtime")
def runtime() -> dict[str, object]:
    return runtime_manifest()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict[str, object]:
    available_tasks = list_tasks()
    default_task_id = "easy" if "easy" in available_tasks else (available_tasks[0] if available_tasks else None)
    return {"tasks": available_tasks, "default_task_id": default_task_id}


@app.post("/reset")
def reset_default(payload: ResetRequest) -> dict[str, object]:
    task_id = payload.task_id or None
    try:
        session_id, observation, info = _ensure_default_session(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown task '{task_id}'.") from exc
    return {"session_id": session_id, "observation": observation, "info": info}


@app.get("/state")
def state_default() -> dict[str, object]:
    session_id, _, _ = _ensure_default_session()
    try:
        state = get_registry().state_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Default session is unavailable.") from exc
    return {"session_id": session_id, "state": state}


@app.post("/step")
def step_default(payload: StepRequest) -> dict[str, object]:
    session_id, _, _ = _ensure_default_session()
    try:
        observation, reward, done, info = get_registry().step_session(session_id, payload.action)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Default session is unavailable.") from exc
    return {
        "session_id": session_id,
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


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


@app.get("/sessions/{session_id}/state")
def state_session(session_id: str) -> dict[str, object]:
    try:
        state = get_registry().state_session(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown session '{session_id}'.") from exc
    return {"session_id": session_id, "state": state}


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
