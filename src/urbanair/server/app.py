from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from urbanair.env.environment import DroneZEnvironment
from .env_factory import get_registry, list_tasks
from urbanair.training.action_format import ACTION_SCHEMAS, build_candidate_actions


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(default_factory=dict)


class ValidActionsRequest(BaseModel):
    session_id: str | None = None
    observation: dict[str, Any] | None = None


app = FastAPI(title="DroneZ Runtime", version="0.1.0")
_DEFAULT_SESSION_ID: str | None = None
ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "demo_ui"
AGENT_DIR = ROOT / "agent_ui"
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

if AGENT_DIR.exists():
    app.mount("/agent", StaticFiles(directory=AGENT_DIR, html=True), name="agent")

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
        "env_spec": "/env-spec",
        "action_space": "/action-space",
        "observation_space": "/observation-space",
        "valid_actions": "/valid-actions",
        "examples": "/examples",
        "agent_playground": "/agent/index.html" if AGENT_DIR.exists() else None,
        "demo": "/demo/index.html" if DEMO_DIR.exists() else None,
        "artifacts": "/artifacts" if ARTIFACTS_DIR.exists() else None,
        "space": "https://huggingface.co/spaces/Krishna2521/dronez-openenv",
        "github": "https://github.com/Hollenite/DroneZ",
    }


def json_dumps_compact(payload: object) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _example_param_value(param: str) -> str:
    examples = {
        "drone_id": "FA-1",
        "order_id": "O1",
        "corridor": "safe",
        "station_id": "C_HUB",
        "drone_a": "FA-1",
        "drone_b": "HE-1",
        "mode": "handoff",
        "locker_id": "L-Z1",
        "zone_id": "Z1",
    }
    return examples.get(param, f"example_{param}")


def _valid_action_items(observation: dict[str, Any]) -> list[dict[str, object]]:
    candidates = build_candidate_actions(observation)
    return [
        {
            "index": index,
            "action": action,
            "reason": _describe_action(action),
            "priority": _action_priority(action),
            "risk": _action_risk(action, observation),
        }
        for index, action in enumerate(candidates, start=1)
    ]


def _describe_action(action: dict[str, Any]) -> str:
    name = action.get("action", "unknown")
    params = action.get("params", {})
    if name == "assign_delivery":
        return f"Assign drone {params.get('drone_id')} to order {params.get('order_id')}."
    if name == "attempt_delivery":
        return f"Attempt delivery handoff with drone {params.get('drone_id')}."
    if name == "reroute":
        return f"Move drone {params.get('drone_id')} onto the {params.get('corridor')} route corridor."
    if name == "prioritize_order":
        return f"Prioritize order {params.get('order_id')} before lower-urgency work."
    if name == "fallback_to_locker":
        return f"Use locker {params.get('locker_id')} for order {params.get('order_id')}."
    if name in {"return_to_charge", "reserve_charger"}:
        return f"Send or reserve charging for drone {params.get('drone_id')} at {params.get('station_id')}."
    if name in {"hold_fleet", "resume_operations"}:
        return f"{name.replace('_', ' ').title()} in zone {params.get('zone_id')}."
    return f"Candidate mission action: {name}."


def _action_priority(action: dict[str, Any]) -> str:
    name = action.get("action")
    if name in {"prioritize_order", "assign_delivery", "attempt_delivery"}:
        return "high"
    if name in {"reroute", "fallback_to_locker", "return_to_charge"}:
        return "medium"
    return "normal"


def _action_risk(action: dict[str, Any], observation: dict[str, Any]) -> str:
    params = action.get("params", {})
    drone_id = params.get("drone_id")
    if drone_id:
        for drone in observation.get("fleet", []):
            if drone.get("drone_id") == drone_id and float(drone.get("battery", 100) or 100) <= 25:
                return "battery_watch"
    if action.get("action") == "reroute":
        return "route_change"
    return "low"


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


@app.get("/artifacts/traces/{trace_name}")
def trace_artifact(trace_name: str):
    """Serve trace files with a clear 404 instead of a silent static fallback miss."""
    if "/" in trace_name or "\\" in trace_name or not trace_name.endswith(".json"):
        raise HTTPException(status_code=404, detail="Trace artifact not found")
    path = ARTIFACTS_DIR / "traces" / trace_name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Trace artifact not found: {trace_name}")
    return FileResponse(path, media_type="application/json")


@app.get("/tasks")
def tasks() -> dict[str, object]:
    available_tasks = list_tasks()
    default_task_id = "easy" if "easy" in available_tasks else (available_tasks[0] if available_tasks else None)
    return {"tasks": available_tasks, "default_task_id": default_task_id}


@app.get("/env-spec")
def env_spec() -> dict[str, object]:
    example_env = DroneZEnvironment()
    observation, info = example_env.reset("easy")
    example_action = _valid_action_items(observation)[0]["action"]
    return {
        "name": "DroneZ Fleet Delivery Environment",
        "version": app.version,
        "runtime": "OpenEnv-compatible FastAPI runtime",
        "tasks": list_tasks(),
        "openenv_loop": {
            "summary": "An agent calls reset(), receives an observation, sends one JSON action to step(), receives reward/done/info, and repeats.",
            "sequence": ["reset", "observation", "action", "step", "reward", "done/info", "learn"],
        },
        "observation_description": {
            "fleet": "Drone status, battery, zone, assignment, ETA, corridor, and health/risk metadata.",
            "orders": "Delivery orders with priority, deadline, target zone, recipient availability, package weight, and assignment.",
            "city": "Sectors with weather, congestion, no-fly/paused status, and operational hazards.",
            "charging": "Charging station capacity, occupied slots, and queue information.",
            "events": "Recent disruptions or scripted environment events.",
            "reward": "Current/cumulative reward breakdown is exposed through info and observation metadata when available.",
        },
        "action_description": {
            "format": {"action": "assign_delivery", "params": {"drone_id": "FA-1", "order_id": "O1"}},
            "allowed_actions": list(ACTION_SCHEMAS.keys()),
            "discovery": "POST /valid-actions returns currently valid candidate actions for the active observation.",
        },
        "reward_description": {
            "positive": [
                "successful delivery",
                "urgent delivery success",
                "deadline completion",
                "safe reroute",
                "battery-safe operation",
                "fleet utilization",
                "regulatory compliance",
                "locker fallback",
            ],
            "negative": [
                "invalid action",
                "missed deadline",
                "failed delivery",
                "critical battery event",
                "unsafe zone entry",
                "idle/no-progress behavior",
                "charging misuse",
            ],
        },
        "termination_conditions": [
            "episode horizon reached",
            "all deliveries resolved",
            "invalid action cap reached",
            "runner action cap reached during evaluation",
        ],
        "example_reset": {
            "request": {"task_id": "easy"},
            "response_preview": {
                "session_id": "example-session-id",
                "observation_keys": sorted(observation.keys()),
                "info_keys": sorted(info.keys()),
            },
        },
        "example_step": {
            "request": {"action": example_action},
            "response_fields": ["session_id", "observation", "reward", "done", "info"],
        },
        "training_status": {
            "dry_run": "passed",
            "format_check": {"valid_json_rate": 0.875, "valid_action_rate": 0.875},
            "sft_examples": 108,
            "safe_candidate_choice_training": "runs without CUDA sampling crash in Colab smoke path",
            "reward_improvement": "not proven; do not claim trained-model reward improvement unless eval_after beats eval_before",
        },
        "links": {
            "docs": "/docs",
            "demo": "/demo/index.html",
            "agent_playground": "/agent/index.html",
            "colab": "https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV",
            "staging_space": "https://huggingface.co/spaces/Krishna2521/Meta_Drone_env",
            "official_submitted_space": "https://huggingface.co/spaces/Krishna2521/dronez-openenv",
        },
    }


@app.get("/action-space")
def action_space() -> dict[str, object]:
    actions = []
    for action_name, params in ACTION_SCHEMAS.items():
        example_params = {param: _example_param_value(param) for param in params}
        actions.append(
            {
                "action": action_name,
                "required_params": list(params),
                "example": {"action": action_name, "params": example_params},
            }
        )
    return {
        "format": {"action": "action_name", "params": {"key": "value"}},
        "actions": actions,
        "candidate_choice_note": "candidate-choice training can ask the model to emit {\"choice\": 1} from /valid-actions instead of inventing action JSON from scratch.",
        "valid_action_discovery": "/valid-actions",
    }


@app.get("/observation-space")
def observation_space() -> dict[str, object]:
    return {
        "description": "Observation is a JSON snapshot of mission-level drone fleet state.",
        "fields": {
            "task_id": "Current task/scenario id.",
            "step": "Current environment step.",
            "max_steps": "Episode horizon.",
            "fleet": {
                "description": "List of drones.",
                "common_fields": ["drone_id", "drone_type", "status", "battery", "current_zone", "assigned_order_id", "eta", "target_zone", "active_corridor"],
            },
            "orders": {
                "description": "List of delivery orders.",
                "common_fields": ["order_id", "priority", "status", "zone_id", "deadline", "recipient_availability", "assigned_drone_id", "package_weight"],
            },
            "city": {
                "description": "City sectors and constraints.",
                "common_fields": ["zone_id", "weather", "congestion_score", "is_no_fly", "operations_paused"],
            },
            "charging": {
                "description": "Charging station status.",
                "common_fields": ["station_id", "capacity", "occupied_slots", "queue_size"],
            },
            "events": "Recent environment events or disruptions if present.",
            "warnings": "Operational warnings if present.",
            "reward_breakdown": "Reward terms may appear in info/observation depending on step state.",
        },
    }


@app.post("/valid-actions")
def valid_actions(payload: ValidActionsRequest | None = None) -> dict[str, object]:
    request = payload or ValidActionsRequest()
    if request.observation is not None:
        observation = request.observation
        session_id = request.session_id or None
    elif request.session_id:
        try:
            state = get_registry().state_session(request.session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Unknown session '{request.session_id}'.") from exc
        session_id = request.session_id
        observation = state["observation"]
    else:
        session_id, observation, _ = _ensure_default_session()
    return {
        "session_id": session_id,
        "task_id": observation.get("task_id"),
        "step": observation.get("step"),
        "actions": _valid_action_items(observation),
    }


@app.get("/examples")
def examples() -> dict[str, object]:
    base = "https://krishna2521-meta-drone-env.hf.space"
    action = {"action": "prioritize_order", "params": {"order_id": "O1"}}
    return {
        "curl_reset": f"curl -X POST {base}/reset -H 'Content-Type: application/json' -d '{{\"task_id\":\"easy\"}}'",
        "curl_valid_actions": f"curl -X POST {base}/valid-actions -H 'Content-Type: application/json' -d '{{}}'",
        "curl_step": f"curl -X POST {base}/step -H 'Content-Type: application/json' -d '{{\"action\":{json_dumps_compact(action)}}}'",
        "python_requests_loop": (
            "import requests\n"
            f"BASE = '{base}'\n"
            "obs = requests.post(f'{BASE}/reset', json={'task_id': 'easy'}).json()\n"
            "for _ in range(20):\n"
            "    candidates = requests.post(f'{BASE}/valid-actions', json={}).json()['actions']\n"
            "    action = candidates[0]['action']\n"
            "    result = requests.post(f'{BASE}/step', json={'action': action}).json()\n"
            "    print(result['reward'], result['done'], result['info'].get('done_reason'))\n"
            "    if result['done']:\n"
            "        break\n"
        ),
        "random_agent_file": "examples/random_remote_agent.py",
        "quickstart_file": "examples/remote_agent_quickstart.py",
        "colab": "https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV",
    }


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


if ARTIFACTS_DIR.exists():
    app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")
