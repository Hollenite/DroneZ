# DroneZ

DroneZ is an OpenEnv-compliant environment for training and evaluating an LLM agent as a **fleet operations controller** for autonomous urban delivery drones.

The environment focuses on mission-level decision making under disruption:
- heterogeneous drones
- order prioritization and reassignment
- charging tradeoffs
- weather drift and no-fly constraints
- delivery failures and recovery actions
- interpretable reward components

## Source of truth
The primary project specifications live in:
- `context/urbanair_architecture.md`
- `context/urbanair_project_context.md`

## Current status
The repository now includes the simulator core, environment wrapper, baseline evaluation flow, a minimal HTTP runtime, and a local `inference.py` entrypoint for running task episodes.

## Runtime entrypoints
- HTTP server app: `src/urbanair/server/app.py`
- Local inference runner: `inference.py`

Example local inference runs:
- `python inference.py --task easy --policy heuristic --summary-only`
- `python inference.py --task demo --policy naive --max-steps 8`

Example local server run:
- `uvicorn urbanair.server.app:app --app-dir src --reload`

Minimal HTTP endpoints:
- `GET /health`
- `GET /tasks`
- `POST /sessions`
- `POST /sessions/{session_id}/reset`
- `POST /sessions/{session_id}/step`

The server keeps per-session environment state in memory and supports the standard tasks, including `demo`, as a regular task.
