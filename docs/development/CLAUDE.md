# Development Notes

These are historical/internal notes from the early DroneZ build. They are kept under `docs/development/` so the public repo root stays clean.

## Source of Truth

- `docs/development/context/urbanair_architecture.md`
- `docs/development/context/urbanair_project_context.md`

These documents describe the original product direction, architecture, scope limits, and MVP priorities for DroneZ.

## Current Repository Shape

The public root is intentionally small:

- `README.md`, `BLOG.md`, `PITCH.md`, and `SUBMISSION_CHECKLIST.md` for judges and reviewers
- runtime files such as `Dockerfile`, `openenv.yaml`, `pyproject.toml`, and `requirements.txt`
- source code in `src/`, `server/`, `scripts/`, `configs/`, and `tests/`
- supporting learning, deployment, training, and report docs in `docs/`

## Architecture Reminder

DroneZ is organized around:

- simulation rules in `src/urbanair/sim/`
- OpenEnv wrapper, reward shaping, validation, and observations in `src/urbanair/env/`
- typed schemas in `src/urbanair/models.py` and `src/urbanair/enums.py`
- server entrypoints in `src/urbanair/server/` and `server/`
- baseline policies in `src/urbanair/policies/`
- evaluation in `src/urbanair/eval/`
- scenario and reward tuning in `configs/`

The project should stay focused on mission-level fleet operations, not real flight physics or certified aviation behavior.

## Useful Commands

```bash
python -m pip install -e .
pytest -q
python scripts/evaluate_policies.py
python scripts/generate_plots.py
openenv validate
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```
