# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth
- `context/urbanair_architecture.md`
- `context/urbanair_project_context.md`

These two documents define the product, architecture, scope limits, and MVP priorities for DroneZ. Treat them as authoritative when implementation details are unclear.

## Repository state
This repository is currently in bootstrap mode. The initial committed structure is being created from the two source docs before the environment implementation lands.

At this stage:
- the repo is centered on planning and scaffold files
- no stable build, lint, or test commands exist yet
- future code should be added in the modular structure described below rather than ad hoc top-level scripts

## Required workflow
- Keep `context/progress.md` updated with timestamped entries for each meaningful repository change.
- Commit and push meaningful milestones to the default branch.
- Keep repository-local Claude settings under `.claude/` and maintain worktree support there.
- Keep project docs in `context/`, but keep this `CLAUDE.md` at the repository root.

## Target architecture
Implement the project as a clean separation between:
- simulation rules in `src/urbanair/sim/`
- OpenEnv wrapper, validation, reward shaping, and observation formatting in `src/urbanair/env/`
- typed shared schemas in `src/urbanair/models.py` and `src/urbanair/enums.py`
- server entrypoints in `src/urbanair/server/`
- baselines in `src/urbanair/policies/`
- evaluation in `src/urbanair/eval/`
- scenario and reward tuning in `configs/`

The project should stay focused on mission-level fleet operations, not flight physics or visualization-heavy work.

## MVP build order
1. Bootstrap repo files and move source docs into `context/`.
2. Add `README.md`, `requirements.txt`, `openenv.yaml`, and config YAML files.
3. Implement typed models and reproducible seeding.
4. Implement the simulator core and order/fleet/city state transitions.
5. Implement the environment wrapper, action validation, reward engine, observation builder, and termination logic.
6. Add naive + heuristic baselines, benchmark/eval flow, and deterministic demo support.
7. Add the server and `inference.py` after local environment stepping is stable.

## Current useful commands
No project-specific build/lint/test commands are established yet.

Useful repository commands during bootstrap:
- `git status`
- `git diff --staged`
- `git push origin main`
- `python -m json.tool .claude/settings.json`

Update this section once runnable environment, evaluation, and test commands exist.
