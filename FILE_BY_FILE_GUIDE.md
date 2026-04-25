# File By File Guide

This guide explains the important files in DroneZ, what they do, why they exist, and how they connect.

## Root Files

### `README.md`

This is the main project page. Judges and new developers read this first.

It explains:

- What DroneZ is.
- Why it fits OpenEnv.
- How the environment works.
- What the reward means.
- What policies are compared.
- What training status is honest right now.
- How to run the demo and API.

### `openenv.yaml`

This is the OpenEnv manifest.

It tells OpenEnv-compatible tooling:

- The environment name.
- The runtime entrypoint.
- The reset, step, state, and health endpoints.
- Metadata about the project.

Think of it as the environment ID card.

### `Dockerfile`

This tells Docker how to build the app.

Hugging Face Spaces uses this file to start the DroneZ server. It installs dependencies, copies the repo, and runs the FastAPI application.

### `pyproject.toml`

This is the Python project configuration.

It defines:

- Project name and metadata.
- Package source layout.
- Dependencies.
- Optional training dependencies.
- Test configuration.

### `requirements.txt`

This is a simpler dependency list for platforms that install with `pip install -r requirements.txt`.

## Server Files

### `src/urbanair/server/app.py`

This is the FastAPI server.

It exposes the API endpoints:

- `/health`
- `/tasks`
- `/reset`
- `/step`
- `/state`
- `/api`
- `/docs`
- `/demo/index.html`

It also serves static demo files from `demo_ui/` and artifacts from `artifacts/`.

### `src/urbanair/server/env_factory.py`

This creates environment instances for the server.

The server should not manually build every simulation object. It asks the factory for a ready DroneZ environment.

### `src/urbanair/server/cli.py`

This provides command-line startup helpers for the server.

## Environment Files

### `src/urbanair/env/environment.py`

This is the main RL environment class.

It owns the `reset`, `step`, and `state` logic. When an agent sends an action, this file coordinates the simulation engine, action router, reward engine, observation builder, and termination checks.

### `src/urbanair/env/action_router.py`

This validates and applies actions.

It checks whether an action such as `assign_delivery`, `reroute`, or `attempt_delivery` is legal in the current state.

This file is important because the LLM must produce valid JSON actions. Invalid actions get penalized.

### `src/urbanair/env/observation_builder.py`

This builds the observation returned to the agent.

It gathers:

- Fleet state.
- Orders.
- City sectors.
- Charging stations.
- Recent events.
- Warnings.
- Human-readable summary.

### `src/urbanair/env/reward_engine.py`

This calculates reward.

It adds points for good behavior and subtracts points for bad behavior. It also keeps reward breakdowns so humans can inspect why a policy scored well or badly.

### `src/urbanair/env/termination.py`

This decides when an episode should end.

Examples:

- All orders are resolved.
- Horizon reached.
- Invalid action cap reached.
- Action cap reached.
- No viable drones remain.

## Simulation Files

### `src/urbanair/sim/engine.py`

This is the simulation engine.

It advances the world after an action is applied. It updates drones, orders, city conditions, charging state, and events.

### `src/urbanair/sim/city.py`

This defines the city sectors.

Sectors can have weather, congestion, no-fly state, operation holds, and risk.

### `src/urbanair/sim/fleet.py`

This defines drone objects and fleet behavior.

It tracks battery, position, assignments, health risk, target zone, and status.

### `src/urbanair/sim/orders.py`

This defines delivery orders.

Orders have priority, zone, deadline, status, fallback options, and delivery state.

### `src/urbanair/sim/delivery_logic.py`

This contains delivery-related rules.

It helps decide what happens when a drone attempts a delivery, fails, or uses locker fallback.

### `src/urbanair/sim/disruptions.py`

This handles disruptions such as weather and no-fly changes.

### `src/urbanair/sim/scripted_events.py`

This creates deterministic events for demos.

For example, the demo can inject an urgent order or no-fly shift at a known step so the replay is reproducible.

## Policy Files

### `src/urbanair/policies/base.py`

This defines the basic policy interface.

A policy receives an observation and returns an action.

### `src/urbanair/policies/baseline.py`

This contains the policies used for comparison:

- `random`
- `naive`
- `heuristic`
- `improved`

The improved policy is deterministic and scripted. It is not a trained neural model. It proves the environment can distinguish better decisions from worse ones.

## Evaluation Files

### `src/urbanair/eval/benchmark.py`

Runs policies across tasks and collects results.

### `src/urbanair/eval/metrics.py`

Defines the metrics used in reports.

Examples:

- Total reward.
- Normalized score.
- Completed deliveries.
- Safety violations.
- Invalid actions.

### `src/urbanair/eval/report.py`

Formats evaluation results for files and human reading.

### `scripts/evaluate_policies.py`

Runs all main policies and writes comparison artifacts.

### `scripts/generate_demo_trace.py`

Runs a policy on a task and saves the frame-by-frame trace used by the browser demo.

### `scripts/generate_plots.py`

Creates plots for rewards, delivery success, invalid actions, and training diagnostics.

### `scripts/enrich_demo_trace.py`

Adds visualization metadata to existing traces.

It creates simulated telemetry for the UI:

- Drone display positions.
- Curved route segments.
- Route risk.
- Weather overlays.
- Wind speed.
- Sensor status.
- Control tower status.

These are labeled as visualization metadata, not real physical drone telemetry.

## Training Files

### `scripts/train_grpo.py`

Provides smoke, dry-run, and template training modes.

It is useful for proving the environment loop and prompt interface without claiming real model improvement.

### `scripts/train_grpo_local.py`

This is the local GPU GRPO-style training path.

It supports:

- `--sanity-check`
- `--format-check`
- `--candidate-choice`
- `--warmstart-data`
- `--real-train`

It correctly refuses real training if CUDA is unavailable.

### `scripts/train_grpo_colab.py`

This helps prepare the Colab training path.

Colab is useful because LLM RL usually needs a GPU.

### `scripts/generate_sft_action_data.py`

This generates supervised action-format examples from the improved policy.

It writes observation-to-action examples so a model can learn the correct JSON format before GRPO.

### `scripts/train_action_format_sft.py`

This is the optional action-format SFT entrypoint.

If dependencies or GPU are missing, it should run in dry-run/template mode and not fake results.

## Demo UI Files

### `demo_ui/index.html`

This defines the browser page layout.

It includes:

- Hero section.
- Replay controls.
- B2B organization configurator.
- Simulation map.
- Telemetry panel.
- Weather and airspace panel.
- Control tower panel.
- Reward and event panels.
- Beginner explanation panel.

### `demo_ui/app.js`

This loads trace JSON and renders the live simulation.

It draws:

- City zones.
- Drones.
- Orders.
- Chargers.
- No-fly/weather zones.
- Curved route corridors.
- Telemetry.
- Control tower state.
- Reward and event updates.

### `demo_ui/styles.css`

This styles the demo as a high-tech B2B control tower.

It is static CSS, so it runs reliably on Hugging Face Spaces without a frontend build system.

## Artifact Files

### `artifacts/results/`

Contains policy comparison results as JSON and CSV.

### `artifacts/traces/`

Contains frame-by-frame traces for demo replay.

The enriched trace files are used by the advanced simulation UI.

### `artifacts/plots/`

Contains charts for rewards, deliveries, invalid actions, and training diagnostics.

### `artifacts/training/`

Contains training diagnostics, previous training artifacts, format checks, sanity checks, and SFT action data.

