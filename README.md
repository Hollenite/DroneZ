---
title: DroneZ OpenEnv
sdk: docker
app_port: 7860
colorFrom: blue
colorTo: green
pinned: false
license: mit
---

# DroneZ Fleet Delivery Environment

DroneZ is an OpenEnv RL environment for training and evaluating mission-level drone fleet controllers under weather, no-fly zones, battery limits, urgent orders, and delivery uncertainty.

## Submission Links

- Hugging Face Space Repository: https://huggingface.co/spaces/Krishna2521/dronez-openenv
- Live Runtime: https://krishna2521-dronez-openenv.hf.space
- Live Demo: https://krishna2521-dronez-openenv.hf.space/demo/index.html
- API Docs: https://krishna2521-dronez-openenv.hf.space/docs
- Health Check: https://krishna2521-dronez-openenv.hf.space/health
- GitHub Team Repo: https://github.com/Hollenite/DroneZ.git
- Colab Notebook Path: `notebooks/train_dronez_grpo_colab.ipynb`
- Public Colab Link: https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV
- Blog / Writeup: `BLOG.md` in this repo. Public blog URL: `TODO - add before final form submission if published externally`.
- YouTube Demo: `TODO - add before final form submission`.
- Slides / Presentation: `TODO - add before final form submission`.

## Introduction

When users order something online, they expect delivery to be fast and reliable. Drones can make delivery faster because they avoid road traffic, but real drone delivery is not automatically optimal. A drone fleet must handle weather, no-fly zones, battery limits, obstacles, urgent orders, changing recipients, charging congestion, and safety rules.

DroneZ focuses on the decision-making layer. It asks: how should a fleet controller assign drones, choose routes, reroute around disruptions, send drones to charge, prioritize urgent orders, and recover from failed delivery attempts?

## What DroneZ Actually Does

DroneZ is not a low-level motor-control simulator. It does not train an RL model to directly spin propellers.

DroneZ is a mission-level drone fleet operations environment:

- Classical drone systems handle stability, PID loops, GPS navigation, sensor fusion, Kalman-style state estimation, geofencing, and emergency safety rules.
- RL/AI handles high-level decisions such as assignment, routing, charging, recovery, urgent-order priority, and fleet-level mission optimization.
- The browser demo visualizes this as a hybrid drone control tower: low-level drone control, high-level RL/AI control, and organization-level fleet monitoring.

This boundary is important. It makes the project realistic: real delivery drones are hybrid systems, not pure RL robots.

## OpenEnv Loop

DroneZ follows the standard OpenEnv interaction loop:

1. `reset` starts a fresh scenario.
2. The agent receives an observation describing drones, orders, sectors, chargers, weather, warnings, and recent events.
3. The agent sends one structured action.
4. `step` applies that action, advances the simulated world, and returns reward plus the next observation.
5. Repeating this loop lets policies be evaluated and improved.

In simple words: the agent sees the fleet, chooses a mission decision, receives a score, and tries to make better decisions over time.

## Environment Design

Each episode contains:

- Drones with battery, location, route, payload capacity, health state, and assignment state.
- Orders with priority, deadline, destination, availability, fallback options, and retry state.
- City sectors with weather, congestion, no-fly restrictions, and safety risk.
- Charging stations with limited capacity and reservations.
- Disruptions such as storms, restricted sectors, failed drops, and urgent-order changes.

The environment is designed to reward safe, useful long-horizon decisions rather than one-step shortcuts.

## Action Space

The main actions are:

- `assign_delivery`: assign an order to a drone.
- `attempt_delivery`: attempt a drop when a drone reaches its destination.
- `reroute`: change a drone route to avoid risk or recover from disruption.
- `prioritize_order`: move an urgent or important order higher in the queue.
- `fallback_to_locker`: send a package to a safe locker when the recipient is unavailable.
- `return_to_charge`: send a drone to a charging station.
- `reserve_charger`: reserve charging capacity.
- `hold_fleet`: pause operations in unsafe conditions.
- `resume_operations`: restart operations after a hold.
- `delay_order` and `swap_assignments`: handle scheduling or assignment changes.

## Reward System

Positive rewards are given for:

- successful deliveries
- urgent delivery success
- completing deliveries before deadlines
- safe reroutes
- recovering from disruptions
- battery-safe operation
- efficient fleet usage
- regulatory compliance
- successful locker fallback

Negative penalties are given for:

- invalid actions
- missed deadlines
- failed delivery attempts
- critical battery events
- unsafe zone entry
- unnecessary reroutes
- abandoned urgent orders
- idle fleet behavior while orders are waiting
- charging misuse
- loop or no-progress behavior

The environment also includes safeguards: invalid actions are capped, episodes have action limits, deadline penalties are not double-counted, and reward breakdowns are logged for inspection.

## Evaluation Results

Results are generated from `artifacts/results/policy_comparison.json` and `artifacts/results/policy_comparison.csv`.

### Aggregate Evaluation Across Easy, Medium, Hard, and Demo

| Policy | Mean Reward | Mean Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | **-42.125** | **0.3051** | 5 | 3 | **0** | 0 |
| heuristic | -281.500 | 0.0797 | 10 | 5 | 182 | 0 |
| random | -921.500 | 0.0100 | 9 | 4 | 396 | 0 |
| naive | -1038.000 | 0.0100 | 6 | 3 | 497 | 0 |

The aggregate result shows that `improved` has the best mean reward, best normalized score, and zero safety violations. The heuristic baseline completes more deliveries overall, but it does so with much higher safety cost.

### Demo Scenario Headline

| Policy | Reward | Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | **89.0** | **0.5861** | 2 | 1 | **0** | 0 |
| heuristic | 32.0 | 0.2889 | 2 | 1 | 8 | 0 |
| random | -162.5 | 0.0100 | 2 | 1 | 33 | 0 |
| naive | -607.5 | 0.0100 | 0 | 0 | 72 | 0 |

Demo takeaway: the deterministic improved policy keeps the same delivery count as the heuristic baseline while reducing safety violations from `8` to `0` and increasing reward from `32.0` to `89.0`.

## Training Status

The current proven improvement is deterministic policy improvement, not trained-model reward improvement.

A real GRPO-style training run was attempted on an NVIDIA RTX 5060 Laptop GPU using `Qwen/Qwen2.5-0.5B-Instruct`. That run did not improve reward. The model produced invalid DroneZ actions, episodes ended with `invalid_action_cap_reached`, reward stayed flat, loss stayed `0.0`, and `eval_before` matched `eval_after`.

The project now includes fixes for that bottleneck:

- compact action prompts
- robust JSON parsing
- safe action repair
- candidate-choice mode, where the model can choose from valid actions instead of inventing JSON from scratch
- format-check diagnostics
- SFT warm-start data from improved-policy traces

Current format diagnostics in `artifacts/training/format_check/format_check.json` show:

- `valid_json_rate`: `0.875`
- `valid_action_rate`: `0.875`

This is format reliability progress. It is not yet proof that a trained model improves reward. The README, pitch, and demo intentionally do not claim trained-model improvement.

## Training Logs and Artifacts

Small training and diagnostic artifacts are included:

- `artifacts/training/local_sanity/sanity_check.json`
- `artifacts/training/format_check/format_check.json`
- `artifacts/training/sft_action_data.jsonl`
- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`
- `artifacts/training/candidate_grpo/training_metrics.json`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/training_loss_curve.png`
- `artifacts/plots/valid_action_rate.png`
- `artifacts/plots/eval_before_after.png`

Training entrypoints:

- `scripts/train_grpo.py`
- `scripts/train_grpo_local.py`
- `scripts/train_grpo_colab.py`
- `scripts/generate_sft_action_data.py`
- `scripts/train_action_format_sft.py`
- `notebooks/train_dronez_grpo_colab.ipynb`

## Interactive Demonstration Interface

The browser replay UI uses real JSON traces from the environment. It presents DroneZ as a high-tech hybrid-drone control tower rather than a simple grid.

Key visual features:

- 2.5D procedural city map
- curved route corridors
- active drones and delivery orders
- charging stations
- weather overlays
- no-fly zones
- simulated telemetry
- reward evolution tracking
- recent events feed
- control tower state monitoring
- Stage Demo Mode for cleaner presentation

Route geometry, wind values, sensor indicators, and control-layer labels are derived visualization metadata. They are useful for explaining the simulated mission, but they are not real aircraft telemetry.

## Standalone Cinematic Simulator

The repo also includes a separate local simulator in `dronez_world_sim/`. It is intended for presentation and visual storytelling, while the OpenEnv runtime remains the authoritative environment.

Run it locally from the repo root:

```bash
python -m http.server 8090
```

Then open:

```text
http://127.0.0.1:8090/dronez_world_sim/index.html
```

The standalone simulator includes a larger 3D-style world, drone switching, day/night mode, mission routes, a minimap, telemetry cards, and trace-aware scenario controls.

## Running Locally

Install the package:

```bash
python -m pip install -e .
```

Run the FastAPI/OpenEnv server:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/demo/index.html
http://127.0.0.1:8000/docs
```

Useful checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/tasks
curl http://127.0.0.1:8000/api
curl http://127.0.0.1:8000/artifacts/traces/demo_improved_enriched.json
```

## Docker

Build and run:

```bash
docker build -t dronez .
docker run --rm -p 8000:7860 dronez
```

Then open:

```text
http://127.0.0.1:8000
```

## Regenerating Results

```bash
python scripts/evaluate_policies.py
python scripts/generate_demo_trace.py --task easy --policy all
python scripts/generate_demo_trace.py --task medium --policy all
python scripts/generate_demo_trace.py --task hard --policy all
python scripts/generate_demo_trace.py --task demo --policy all
python scripts/enrich_demo_trace.py --task easy --policy all
python scripts/enrich_demo_trace.py --task medium --policy all
python scripts/enrich_demo_trace.py --task hard --policy all
python scripts/enrich_demo_trace.py --task demo --policy all
python scripts/generate_plots.py
```

Training diagnostics:

```bash
python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity
python scripts/train_grpo_local.py --format-check --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,demo --output-dir artifacts/training/format_check
python scripts/generate_sft_action_data.py --tasks easy,medium,demo,hard --output artifacts/training/sft_action_data.jsonl
```

## Hugging Face Space

The Hugging Face Space is the main evaluation target:

- Space repo: https://huggingface.co/spaces/Krishna2521/dronez-openenv
- Runtime: https://krishna2521-dronez-openenv.hf.space
- Demo: https://krishna2521-dronez-openenv.hf.space/demo/index.html
- Docs: https://krishna2521-dronez-openenv.hf.space/docs

## Limitations

DroneZ does not yet simulate:

- full Isaac Sim, Gazebo, AirSim, or real aerodynamics
- real propeller or motor control
- certified aviation behavior
- real GPS, camera, LiDAR, radar, or thermal streams
- proven trained-model reward improvement from GRPO

The current honest status is:

- Environment: working
- OpenEnv API: working
- Docker/HF Space deployment: working
- Deterministic improved policy: proven better than baselines on reward and safety
- Real GRPO improvement: attempted, but not proven yet
- Training pipeline: improved with candidate-choice, format diagnostics, and SFT warm-start data

## Conclusion

Drone delivery is not just about flying. It is about decision-making under uncertainty. DroneZ shifts the focus from fixed routes to adaptive fleet intelligence, where actions are evaluated through rewards and policies can move toward safer, faster, and more reliable delivery operations.
