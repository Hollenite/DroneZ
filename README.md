---
title: DroneZ OpenEnv
sdk: docker
app_port: 7860
colorFrom: blue
colorTo: green
pinned: false
license: mit
---

# DroneZ

DroneZ is an OpenEnv-style RL environment for training an LLM to act as a mission-level fleet operations controller for autonomous delivery drones.

## 1. Problem

Drone delivery does not fail because of pathfinding alone. Real operations break when urgent orders arrive late, weather shifts, no-fly zones move, batteries drain unevenly, recipients disappear, chargers back up, and the controller must continuously replan across a heterogeneous fleet.

DroneZ models that operational layer. The agent is not piloting a drone. It is deciding which drone should take which mission, when to reroute, when to hold a zone, when to send a drone to charge, and how to recover from failed drops.

## 2. Why This Is an OpenEnv / RL Problem

DroneZ is designed around the exact loop OpenEnv is meant to standardize:

1. Reset into a fresh logistics scenario.
2. Observe structured fleet, order, city, and disruption state.
3. Choose one action at a time.
4. Let the environment advance and return reward, done, and next observation.
5. Improve the policy through repeated interaction, not static labels alone.

The environment is stateful, partially observable, multi-step, and reward-driven. That makes it a strong fit for OpenEnv and for verifiable RL pipelines such as GRPO.

## 3. Theme Fit

- Primary: Theme `#3.1` World Modeling / Professional Tasks
- Secondary: Theme `#1` Multi-Agent Interactions
- Optional: Theme `#2` Long-Horizon Planning

## 4. Environment Overview

The agent controls a heterogeneous fleet:

- `fast_light`
- `heavy_carrier`
- `long_range_sensitive`
- `relay`

Each episode tracks:

- fleet battery, position, assignment, route, and risk
- order priority, deadlines, availability, fallback options, and retries
- sector weather, congestion, no-fly state, and manual holds
- charging occupancy and reservations
- scripted and stochastic disruptions

### Reset / Step / State

- `reset(task_id)` starts a scenario
- `step(action)` advances the world
- `state` is exposed through the HTTP runtime at `GET /sessions/{id}/state`

## 5. Observation Space

Each observation includes:

- fleet states
- order queue
- city sectors
- charging stations
- recent events
- warnings
- action reminder
- summary string for human-readable debugging

## 6. Action Space

Current supported actions:

- `assign_delivery`
- `reroute`
- `return_to_charge`
- `reserve_charger`
- `delay_order`
- `prioritize_order`
- `swap_assignments`
- `attempt_delivery`
- `fallback_to_locker`
- `hold_fleet`
- `resume_operations`

## 7. Termination Conditions

Episodes terminate when:

- all orders are resolved
- horizon is reached
- no viable drones remain
- invalid action cap is reached
- action cap is reached

The environment and evaluator both enforce safety caps so policy bugs cannot stall the demo.

## 8. Reward Design

### Positive components

- `delivery_success`
- `urgent_delivery_success`
- `deadline_met`
- `safe_reroute`
- `disruption_recovery`
- `battery_safe_operation`
- `efficient_assignment`
- `fleet_utilization`
- `regulatory_compliance`
- `successful_locker_fallback`

### Negative components

- `invalid_action`
- `missed_deadline`
- `failed_delivery_attempt`
- `battery_critical`
- `unsafe_zone_entry`
- `unnecessary_reroute`
- `abandoned_urgent_order`
- `idle_with_pending_orders`
- `overloaded_assignment`
- `charging_misuse`
- `loop_or_no_progress`

### Anti-Reward-Hacking Safeguards

- invalid actions are penalized and capped
- action count is capped independently from environment ticks
- safety reroutes are not double-penalized as unnecessary reroutes
- deadline misses only count once when a deadline is first crossed
- deterministic demo scenarios make judge replays reproducible
- reward breakdowns are logged for every step and episode
- normalized scores are clamped into `[0.01, 0.99]`

## 9. Current Policies

- `random`: deterministic pseudo-random baseline
- `naive`: simple greedy controller
- `heuristic`: stable hand-built baseline
- `improved`: deterministic scripted policy for judge-facing demo and replay

## 10. Evaluation

Generated artifacts:

- `artifacts/results/policy_comparison.json`
- `artifacts/results/policy_comparison.csv`
- `artifacts/traces/demo_random_trace.json`
- `artifacts/traces/demo_naive_trace.json`
- `artifacts/traces/demo_heuristic_trace.json`
- `artifacts/traces/demo_improved_trace.json`
- `artifacts/plots/reward_comparison.png`
- `artifacts/plots/delivery_success_comparison.png`
- `artifacts/plots/invalid_actions_comparison.png`

Current deterministic comparison snapshot:

| Task | Random | Naive | Heuristic | Improved |
| --- | ---: | ---: | ---: | ---: |
| Easy reward | -278.0 | -315.5 | -154.0 | **35.0** |
| Medium reward | -1231.5 | -1259.5 | -439.5 | **-77.0** |
| Hard reward | -2014.0 | -1969.5 | -564.5 | **-215.5** |
| Demo reward | -162.5 | -607.5 | 32.0 | **89.0** |

Current aggregate story:

- `improved` has the best mean reward and best normalized score
- `improved` keeps `0` invalid actions and `0` safety violations across the sweep
- `heuristic` completes more deliveries, but does so with much higher safety cost
- this repo currently proves deterministic policy improvement, not trained-model improvement

Demo headline:

| Policy | Reward | Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | **89.0** | **0.5861** | 2 | 1 | **0** | 0 |
| heuristic | 32.0 | 0.2889 | 2 | 1 | 8 | 0 |
| random | -162.5 | 0.0100 | 2 | 1 | 33 | 0 |
| naive | -607.5 | 0.0100 | 0 | 0 | 72 | 0 |

Run them with:

```bash
python scripts/evaluate_policies.py
python scripts/generate_demo_trace.py --task demo --policy all
python scripts/generate_plots.py
```

Plot references:

- `artifacts/plots/reward_comparison.png`
- `artifacts/plots/delivery_success_comparison.png`
- `artifacts/plots/invalid_actions_comparison.png`

## 11. Training Pipeline

This repo currently includes:

- `scripts/train_grpo.py --mode smoke`
- `scripts/train_grpo.py --mode dry-run`
- `scripts/train_grpo.py --mode trl-template`
- `scripts/train_grpo_local.py`
- `scripts/train_grpo_colab.py`
- `notebooks/train_dronez_grpo_colab.ipynb`
- `COLAB_TRAINING.md`

What is honest right now:

- the smoke mode uses the actual environment loop locally
- the dry-run mode prepares the real prompt/action interface and writes reference artifacts
- `scripts/train_grpo_local.py` is the dedicated local GPU path for a real GRPO-style run
- the Colab entrypoint remains a dependency-gated template wrapper
- no fake trained-model claims or fake GRPO results are included in tracked artifacts

Honesty box:

- Current measured improvement in this repo: deterministic `improved` policy vs baselines
- Real GRPO / Unsloth training has still not been executed in this tracked repo state
- Update this section only after a real run produces `eval_before`, `eval_after`, and training plots

Recommended next step onsite:

1. Install train extras: `pip install -e .[train]`
2. Run `python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity`
3. Run `python scripts/train_grpo_local.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --eval-tasks easy,medium,demo,hard --output-dir artifacts/training` on a local GPU machine
4. Save real reward curves and before/after comparisons back into `artifacts/`
5. Replace the honesty box with real training evidence only after the run finishes

Notes on the new local script:

- `--sanity-check` writes honest dependency/GPU metadata without downloading a model or claiming training happened.
- The real run writes canonical `training_metrics.json`, `eval_before.json`, and `eval_after.json` only after actual optimizer updates complete.
- Large checkpoints or model weights should stay out of the git commit unless you intentionally want to publish them.

Fallback Colab path:

- `python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training` remains the supported Colab prep path today.
- `python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training` currently checks dependencies and writes a template plan; it does not yet produce real trained-model artifacts by itself.
- If your laptop GPU is unavailable, use Colab for smoke/template validation today and run the real training loop from the local script logic on a GPU machine until the Colab wrapper is upgraded.

Recent command notes:

- `python scripts/train_grpo.py --mode smoke`
- `python scripts/train_grpo.py --mode dry-run`
- `python scripts/train_grpo_local.py --sanity-check --output-dir artifacts/training/local_sanity`
- `python scripts/train_grpo_local.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --eval-tasks easy,medium,demo,hard --output-dir artifacts/training`
- `python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training`
- `python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training`

Remember: only the local GPU script is intended to produce `training_executed: true` in tracked training artifacts.

What the current repo state proves:

- local smoke execution works
- dry-run artifact generation works
- the local GPU training entrypoint now exists and can self-report dependency/GPU readiness
- a real trained-model claim still requires an actual GPU run and the resulting evaluation artifacts

What still requires external compute:

- any real GRPO/TRL/Unsloth optimization pass
- before/after trained-model evaluation evidence
- training plots generated from a completed run

Use the new local script if you want the shortest path from codebase to an actual run on your own machine.

Recommended models:

- `Qwen/Qwen2.5-0.5B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `google/gemma-2-2b-it`

The default curriculum remains:

- `easy -> medium -> demo`, with `hard` reserved for evaluation unless you explicitly train on it.

The local training loop keeps the existing prompt/action contract:

- prompt source: DroneZ observation summary plus supported action list
- action format: strict JSON with keys `action` and `params`
- reward source: the real reward returned by `DroneZEnvironment.step(...)`

Keep all judge-facing claims aligned with those facts.

If you publish results later, prefer summarizing the actual `eval_before` / `eval_after` deltas rather than describing the training method in abstract.

The repo now has a clear split between scaffolding and real local-GPU execution, but real training evidence still depends on an actual GPU run. Use the local script for your laptop/workstation GPU and keep judge-facing claims tied to the generated `eval_before` and `eval_after` files.

## 12. Visual Demo

The replay UI is trace-driven and intentionally lightweight.

- UI entry: `demo_ui/index.html`
- Data source: `artifacts/traces/*.json`
- Suggested local run:

```bash
python -m http.server 8080
```

Then open:

- `http://localhost:8080/demo_ui/index.html`
- HF Space after deployment: `https://krishna2521-dronez-openenv.hf.space/demo/index.html`

The UI replays real environment traces and shows:

- city sectors and hazards
- charging stations
- active drone locations and paths
- pending and urgent orders
- reward and status panels
- recent event log

Suggested stage flow:

1. load `random` or `naive`
2. show poor reward and unsafe behavior
3. switch to `heuristic`
4. switch to `improved`
5. point at the reward breakdown and policy snapshot panels
6. finish on the reward comparison plot

## 13. Local Setup

```bash
pip install -e .
pytest -q
python inference.py --task demo --policy heuristic --max-steps 8 --summary-only
python inference.py --task demo --policy improved --summary-only
python scripts/evaluate_policies.py
python scripts/generate_demo_trace.py --task demo --policy all
python scripts/generate_plots.py
python scripts/train_grpo.py --mode smoke
python scripts/train_grpo.py --mode dry-run
```

## 14. Server / Docker / HF Deployment

Run locally:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

State endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `POST /step`
- `GET /state`
- `POST /sessions`
- `POST /sessions/{session_id}/reset`
- `GET /sessions/{session_id}/state`
- `POST /sessions/{session_id}/step`

Docker:

```bash
docker build -t dronez .
docker run --rm -p 8000:7860 dronez
```

See `DOCKER_TESTING.md` for curl checks. The root Dockerfile listens on `7860` by default because Hugging Face Docker Spaces require that port.

OpenEnv / HF Space:

- manifest: `openenv.yaml`
- app entry: `server.app:app`
- canonical HF Dockerfile: `Dockerfile`
- mirrored OpenEnv-layout Dockerfile: `server/Dockerfile`
- OpenEnv validation command: `openenv validate`
- current validation result: `PASS` (`Ready for multi-mode deployment`)
- deployment guide: `HF_SPACE_DEPLOYMENT.md`
- Colab training guide: `COLAB_TRAINING.md`
- final submission checklist: `SUBMISSION_CHECKLIST.md`
- Docker testing guide: `DOCKER_TESTING.md`
- blog/writeup draft: `BLOG.md`
- stage script: `FINAL_STAGE_SCRIPT.md`
- official OpenEnv docs: [OpenEnv](https://meta-pytorch.org/OpenEnv/index.html)
- install docs: [Installation](https://meta-pytorch.org/OpenEnv/installation.html)
- official repo: [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)

## 15. Submission Links To Fill Before Deadline

- Team GitHub repo: `https://github.com/Hollenite/DroneZ.git`
- HF Space repo: `https://huggingface.co/spaces/Krishna2521/dronez-openenv`
- HF Space runtime: `https://krishna2521-dronez-openenv.hf.space`
- Root app / judge demo: `https://krishna2521-dronez-openenv.hf.space`
- API docs: `https://krishna2521-dronez-openenv.hf.space/docs`
- Health: `https://krishna2521-dronez-openenv.hf.space/health`
- GitHub repo: `https://github.com/Hollenite/DroneZ`
- Colab notebook: `https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV`
- Blog / writeup: `BLOG.md` locally, replace with public Hugging Face/blog URL before final submission
- Video / YouTube link: `FILL_ME`
- Slides / presentation link: `FILL_ME`

The root runtime URL opens the judge-facing DroneZ replay demo. API metadata is available at `/api` and `/runtime`, while the OpenAPI docs remain available at `/docs`.

## 15.1 What Is Already Proven

- The environment runs locally.
- `pytest -q` passes.
- `openenv validate` passes.
- Evaluation artifacts regenerate.
- The trace-driven demo UI replays real environment logs.
- Deterministic `improved` policy beats random, naive, and heuristic baselines on reward/normalized score.

## 15.2 What Requires External Compute

- Real GRPO / TRL / Unsloth training has not been run in this repo.
- Run the Colab notebook or `scripts/train_grpo_colab.py` on GPU compute.
- Add trained-model evidence only after real `eval_before`, `eval_after`, and training plots exist.


## 15.3 How To Run Real Training With HF / Colab Credits

Use the available Hugging Face or Colab GPU credits only after confirming the environment and demo are stable. Do not commit large checkpoints. The recommended command is:

```bash
python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

Expected real-training outputs after an actual run:

- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`
- `artifacts/training/training_metrics.json`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/training_loss_curve.png` if loss is available

Current honesty boundary: smoke and dry-run training are validated, but real GRPO / TRL / Unsloth training is `NOT RUN` unless these artifacts are regenerated by an actual GPU job.

## 16. Drone Customization

Internal archetypes remain stable for training, while richer deployment profiles live in:

- `configs/fleet_profiles.yaml`

Deployment profiles include:

- `light_food_delivery_drone`
- `medical_priority_drone`
- `heavy_package_drone`
- `long_range_rural_drone`
- `urban_fast_drone`
- `relay_support_drone`

Each profile includes speed, payload, battery, charging, range, energy use, failure modifier, compliance, and cost fields so companies can map the simulator to their own real fleets.

## 17. Known Limitations

- Real GRPO training is scaffolded and Colab-ready, but not yet executed in this repo
- The replay UI is static HTML/JS, not yet a full React build
- Deployment profile selection is represented in config but not yet fully wired into task-specific fleet composition
- The environment is intentionally mission-level, not physics-level

## 18. Future Work

- run real GRPO / Unsloth training on hackathon compute
- add per-task replay traces for all scenarios
- wire deployment profile selection directly into environment reset
- publish HF Space and remote OpenEnv client package
- add richer browser animation and side-by-side baseline vs improved replay
