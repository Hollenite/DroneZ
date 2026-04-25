# DroneZ: Training LLMs to Operate Drone Delivery Fleets with OpenEnv

## Problem

Drone delivery is not just a pathfinding problem. Real delivery operations involve urgent orders, shifting no-fly zones, bad weather, failed drops, charging congestion, battery risk, and heterogeneous drone fleets.

DroneZ turns that operational layer into an OpenEnv-style reinforcement learning environment.

Product-wise, DroneZ is meant to feel like an enterprise simulation platform: organizations can adapt drone profiles, payload assumptions, weather tolerance, charging constraints, and safety objectives before training a mission-level controller.

## Why Fleet Control Is Hard

A mission controller must make tradeoffs over time. Sending the fastest drone may drain battery. Completing one normal order may delay a medical order. Flying through a risky sector may complete delivery faster, but creates regulatory and safety cost. These are exactly the kinds of sequential decisions that static prompt-response evaluation does not capture well.

## Why This Is An OpenEnv / RL Environment

DroneZ exposes the core environment loop:

1. `reset` starts a fresh scenario.
2. The agent observes fleet, order, sector, charger, and disruption state.
3. The agent emits one structured JSON action.
4. `step` executes the action and advances the simulator.
5. The environment returns reward, done status, and the next observation.
6. `state` exposes the current environment state for runtime clients.

The agent is not doing low-level flight control. It is acting as a mission-level fleet operations controller.

That distinction is the core product story. Modern drones are hybrid systems: PID and flight-control logic keep the aircraft stable, sensor fusion and Kalman-style estimation clean up noisy state, GPS navigation handles waypoint movement, and rule-based safety handles emergencies. DroneZ sits above that stack. It is the AI/control-tower layer for fleet assignment, route adaptation, charging decisions, recovery, and mission optimization.

## Action Space

DroneZ supports actions such as:

- `assign_delivery`
- `reroute`
- `return_to_charge`
- `reserve_charger`
- `prioritize_order`
- `attempt_delivery`
- `fallback_to_locker`
- `hold_fleet`
- `resume_operations`

## Reward Design

Positive reward components include successful delivery, urgent delivery success, deadline completion, safe rerouting, recovery from disruption, battery-safe operation, fleet utilization, regulatory compliance, and successful locker fallback.

Negative reward components include invalid actions, missed deadlines, failed delivery attempts, critical battery events, unsafe zone entry, unnecessary reroutes, abandoned urgent orders, idle fleet behavior, charging misuse, and loop/no-progress behavior.

## Anti-Reward-Hacking Safeguards

DroneZ uses layered safeguards:

- invalid actions are penalized and capped
- episode actions are capped independently from simulator horizon
- safety reroutes are rewarded separately and not double-penalized
- deadline misses are counted once when a deadline is crossed
- reward breakdowns are exported for audit
- deterministic demo traces allow manual inspection

## Evaluation Results

Current deterministic evaluation ranks:

`improved > heuristic > random > naive`

Demo scenario:

| Policy | Reward | Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | 89.0 | 0.5861 | 2 | 1 | 0 | 0 |
| heuristic | 32.0 | 0.2889 | 2 | 1 | 8 | 0 |
| random | -162.5 | 0.0100 | 2 | 1 | 33 | 0 |
| naive | -607.5 | 0.0100 | 0 | 0 | 72 | 0 |

The improved policy does not claim to be a trained LLM. It is a deterministic reference controller that demonstrates what better behavior should look like: same delivery count as heuristic in the demo, but with safety violations reduced from 8 to 0.

## Current Honest Training Status

A real local GRPO-style run was attempted on an NVIDIA RTX 5060 Laptop GPU with `Qwen/Qwen2.5-0.5B-Instruct`, but it did not improve. The run exposed the exact bottleneck we now need to solve: the model generated invalid DroneZ actions, every episode ended with `invalid_action_cap_reached`, reward stayed flat, and loss stayed at `0.0`.

Current claim:

`Deterministic improved policy beats baselines. Real training was attempted and revealed an action-format learning bottleneck. The training pipeline now includes action repair, candidate-choice prompts, and SFT warm-start data generation.`

Future claim after real training:

`A trained model improves over its pre-training baseline`, only if real `eval_before`, `eval_after`, and training plots are produced.

## Why The First Training Attempt Failed

The original training prompt asked a small model to emit arbitrary nested JSON from a long operational summary. When every sampled rollout produced invalid actions, all rewards collapsed to the same value. GRPO-style group normalization then had no useful advantage signal, so the loss stayed flat. The fix is to teach format before strategy:

- compact observations instead of giant summaries
- robust JSON extraction and repair
- generated valid candidate actions
- early candidate-choice mode such as `{"choice": 2}`
- SFT data from improved-policy traces before online RL

## Advanced Control Tower UI

The browser replay UI uses real JSON traces from the environment. It now presents the project as a high-tech hybrid-drone control tower instead of a toy grid. It shows a 2.5D procedural city, curved route corridors, drones, orders, chargers, weather overlays, no-fly zones, simulated telemetry, reward evolution, recent events, and control tower state. It is not a disconnected animation.

The extra route geometry, wind values, sensor indicators, and control-layer labels are derived visualization metadata. They help explain the simulated mission, but they are not real GPS or aircraft sensor streams.

Run locally:

```bash
python -m http.server 8080
```

Then open:

`http://localhost:8080/demo_ui/index.html`

On Hugging Face Space, use:

`https://krishna2521-dronez-openenv.hf.space/demo/index.html`

## Deployment

DroneZ is prepared for Docker-based Hugging Face Spaces:

- Space repo: `https://huggingface.co/spaces/Krishna2521/dronez-openenv`
- Runtime URL: `https://krishna2521-dronez-openenv.hf.space`
- API docs: `/docs`
- Health check: `/health`

## Future Work

- run action-format SFT, then candidate-choice GRPO on Colab or hackathon compute
- add trained-model before/after metrics
- wire deployment profiles into scenario-specific fleet composition
- add more generated scenarios and curriculum stages
- expand the replay UI into side-by-side baseline vs improved playback

## Sim-To-Real Boundary

DroneZ does not train real physical drones directly. It trains and evaluates an LLM-style mission controller in a simulated logistics environment. A real aircraft stack would still require low-level flight controllers, PID control, GPS/IMU sensor fusion, Kalman filtering, certified geofencing, aviation-grade safety systems, and extensive sim-to-real validation.
