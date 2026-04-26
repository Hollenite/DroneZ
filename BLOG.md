# DroneZ: Training Mission-Level Drone Fleet Controllers with OpenEnv

DroneZ is an OpenEnv reinforcement-learning environment for drone delivery operations. It is designed around a practical idea: delivery drones should not only follow fixed routes. A fleet controller should be able to make adaptive decisions when weather changes, batteries run low, urgent orders arrive, no-fly zones appear, or delivery attempts fail.

## The Problem

Drone delivery sounds simple from far away: pick up a package, fly to a destination, drop it off. Real operations are harder.

A fleet manager must answer questions like:

- Which drone should take the urgent order?
- Should a drone reroute around a storm or restricted area?
- When should a drone stop and charge?
- What happens if the customer is unavailable?
- Is a faster route worth the safety risk?
- Should the control tower pause operations in a dangerous sector?

These are long-horizon decisions. One good or bad action can affect the rest of the episode.

## Why DroneZ Uses OpenEnv

OpenEnv standardizes the environment loop:

1. `reset` starts a scenario.
2. The agent observes the fleet, orders, weather, charging stations, and warnings.
3. The agent chooses one structured action.
4. `step` executes the action and returns reward plus the next observation.
5. Repeated interaction lets policies be evaluated and trained.

DroneZ uses that loop to turn drone fleet operations into a measurable decision-making problem.

## Hybrid Drone Architecture

DroneZ is not training propeller control. That is a very important boundary.

Modern drone systems are hybrid:

- Low-level control handles PID stability, GPS navigation, sensor fusion, Kalman-style estimation, and emergency safety.
- High-level AI/RL handles assignment, route adaptation, charging, recovery, priority, and mission optimization.
- A control tower or parent server can coordinate the whole fleet.

DroneZ focuses on the high-level mission-control layer.

## Environment Design

Each DroneZ scenario includes:

- multiple drones with battery, position, route, assignment, and risk
- orders with priority, deadline, destination, and fallback options
- city sectors with weather, congestion, restrictions, and no-fly risk
- charging stations with limited capacity
- disruptions such as storms, failed delivery attempts, and urgent order changes

The action space includes:

- `assign_delivery`
- `attempt_delivery`
- `reroute`
- `prioritize_order`
- `fallback_to_locker`
- `return_to_charge`
- `reserve_charger`
- `hold_fleet`
- `resume_operations`

## Reward Design

DroneZ rewards useful and safe decisions:

- successful delivery
- urgent delivery success
- meeting deadlines
- safe rerouting
- disruption recovery
- battery-safe operation
- efficient fleet utilization
- regulatory compliance
- successful locker fallback

It penalizes unsafe or wasteful behavior:

- invalid actions
- missed deadlines
- failed delivery attempts
- critical battery events
- unsafe zone entry
- unnecessary reroutes
- abandoned urgent orders
- idle fleet behavior
- charging misuse
- loop/no-progress behavior

This makes the environment more than a simple pathfinding toy. The agent must balance speed, safety, battery, priority, and reliability.

## Evaluation Results

The current proven result is deterministic policy improvement, not trained-model improvement.

Aggregate evaluation across `easy`, `medium`, `hard`, and `demo`:

| Policy | Mean Reward | Mean Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | **-42.125** | **0.3051** | 5 | 3 | **0** | 0 |
| heuristic | -281.500 | 0.0797 | 10 | 5 | 182 | 0 |
| random | -921.500 | 0.0100 | 9 | 4 | 396 | 0 |
| naive | -1038.000 | 0.0100 | 6 | 3 | 497 | 0 |

Demo scenario:

| Policy | Reward | Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | **89.0** | **0.5861** | 2 | 1 | **0** | 0 |
| heuristic | 32.0 | 0.2889 | 2 | 1 | 8 | 0 |
| random | -162.5 | 0.0100 | 2 | 1 | 33 | 0 |
| naive | -607.5 | 0.0100 | 0 | 0 | 72 | 0 |

The improved controller is deterministic and hand-built. It demonstrates that the environment can distinguish safer and more useful behavior. It should not be described as a trained LLM result.

## Training Status

A real local GRPO-style run was attempted on an NVIDIA RTX 5060 Laptop GPU with `Qwen/Qwen2.5-0.5B-Instruct`. It did not improve reward. The model generated invalid DroneZ actions, episodes hit `invalid_action_cap_reached`, reward stayed flat, loss stayed `0.0`, and `eval_after` did not improve over `eval_before`.

That failure was useful because it exposed the real bottleneck: action formatting.

The project now includes:

- compact action prompts
- candidate-choice mode
- robust JSON parsing
- safe action repair
- format-check diagnostics
- SFT warm-start examples generated from the improved policy

Current format-check artifacts report:

- `valid_json_rate`: `0.875`
- `valid_action_rate`: `0.875`

This means the action interface is improving, but it is still not a reward-improving trained model claim.

## Demo and Links

- Hugging Face Space: https://huggingface.co/spaces/Krishna2521/dronez-openenv
- Live Runtime: https://krishna2521-dronez-openenv.hf.space
- Live Demo: https://krishna2521-dronez-openenv.hf.space/demo/index.html
- API Docs: https://krishna2521-dronez-openenv.hf.space/docs
- GitHub Team Repo: https://github.com/Hollenite/DroneZ.git
- Colab Notebook: `notebooks/train_dronez_grpo_colab.ipynb`
- Public Colab Link: https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV
- YouTube Demo: `TODO - add before final form submission`
- Slides: `TODO - add before final form submission`

## Visual Demo

The browser demo uses real DroneZ trace files. It adds a professional control-tower visualization with:

- a 2.5D procedural city
- route corridors
- drone telemetry panels
- weather and no-fly overlays
- charging stations
- order markers
- reward evolution
- event timelines
- hybrid control architecture panels

Some visual fields are derived for explanation, such as route geometry, wind values, and sensor status labels. These are simulated visualization metadata, not real aircraft telemetry.

## Limitations

DroneZ does not yet provide:

- full Isaac Sim, Gazebo, or AirSim physics
- real motor or propeller control
- certified aviation behavior
- real camera/LiDAR/GPS streams
- proven GRPO reward improvement

Future work includes action-format SFT, candidate-choice GRPO, richer scenario generation, sim-to-real validation, and optional integration with a real robotics simulator.

## Final Takeaway

DroneZ shows drone delivery as a decision-making problem under uncertainty. The strongest current result is a working OpenEnv environment, a deployed Hugging Face Space, a control-tower demo, deterministic policy improvements, and an honest training pipeline that exposes and addresses the next research bottleneck.
