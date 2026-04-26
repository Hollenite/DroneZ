# DroneZ Fleet Delivery Environment

**Showcase folder:** https://drive.google.com/drive/folders/1EvQvEcNg9AasMICUYt_AkMDGtoTlMbJE?usp=sharing

**Hugging Face Space Repository:** https://huggingface.co/spaces/Krishna2521/dronez-openenv

**Live Runtime:** https://krishna2521-dronez-openenv.hf.space

**Live Demo:** https://krishna2521-dronez-openenv.hf.space/demo/index.html

**API Docs:** https://krishna2521-dronez-openenv.hf.space/docs

**Health Check:** https://krishna2521-dronez-openenv.hf.space/health

**GitHub Team Repo:** https://github.com/Hollenite/DroneZ.git

---

## Introduction

When we order something online, we usually expect a delivery person to bring the parcel to our doorstep. Sometimes the parcel arrives early, sometimes late, and one of the biggest reasons is road traffic.

Drones promise a faster future. They can move through the air instead of roads, and their routes can be more direct. At first, it feels obvious that drone delivery should always be quicker than human delivery.

But there is an important problem: many drone systems follow predictable, fixed routes. Fixed routes are safe and easy to understand, but they are not always the best, quickest, or most efficient. A drone can still be delayed by weather, no-fly zones, obstacles, battery limits, unavailable recipients, or sudden urgent orders.

That is where DroneZ becomes useful.

DroneZ is an OpenEnv reinforcement-learning environment for mission-level drone fleet control. It helps train and evaluate decision-making agents in uncertain delivery situations. The goal is not to control propellers directly. The goal is to make better high-level decisions: which drone should take which order, when to reroute, when to charge, when to prioritize urgent deliveries, and how to recover from disruptions.

In DroneZ, the agent makes one decision at a time. Every action is evaluated with a reward. Good decisions, such as safe delivery, urgent delivery success, safe rerouting, and battery-aware behavior, receive positive rewards. Bad decisions, such as invalid actions, unsafe zones, missed deadlines, failed deliveries, or wasteful loops, receive penalties.

Over time, a policy can learn which decisions are more likely to produce safe, efficient, and reliable delivery operations.

---

## What DroneZ Actually Simulates

DroneZ simulates the mission-control layer of a drone fleet.

It includes:

- multiple drones
- delivery orders
- urgent orders
- weather disruptions
- no-fly zones
- charging stations
- battery constraints
- delivery deadlines
- recipient availability
- fallback delivery options
- safety penalties and reward shaping

DroneZ is not a full physics simulator like Isaac Sim, Gazebo, or AirSim. It does not simulate real propellers, aerodynamics, or certified aviation behavior. Instead, it focuses on the decision-making layer that sits above the drone hardware.

This is realistic because modern drone systems are hybrid systems:

- PID and flight-control logic keep drones stable.
- GPS, IMU, and sensor fusion help drones understand their state.
- Safety rules handle emergencies.
- AI/RL can help with high-level decisions like routing, fleet assignment, charging, and recovery.

DroneZ focuses on that AI/control-tower layer.

---

## Interactive Demonstration Interface

The browser replay interface visualizes DroneZ using real JSON traces from the environment. Instead of showing a simple grid, the demo presents the system as a hybrid drone control tower.

The interface shows:

- a 2.5D procedural city
- curved route corridors
- active drones
- delivery locations
- charging stations
- weather overlays
- no-fly zones
- simulated telemetry
- reward evolution
- recent event feed
- control tower state monitoring

This makes the environment easier to understand during a hackathon demo because judges can see the mission flow, not just read raw JSON.

**Important note:** route geometry, wind values, sensor indicators, and control-layer labels are derived visualization metadata. They are not real-world GPS or aircraft sensor streams.

---

## Reward System

DroneZ uses a structured reward system to encourage intelligent fleet behavior.

### Positive Rewards

The environment rewards:

- successful delivery
- urgent delivery success
- meeting deadlines
- safe rerouting
- recovering from disruptions
- battery-safe operation
- efficient fleet utilization
- regulatory compliance
- successful locker fallback

### Negative Penalties

The environment penalizes:

- invalid actions
- missed deadlines
- failed delivery attempts
- critical battery events
- unsafe zone entry
- unnecessary reroutes
- abandoned urgent orders
- idle fleet behavior
- charging misuse
- loop or no-progress behavior

This reward design encourages the controller to balance speed, safety, battery health, urgency, and reliability.

---

## Evaluation Results

The current proven improvement is from a deterministic improved policy, not from a trained GRPO model.

### Demo Scenario

| Policy | Reward | Normalized Score | Deliveries | Urgent Successes | Safety Violations | Invalid Actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| improved | **89.0** | **0.5861** | 2 | 1 | **0** | 0 |
| heuristic | 32.0 | 0.2889 | 2 | 1 | 8 | 0 |
| random | -162.5 | 0.0100 | 2 | 1 | 33 | 0 |
| naive | -607.5 | 0.0100 | 0 | 0 | 72 | 0 |

The improved policy keeps the same delivery count as the heuristic policy in the demo, but it reduces safety violations from `8` to `0` and improves reward from `32.0` to `89.0`.

---

## Training Status

A real GRPO-style training attempt was run earlier with `Qwen/Qwen2.5-0.5B-Instruct`, but it did not improve reward. The model generated invalid DroneZ actions, so episodes ended with invalid-action failure and the reward stayed flat.

That result is documented honestly. DroneZ does not claim trained-model improvement yet.

To address the issue, the project now includes:

- compact action prompts
- robust JSON parsing
- candidate-choice mode
- action repair
- format-check diagnostics
- SFT warm-start action data

The current action-format diagnostics show:

- `valid_json_rate`: `0.875`
- `valid_action_rate`: `0.875`
- SFT examples: `108`

This is progress in action-format reliability, but not yet proof of reward-improving GRPO training.

---

## Running the Environment Locally

Install the package:

```bash
python -m pip install -e .
```

Run the API server:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
http://127.0.0.1:8000/demo/index.html
http://127.0.0.1:8000/docs
```

---

## Running on Hugging Face

Open the live demo:

```text
https://krishna2521-dronez-openenv.hf.space/demo/index.html
```

Open the API docs:

```text
https://krishna2521-dronez-openenv.hf.space/docs
```

---

## Conclusion

Drone delivery is not just about flying. It is about decision-making under uncertainty.

DroneZ shifts the focus from fixed drone routes to adaptive fleet intelligence. Every action can be observed, evaluated, rewarded, penalized, and improved. This makes DroneZ a useful environment for exploring safer and more reliable mission-level drone control.
