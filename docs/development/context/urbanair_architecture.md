# UrbanAir Ops — Architecture Specification

## 1. Purpose

This document defines the technical architecture for **DroneZ Ops**, an OpenEnv-compliant environment where an LLM policy acts as a fleet operations controller for autonomous urban delivery drones under disruption.

The goal is not low-level flight control. The goal is **mission-level decision making** over a dynamic, partially observable environment with delayed consequences.

The architecture is designed for:

- fast hackathon execution
- strong OpenEnv compliance
- easy local testing
- compatibility with Hugging Face Spaces deployment
- a clean reward/evaluation pipeline
- minimal but extensible training using HF TRL / Unsloth

---

## 2. Architectural Principles

### 2.1 What this system is

- A **turn-based, stateful environment server**
- A **fleet operations simulator** with structured observations
- An **LLM-action environment** with compact, meaningful actions
- A system designed to demonstrate **world modeling, planning, fleet coordination, and disruption handling**

### 2.2 What this system is not

- Not a full flight physics simulator
- Not a robotics control stack
- Not a vision-based autopilot system
- Not a shortest-path-only logistics problem

### 2.3 Design priorities

1. Keep the environment logic deterministic enough to debug
2. Add enough stochasticity to create meaningful training
3. Ensure every action has downstream effects
4. Make observations understandable for LLMs
5. Keep the action space compact and tractable
6. Keep reward components decomposable and inspectable
7. Keep server structure simple enough for a 2-day hackathon build

---

## 3. High-Level System Overview

```text
+--------------------------------------------------------------+
|                        HF Space / Local App                  |
|                                                              |
|  +----------------------+     +---------------------------+  |
|  |  OpenEnv Server      |<--->|  Environment Client      |  |
|  |  (FastAPI/WebSocket) |     |  / inference / trainer   |  |
|  +----------+-----------+     +------------+--------------+  |
|             |                                  |             |
|             v                                  v             |
|  +----------------------+         +------------------------+ |
|  | DroneZ Environment |         | LLM Policy / Baseline  | |
|  | Episode Orchestrator |         | Agent                  | |
|  +----------+-----------+         +------------------------+ |
|             |                                                |
|             v                                                |
|  +----------------------+                                    |
|  | Simulation Engine    |                                    |
|  | - city state         |                                    |
|  | - fleet state        |                                    |
|  | - order pipeline     |                                    |
|  | - disruptions        |                                    |
|  | - reward scoring     |                                    |
|  +----------+-----------+                                    |
|             |                                                |
|             v                                                |
|  +----------------------+                                    |
|  | Task / Scenario      |                                    |
|  | Configurations       |                                    |
|  +----------------------+                                    |
+--------------------------------------------------------------+
```

---

## 4. Suggested Repository Structure

```text
urbanair-ops/
├── inference.py
├── openenv.yaml
├── README.md
├── requirements.txt
├── Dockerfile
├── .env.example
├── configs/
│   ├── tasks/
│   │   ├── easy.yaml
│   │   ├── medium.yaml
│   │   └── hard.yaml
│   ├── reward_weights.yaml
│   └── fleet_profiles.yaml
├── src/
│   └── urbanair/
│       ├── __init__.py
│       ├── models.py
│       ├── prompts.py
│       ├── enums.py
│       ├── server/
│       │   ├── app.py
│       │   └── env_factory.py
│       ├── env/
│       │   ├── environment.py
│       │   ├── observation_builder.py
│       │   ├── reward_engine.py
│       │   ├── task_manager.py
│       │   ├── action_router.py
│       │   └── termination.py
│       ├── sim/
│       │   ├── engine.py
│       │   ├── city.py
│       │   ├── fleet.py
│       │   ├── drone.py
│       │   ├── orders.py
│       │   ├── disruptions.py
│       │   ├── charging.py
│       │   ├── delivery_logic.py
│       │   └── hidden_dynamics.py
│       ├── policies/
│       │   ├── baseline.py
│       │   ├── heuristic.py
│       │   └── scripted_demo.py
│       ├── eval/
│       │   ├── metrics.py
│       │   ├── benchmark.py
│       │   └── report.py
│       └── utils/
│           ├── logging.py
│           ├── seeding.py
│           └── formatting.py
├── scripts/
│   ├── run_local_server.py
│   ├── run_manual_episode.py
│   ├── train_trl.py
│   ├── evaluate_baseline.py
│   └── generate_demo_trace.py
└── notebooks/
    └── urbanair_training_demo.ipynb
```

---

## 5. Core Components

## 5.1 `models.py`

Defines all strongly typed schemas used across server, simulation, inference, and training.

### Main model groups

- `DroneState`
- `OrderState`
- `SectorState`
- `ChargingStationState`
- `PolicyNotice`
- `EmergencyEvent`
- `EnvironmentObservation`
- `FleetAction`
- `StepResult`
- `EpisodeSummary`
- `TaskConfig`
- `RewardBreakdown`

### Important rule

All external environment I/O should pass through explicit typed models. Do not let ad hoc dictionaries spread throughout the codebase.

---

## 5.2 `sim/engine.py`

The authoritative world-state transition engine.

### Responsibilities

- own the episode clock
- apply actions
- advance drone movement / status
- progress orders and deadlines
- trigger disruptions
- update hidden state
- compute raw environment events before reward shaping

### Simulation step sequence

Each environment step should follow this order:

1. Validate incoming action
2. Apply operational decision
3. Advance time by one tick
4. Update drone positions / statuses
5. Resolve charging events
6. Resolve delivery attempts and outcomes
7. Update customer availability windows
8. Trigger or evolve disruptions
9. Update hidden dynamics
10. Compute terminal conditions
11. Produce reward breakdown
12. Build next observation

This sequence should remain stable across all difficulty levels.

---

## 5.3 `env/environment.py`

The OpenEnv-compatible environment wrapper.

### Responsibilities

- expose reset/step/task-selection interface
- map task IDs to scenario configs
- connect OpenEnv actions to simulator calls
- return compliant observation/reward/done/info structures

### Environment responsibilities vs simulation responsibilities

- **Environment layer**: interface, validation, formatting, task lifecycle
- **Simulation layer**: domain rules and world transitions

Do not mix these.

---

## 5.4 `env/action_router.py`

Maps action names and parameters to simulator operations.

### Supported action set

- `assign_delivery(drone_id, order_id)`
- `reroute(drone_id, corridor)`
- `return_to_charge(drone_id, station_id)`
- `reserve_charger(drone_id, station_id)`
- `delay_order(order_id)`
- `prioritize_order(order_id)`
- `swap_assignments(drone_a, drone_b)`
- `attempt_delivery(drone_id, mode)`
- `fallback_to_locker(order_id, locker_id)`
- `hold_fleet(zone_id)`
- `resume_operations(zone_id)`

### Validation rules

Each action must validate:

- object IDs exist
- action is legal in current state
- actor/resource constraints are satisfied
- action is not contradictory to current policy lock
- drone is not already unavailable / offline / charging / critical

Every invalid action should return a penalty and explicit error context.

---

## 5.5 `env/observation_builder.py`

Builds LLM-friendly observations.

### Observation design goals

- structured enough for reliable parsing
- readable enough for prompting
- compact enough to fit longer trajectories
- include uncertainty indicators
- include previous action outcome and reward summary

### Recommended observation sections

1. episode metadata
2. fleet summary
3. top pending orders
4. city/disruption summary
5. charging network state
6. recent events
7. warnings and risk indicators
8. allowed actions reminder

### Example observation style

```text
TIME_STEP: 12 / 50
TASK: medium
TOTAL_PENDING_ORDERS: 7
ACTIVE_NO_FLY_ZONES: Z3, Z7
WEATHER_ALERTS: Z2=moderate_wind, Z5=heavy_rain

FLEET:
- D1 | type=fast_light | battery=31 | zone=Z2 | assigned=O14 | eta=2 | risk=medium
- D2 | type=heavy_carrier | battery=79 | zone=hub | assigned=None | eta=None | risk=low
- D3 | type=long_range_sensitive | battery=56 | zone=Z6 | assigned=O11 | eta=1 | risk=weather_sensitive
- D4 | type=relay | battery=68 | zone=Z4 | assigned=network_support | signal=strong

ORDERS:
- O14 | priority=urgent | deadline=3 | recipient=available | retry=0 | late_penalty=high
- O11 | priority=normal | deadline=1 | recipient=uncertain | retry=1 | late_penalty=medium
- O19 | priority=medical | deadline=4 | recipient=available | retry=0 | late_penalty=critical

CHARGING:
- C1 | occupancy=2/2 | queue=1
- C2 | occupancy=1/3 | queue=0

RECENT EVENTS:
- Z3 restricted due to emergency response
- O11 previous drop attempt failed due to recipient mismatch
- D1 battery is approaching critical reserve threshold
```

---

## 5.6 `env/reward_engine.py`

Computes decomposed reward signals.

### Reward philosophy

Avoid a single opaque scalar. Compute component rewards, then aggregate.

### Recommended reward components

#### Positive

- on-time successful deliveries
- urgent delivery completion
- fleet utilization without overload
- energy efficiency
- balanced charging utilization
- disruption recovery quality
- low reattempt count
- regulatory compliance

#### Negative

- missed delivery windows
- battery-critical flight state
- restricted-zone entry
- unnecessary reroutes
- failed delivery attempt
- idle drone while backlog grows
- congestion due to poor allocation
- overloading one drone while others idle
- abandoning urgent orders

### Additional shaping ideas

- reward preserving optionality under uncertainty
- reward returning early before hard battery failure
- mild penalty for action thrashing
- mild penalty for indecision (too many delay/hold actions without progress)

### Output format

`RewardBreakdown` should include:

- total reward
- each component value
- textual explanation
- episode cumulative totals

This makes debugging, demos, and judge explanation much easier.

---

## 5.7 `sim/drone.py`

Encapsulates heterogeneous drone behavior.

### Fleet profiles

#### `fast_light`

- high speed
- low battery
- low payload
- good for urgent short-haul deliveries

#### `heavy_carrier`

- lower speed
- high payload
- higher energy efficiency under load
- good for bulky deliveries

#### `long_range_sensitive`

- long range
- good battery reserve
- strong delivery coverage
- weather-sensitive, especially wind/rain

#### `relay`

- no package delivery
- used to improve communication strength in weak sectors
- increases effective reliability of nearby drones

### Minimum drone state fields

- id
- type
- battery
- payload_capacity
- speed
- current_zone
- assigned_order_id
- status
- eta
- risk_status
- communication_strength
- maintenance_health

---

## 5.8 `sim/orders.py`

Manages package jobs and lifecycle transitions.

### Order lifecycle

- created
- queued
- assigned
- in_transit
- delivery_attempted
- delivered
- failed
- locker_fallback
- escalated_to_human
- deferred
- canceled

### Important order fields

- order_id
- priority
- deadline_window
- drop_mode
- recipient_availability
- retry_count
- late_penalty
- zone
- package_weight
- fallback_options

---

## 5.9 `sim/disruptions.py`

Generates and evolves dynamic disruptions.

### Core disruption classes

- weather drift
- temporary no-fly zones
- charging congestion spikes
- customer no-show / unavailable
- drop-site obstruction
- emergency shipment insertion
- communication degradation
- policy notice change

### Design rule

Disruptions should be configurable by difficulty level, and partially stochastic, but never so random that good decisions stop mattering.

---

## 5.10 `sim/hidden_dynamics.py`

Contains latent factors the agent must infer indirectly.

### Hidden/partially hidden factors

- future sector weather change tendencies
- high-risk failure zones
- customer no-show probability
- charging queue growth likelihood
- downstream cost of delaying certain orders
- reliability drift after repeated failures in a sector

### Purpose

These hidden variables create real world-modeling pressure without making the environment unfair.

---

## 6. Scenario / Task Design

The hackathon requires at least 3 tasks. Use difficulty tiers with increasing complexity.

## 6.1 Easy

### Characteristics

- 2 delivery drones + 1 relay drone
- 5–7 orders
- few disruptions
- short horizon
- light weather variation
- no-fly zones mostly static

### Intended behavior

Teach basic assignment, charging, and urgent-order prioritization.

## 6.2 Medium

### Characteristics

- 3 delivery drones + 1 relay drone
- 8–12 orders
- dynamic charging congestion
- customer availability windows matter
- occasional failed drop attempts
- temporary policy notices

### Intended behavior

Teach replanning, fallback delivery, and coordination.

## 6.3 Hard

### Characteristics

- 4+ heterogeneous drones including relay
- 12–18 orders
- dynamic disruptions throughout episode
- medical/emergency order insertion
- multiple no-fly zones
- weather drift and communication degradation
- stronger hidden-state pressure

### Intended behavior

Teach long-horizon fleet optimization under uncertainty.

---

## 7. Episode Flow

### 7.1 Reset phase

- select task config
- initialize city sectors
- initialize fleet with drone types
- spawn order set
- set hidden variables
- set episode length / tick limit
- build initial observation

### 7.2 Action phase

At each step, the agent chooses one operational action.

### 7.3 Transition phase

Simulation advances one time tick and resolves consequences.

### 7.4 Termination phase

Episode ends when any of the following hold:

- all orders resolved
- max steps reached
- catastrophic failure threshold crossed
- all viable drones unavailable and unresolved backlog remains

---

## 8. Failure-Recovery Loop

This is a core differentiator and must be explicit in the architecture.

### Failure events can trigger

- retry
- reassignment
- locker delivery fallback
- human escalation
- deferred delivery

### Why it matters

This creates branching operational consequences and makes the environment feel real.

### Recommended implementation

Delivery attempts should return a structured outcome:

- success
- recipient_unavailable
- obstruction
- zone_blocked
- battery_abort
- policy_disallowed
- communication_loss

These outcomes then unlock follow-up action possibilities.

---

## 9. Metrics and Evaluation

## 9.1 Core episode metrics

- delivery success rate
- urgent-order success rate
- average lateness
- total energy consumed
- battery safety incidents
- restricted-zone violations
- failed delivery attempts
- average drone idle ratio
- charging balance score
- disruption recovery score
- total reward

## 9.2 Benchmark comparison modes

Implement at least these:

- random / naive baseline
- heuristic baseline
- LLM policy before training
- LLM policy after training

## 9.3 Demo-friendly reporting

At episode end, print:

- summary metrics
- reward breakdown totals
- notable events timeline
- key decisions made by the agent

---

## 10. Baselines

## 10.1 Naive baseline

Greedy nearest-delivery assignment.

Weaknesses:

- ignores future battery constraints
- poor charger planning
- reacts badly to disruptions
- overloads fastest drone

## 10.2 Heuristic baseline

Rule-based operational manager:

- prioritize urgent orders first
- reserve charge when below threshold
- avoid current no-fly zones
- use heavy drone for heavy orders
- use locker fallback after repeated delivery failure

This baseline is important because it gives you a meaningful comparison before RL fine-tuning.

---

## 11. OpenEnv Integration Notes

## 11.1 Environment interface requirements

The environment must expose a clean OpenEnv-compatible interface for:

- task creation / task selection
- reset
- step
- observation emission
- reward emission
- done flag
- info / metadata

## 11.2 `openenv.yaml`

Define:

- environment name
- tasks (`easy`, `medium`, `hard`)
- action contract
- observation contract
- evaluation expectations

## 11.3 `inference.py`

Should:

- read required environment variables
- connect to the environment
- run episodes for selected tasks
- print required step trace format expected by the hackathon

Do not tightly couple `inference.py` to internal training code.

---

## 12. Training Architecture

## 12.1 Training goal

Train an LLM policy to improve mission-level operational decisions over repeated episodes.

## 12.2 Minimal pipeline

1. start OpenEnv server
2. connect TRL environment factory
3. use a simple baseline prompt / policy format
4. train on `easy`
5. validate on `medium`
6. optionally fine-tune on mixed difficulty
7. report reward improvement and metric deltas

## 12.3 Training data strategy

The environment itself generates trajectories. You do not need a massive offline dataset to begin.

Useful bootstrap options:

- scripted heuristic rollouts
- synthetic demonstrations for a few cases
- preference ranking between good and bad action traces

## 12.4 Logging

Log:

- episode return
- reward component breakdowns
- invalid action count
- urgent order completion rate
- battery incidents
- no-fly violations

---

## 13. Deployment Architecture

## 13.1 Local dev mode

- run FastAPI/OpenEnv server
- run manual client
- run scripted demo traces

## 13.2 Hugging Face Spaces mode

- containerized deployment
- one environment server process
- optional simple UI panel or log view
- lightweight enough to avoid resource issues

## 13.3 Demo mode

Add a deterministic seeded scenario specifically for the final pitch.

This demo task should reliably show:

- baseline mistake
- disruption event
- trained-policy recovery
- stronger final metrics

---

## 14. Implementation Priorities for Hackathon Timeline

## Phase 1 — Must Have

- core OpenEnv environment works
- easy/medium/hard tasks exist
- heterogeneous fleet supported
- reward breakdown works
- failure-recovery loop works
- heuristic baseline works
- inference script prints compliant logs

## Phase 2 — Strong Upgrade

- relay drone affects communication reliability
- dynamic no-fly zones
- temporary policy notices
- locker fallback logic
- richer metrics dashboard

## Phase 3 — Nice to Have

- scenario generator for harder evaluation cases
- self-improvement curriculum extension
- richer visual demo layer

---

## 15. Suggested Class Skeletons

## `TaskConfig`

```python
class TaskConfig:
    task_id: str
    max_steps: int
    num_orders: int
    fleet_profile: list[str]
    disruption_profile: dict
    weather_profile: dict
    charging_profile: dict
    reward_weights: dict
    seed: int | None
```

## `EnvironmentObservation`

```python
class EnvironmentObservation:
    step: int
    max_steps: int
    fleet: list[DroneState]
    orders: list[OrderState]
    sectors: list[SectorState]
    charging: list[ChargingStationState]
    active_notices: list[PolicyNotice]
    active_events: list[EmergencyEvent]
    recent_events: list[str]
    valid_actions: list[str]
```

## `RewardBreakdown`

```python
class RewardBreakdown:
    total: float
    on_time_delivery: float
    urgent_completion: float
    fleet_utilization: float
    energy_efficiency: float
    charging_balance: float
    disruption_recovery: float
    compliance: float
    lateness_penalty: float
    battery_penalty: float
    violation_penalty: float
    failed_attempt_penalty: float
    idle_penalty: float
    overload_penalty: float
    notes: list[str]
```

---

## 16. Common Failure Modes to Defend Against

- action space too large and noisy
- observation too verbose for the model to act well
- randomness overwhelming skill signal
- reward components conflicting too hard
- no meaningful baseline to compare against
- demo scenario not deterministic enough
- hidden state being effectively impossible to infer
- too many environment mechanics for a 2-day build

---

## 17. Final Technical Positioning

DroneZ should be built and explained as:

> A stateful, partially observable, OpenEnv-compliant urban drone operations environment where an LLM learns mission-level fleet management under disruption, rather than low-level autonomous flight control.

That is the architectural identity of the project.
