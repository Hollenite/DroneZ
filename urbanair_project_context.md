# DroneZ — Project Context for Claude Code

## 1. Project Identity

**Project name:** DroneZ

**Core concept:**
DroneZ is an OpenEnv-compliant RL environment for the Meta PyTorch OpenEnv Hackathon. An LLM agent acts as a **fleet operations controller** for autonomous urban delivery drones. The environment is not about low-level piloting. It is about mission-level logistics, disruption response, fleet coordination, and long-horizon decision making in a dynamic urban world.

This project exists to feel much more serious and interesting than a simple “delivery drone simulator.”

---

## 2. Official Problem Statement

Modern drone logistics systems do not fail because of pathfinding alone. They fail because real-world delivery operations involve shifting constraints, partial observability, competing priorities, regulatory changes, battery limitations, delivery exceptions, and coordination across fleets. We propose an OpenEnv environment where an LLM agent acts as a fleet operations controller for urban delivery drones, learning to allocate tasks, respond to disruptions, and optimize fleet-wide mission outcomes over long trajectories.

---

## 3. Why This Project Exists

The hackathon is not asking for a toy simulator. It is asking for an environment that meaningfully trains and evaluates agent behavior.

DroneZ is designed to satisfy that by creating a stateful world where:

- actions have delayed consequences
- the agent must model hidden risks
- competing objectives must be balanced
- multiple assets must be coordinated
- disruptions force replanning
- reward improvements can be clearly shown in a demo

This is meant to score strongly on:

- environment innovation
- storytelling
- visible reward improvement
- coherent reward/training pipeline

---

## 4. Theme Fit

### Primary theme fit

**Theme #3.1 — World Modeling across professional tasks**

This is the strongest positioning.
The environment simulates a real operational system where the model interacts with a dynamic world, tracks state over time, and must respond to nontrivial changes.

### Secondary theme fit

**Theme #1 — Multi-Agent Interactions**

Even though the controller is one primary policy, it coordinates a heterogeneous fleet of semi-autonomous drones with different capabilities and constraints.

### Optional future extension

**Theme #4 — Self-Improvement**

If time allows, this environment can later support scenario generation or targeted hard-case generation from failure traces. But this is not the initial focus.

---

## 5. What We Are Building

We are building a **turn-based operational decision environment**.

The agent observes a structured summary of:

- drone fleet state
- package/order backlog
- charging infrastructure state
- urban sector conditions
- disruptions and notices
- recent events and consequences

Then at each step it chooses one operational action, such as:

- assigning a drone to a delivery
- rerouting a drone
- returning a drone to charge
- reserving a charger
- prioritizing or delaying an order
- attempting a delivery
- triggering locker fallback
- holding operations in a risky zone

The environment advances by one tick and updates the world.

---

## 6. What We Are Explicitly NOT Building

Claude should avoid drifting into any of the following:

- full 3D physics simulation
- raw drone flight control
- PID/autopilot tuning
- computer vision perception stack
- complex real-world mapping stack
- advanced browser-based visualizations as a first priority

This project should stay focused on **decision intelligence** and **environment quality**.

---

## 7. Success Criteria

A successful submission should achieve the following:

### Environment quality

- the environment feels dynamic and realistic
- actions matter beyond the current step
- the state contains meaningful tradeoffs
- there is uncertainty but not chaos

### Hackathon readiness

- OpenEnv-compliant
- hosted on HF Spaces
- has a minimal training script
- has at least 3 tasks / difficulties
- demoable in under 3 minutes

### Demo quality

- clear baseline vs improved behavior
- visible reward differences
- understandable scenario and metrics
- memorable story

### Engineering quality

- modular codebase
- typed models
- reproducible seeds
- deterministic demo scenario
- easy local testing

---

## 8. Key Product Differentiators

These are core identity features. Claude should preserve them.

### 8.1 Mission-level autonomy, not flight control

The project must always feel like a fleet operations environment.

### 8.2 Heterogeneous fleet

Different drone types force smarter assignment.
Required drone archetypes:

- fast but low battery
- slow but heavy payload
- long-range but weather-sensitive
- relay drone with no delivery payload

### 8.3 Partial observability

The model should not know everything directly.
Hidden or partially hidden elements include:

- future weather changes
- likely failure sectors
- customer no-show probability
- charging queue evolution
- downstream effects of delaying some orders

### 8.4 Failure-recovery loop

Failed deliveries should create branching operational choices:

- retry
- reassignment
- locker fallback
- human escalation
- deferred delivery

### 8.5 Composite reward with visible submetrics

Reward must be explainable and decomposed.

---

## 9. Environment State Requirements

Each observation should include the following categories.

### Fleet state

For each drone:

- drone id
- battery
- payload capacity
- current position/zone
- assigned order
- ETA
- health/risk status
- communication strength

### Order state

For each order:

- package id
- priority level
- deadline/window
- drop mode
- recipient availability
- retry count
- penalty severity if late

### City state

- sector weather
- charging station occupancy
- active no-fly zones
- congestion score
- emergency events
- temporary policy notices

### Hidden/partially hidden factors

These should influence transitions, but may not be fully exposed:

- future weather changes
- likely failure sectors
- customer no-show probability
- charging queue evolution
- downstream effects of delaying some orders

---

## 10. Action Space Requirements

The action space should remain compact but meaningful.
These are the required core actions:

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

Claude may add a small number of closely related support actions if truly necessary, but should avoid action-space bloat.

---

## 11. Reward Philosophy

This project should not use one flat opaque reward.
It should use a **composite reward** with named components and a total score.

### Positive reward components

- on-time successful deliveries
- urgent delivery completion
- high fleet utilization without overload
- efficient energy use
- balanced charging usage
- successful disruption recovery
- low reattempt rate
- safe regulatory compliance

### Negative reward components

- missed delivery window
- battery-critical flight state
- entering restricted zone
- unnecessary reroutes
- failed delivery attempt
- drone idling while orders accumulate
- congestion caused by poor allocation
- too many orders assigned to one drone
- abandoning urgent orders

### Implementation expectation

Each step should compute a breakdown, and episode summaries should aggregate these values.

---

## 12. Difficulty Design

There should be 3 initial tasks.

### Easy

Used for basic training and debugging.

- fewer orders
- fewer disruptions
- shorter horizon
- basic charging and prioritization

### Medium

Used for more realistic operational complexity.

- customer windows
- failed drops
- dynamic charging congestion
- policy notices

### Hard

Used for final evaluation/demo.

- more drones
- more disruptions
- no-fly zone changes
- urgent insertions
- more hidden-state pressure

Claude should structure the code so adding more scenarios later is easy.

---

## 13. Demo Story We Want

This matters a lot. The final demo should be narratively strong.

### Desired demo flow

1. show the city, fleet, and pending orders
2. show a baseline making greedy or short-sighted decisions
3. trigger a disruption: weather shift, urgent medical order, charging congestion, failed drop attempt, or no-fly zone update
4. show the improved agent handling the situation better
5. show metrics and reward breakdown proving the difference

The demo should make judges feel:

- “this is a real agent environment”
- “the trained version is clearly more competent”
- “this is not just a pathfinding toy”

---

## 14. Constraints Claude Must Respect

### Constraint 1

Keep scope hackathon-realistic. Avoid overengineering.

### Constraint 2

Prioritize a working OpenEnv environment over optional UI polish.

### Constraint 3

Build clean baselines early.
A heuristic baseline is required for comparison.

### Constraint 4

Favor modularity and clarity over cleverness.

### Constraint 5

Every mechanic should justify itself in the pitch.
If a feature does not improve the environment story, reward logic, or world-modeling depth, it is lower priority.

### Constraint 6

Maintain a deterministic seeded demo scenario.
This is important for a reliable final presentation.

---

## 15. What Claude Should Optimize For

When making technical decisions, Claude should optimize for:

1. OpenEnv compliance
2. strong environment design
3. reward interpretability
4. demo quality
5. implementation speed
6. modularity for future extension

Not for:

- maximum realism at all costs
- complex visuals before environment quality
- excessive action vocabulary
- unnecessary simulation complexity

---

## 16. Baseline Behavior Expectations

The project needs at least two non-trained baselines:

### Naive baseline

A simple greedy scheduler that looks reasonable but fails under disruption.

### Heuristic baseline

A more informed rule-based manager that:

- respects battery thresholds
- prioritizes urgent orders
- uses the right drone type for the right job
- avoids active no-fly zones
- uses fallback delivery after repeated failures

This is important because the final comparison should not be “trained agent vs nonsense.”

---

## 17. Implementation Priorities

### Highest priority

- environment core loop
- typed state models
- action validation
- reward engine
- task configs
- basic baselines
- compliant inference script

### Medium priority

- relay drone communication logic
- richer failure reasons
- more nuanced policy notices
- benchmark report generator

### Lower priority

- visual front-end
- advanced analytics dashboard
- self-improving scenario generation

---

## 18. Suggested Language for README / Pitch Alignment

Claude should align the codebase and documentation to the following framing:

> DroneZ is a world-modeling environment for training LLM agents to manage a heterogeneous fleet of autonomous delivery drones under real operational disruption. Instead of testing raw flight control, it evaluates mission-level decision making under uncertainty, delayed consequences, and competing objectives.

This phrasing should remain consistent across README, docs, demo script, and code comments.

---

## 19. Developer Guidance for Claude Code

### When implementing

- prefer clean, testable modules
- add explicit models and enums
- keep simulation logic isolated from OpenEnv glue code
- keep reward components named and inspectable
- keep scenario configs editable via YAML or simple config files

### When uncertain

Claude should choose the simpler solution that preserves:

- environment quality
- explainability
- demo reliability

### When adding features

Claude should ask:

- does this improve world modeling?
- does this improve the reward story?
- does this make the demo more compelling?
- can this be implemented and debugged quickly?

If the answer is no, deprioritize it.

---

## 20. Final Instruction to Claude

Treat this project as a **hackathon-winning environment design problem**, not as a generic drone simulator.

The code should make it easy to demonstrate that the agent is learning:

- better assignment
- better prioritization
- better disruption handling
- better battery/charging management
- better recovery after failure

Everything should serve that goal.
