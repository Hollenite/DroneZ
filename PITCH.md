# DroneZ Pitch Pack

## 30-Second Pitch

DroneZ is a hybrid drone fleet operations RL environment. It does not train propellers. It trains and evaluates an LLM-style mission controller that assigns drones, reroutes around hazards, manages charging, handles failed drops, and prioritizes urgent orders under real operational constraints.

## One-Line Judge Framing

DroneZ shows the realistic hybrid model: PID, sensor fusion, GPS navigation, and safety rules keep drones stable, while RL/AI handles high-level fleet decisions through a control tower or parent server.

## 60-Second Pitch

Most drone demos are about flight. Ours is about operations. DroneZ simulates what actually makes delivery systems hard: dynamic weather, shifting no-fly zones, urgent orders, charger contention, battery risk, failed deliveries, and a heterogeneous fleet. We built it as a step-by-step OpenEnv environment so an LLM agent can act inside it, get reward feedback, and improve. We also built deterministic traces and an advanced control-tower replay UI so judges can see the drone routes, telemetry, weather, RL decision, reward impact, and hybrid control stack at the same time.

## 2-Minute Stage Pitch

DroneZ tackles a real capability gap in LLM agents: persistent operational control in a dynamic world. The agent is not answering one prompt. It is repeatedly observing a logistics system, deciding an action, seeing the consequences, and adapting. That makes it a strong OpenEnv problem.

Our environment models a professional drone-delivery control room. The agent sees fleet state, order backlog, sector hazards, no-fly zones, charger occupancy, and recent disruptions. It can assign drones, reroute flights, send drones to charge, hold a risky zone, fall back to lockers, and recover from failed drops.

This is a hybrid drone system, not a pure-RL aircraft controller. Low-level drone systems handle PID stability, sensor fusion, state estimation, GPS navigation, and safety logic. DroneZ focuses on high-level RL/AI decisions: fleet assignment, route adaptation, charging, recovery, and mission optimization. The parent-server control tower is the AI decision layer companies would customize and validate before connecting to real aircraft systems.

The reward system is decomposed and judge-explainable. We reward successful deliveries, urgent deliveries, meeting deadlines, safe reroutes, battery-aware operation, utilization, and recovery. We penalize missed deadlines, unsafe routing, battery critical states, invalid actions, unnecessary reroutes, charging misuse, and looping. We also explicitly guard against reward hacking by capping invalid actions and action count and by making safety reroutes distinct from unnecessary reroutes.

For evaluation, we compare random, naive, heuristic, and improved policies and export JSON, CSV, plots, and trace logs. For the demo, we replay real traces in the browser so you can see drones, sectors, hazards, telemetry, control-tower state, route changes, orders, and rewards evolve over time. The deterministic improved policy proves the environment can distinguish good and bad decisions. The GRPO attempt revealed the next real research bottleneck: action-format learning. Our next step is SFT warm start plus candidate-choice GRPO.

## Technical Judge Q&A

### What exactly is the agent learning?

It is learning operational sequencing under delayed consequences: which drone to assign, when to reroute, when to hold a zone, when to recover with lockers, and how to manage battery and deadlines across multiple assets.

### Is DroneZ training propellers?

No. DroneZ is mission-level drone fleet control. Classical drone systems handle stability and low-level movement. The RL/AI policy handles fleet and mission decisions from the control tower layer.

### Why is this not just a toy drone simulator?

Because the hard part is not toy movement. The hard part is world state, disruptions, tradeoffs, delayed reward, safety, and parent-server decisions across a fleet. DroneZ is mission control, not joystick control.

### How do rewards avoid hacking?

- multiple reward components instead of one scalar proxy
- explicit invalid-action penalties
- action caps and invalid-action caps
- separate safety reroute reward so good safety behavior is not punished
- deterministic demo scenarios for manual inspection
- step and episode reward breakdowns for auditability

### How does this fit OpenEnv?

It uses the standard environment pattern:

- `reset`
- `step`
- `state`

It is packaged as a FastAPI app with an OpenEnv-style manifest and Docker entrypoint for Space deployment.

### Why do you have both heuristic and improved policies?

The heuristic is the stable hand-built baseline. The improved policy is the deterministic scripted judge-facing policy that shows what good operational behavior looks like before and after training infrastructure is run on hackathon compute.

### What would companies customize?

They would customize drone profiles, payload and battery constraints, routing preferences, compliance rules, charger layout, failure characteristics, cost assumptions, and mission priority types. We already added richer deployment profiles to support that direction.

## Non-Technical Explanation

Think of DroneZ like an air-traffic and logistics controller for delivery drones. The AI is deciding which drone should do which job, which jobs are urgent, which areas are temporarily unsafe, when a drone needs charging, and how to recover if a delivery goes wrong.

## What Did the Agent Learn?

- do not keep choosing invalid actions
- do not send drones into unsafe zones without adaptation
- use rerouting for safety, not randomly
- recover failed deliveries with locker fallback when recipients are unavailable
- keep the fleet moving instead of idling with pending work

## Why This Matters

If LLM agents are going to run real operational workflows, they need to learn to maintain state, react to changing constraints, and optimize over multiple competing objectives. DroneZ is a compact but realistic testbed for that.

## Five Likely Judge Questions

### 1. What exactly is trained?

The thing being trained is the mission-control policy: the model chooses one structured operational action at a time from the DroneZ observation, then gets the next observation and reward from the environment.

### 2. What is the observation / action / reward loop?

The loop is `reset -> observe fleet/orders/sectors/charging/events -> choose one JSON action -> environment executes -> reward + done + next observation -> repeat`.

### 3. Why is this OpenEnv and not just a simulator?

Because it is packaged as an environment with explicit `reset`, `step`, and `state`, a FastAPI runtime, Docker packaging, and an OpenEnv manifest. The simulator is only one layer; the project is built to be used inside an RL training loop.

### 4. What evidence shows improvement?

This repo currently shows deterministic improvement: `improved` beats `heuristic`, `random`, and `naive` on the exported benchmark artifacts, while keeping zero invalid actions and zero safety violations in the current sweep. A real local GRPO-style run was attempted, but it did not improve because the model produced invalid actions. We are honest about that and now use action-format SFT plus candidate-choice prompts as the next training step.

### 5. What remains after the hackathon?

The main post-hackathon step is running the improved training ladder: action-format SFT, candidate-choice GRPO, then longer online RL at larger scale. We also want to expand configurable fleet profiles into customer-specific operational variants.

### 6. Why did the first real training attempt not improve?

It revealed an action-format bottleneck. The model did not reliably emit valid DroneZ JSON actions, so episodes hit `invalid_action_cap_reached`, rewards were identical, and the optimizer had no useful learning signal. That is why we added compact prompts, JSON repair, candidate-choice mode, and SFT action data.

### 7. Is this controlling real drone motors?

No. DroneZ is mission-level fleet control. Real drones still need certified low-level flight stacks: PID control, GPS/IMU fusion, Kalman filtering, geofencing, hardware failsafes, and aviation-grade validation.

### 8. What should judges notice in the new UI?

The demo is trace-driven, not fake. It shows a professional control tower with a 2.5D city map, curved route corridors, no-fly/weather overlays, simulated telemetry, control tower queues, RL recommendations, and a clear split between low-level drone control and high-level RL mission decisions.
