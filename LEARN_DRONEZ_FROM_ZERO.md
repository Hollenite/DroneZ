# Learn DroneZ From Zero

This guide explains DroneZ in beginner-friendly language. You do not need to know reinforcement learning, drones, Hugging Face, or FastAPI before reading this.

## 1. What DroneZ Is

DroneZ is a simulated drone fleet operations environment.

That means it creates a fake but realistic world where a company has multiple drones, delivery orders, weather problems, no-fly zones, charging stations, deadlines, and urgent deliveries.

The AI agent does not directly spin propellers. It acts like a control tower or fleet manager.

It decides things like:

- Which drone should take which order.
- Whether a drone should reroute around a dangerous zone.
- Whether a drone should go charge before taking a new mission.
- Whether urgent medical orders should be prioritized.
- Whether a failed delivery should fall back to a locker.
- Whether an unsafe zone should be held until conditions improve.

## 2. What We Built

The project has four main parts:

- A Python environment that simulates the drone delivery world.
- A FastAPI server that exposes the environment through HTTP endpoints.
- A browser demo that replays real environment traces.
- Training scripts that prepare an LLM-style model to produce valid DroneZ actions.

The browser demo is not the core environment. It is the visual layer that helps judges and beginners understand what the environment is doing.

## 3. What Problem DroneZ Solves

Real delivery drones need more than pathfinding.

A fleet operator must handle:

- Many drones at once.
- Different drone types.
- Battery limits.
- Charging station capacity.
- Weather.
- No-fly zones.
- Late deadlines.
- Urgent medical deliveries.
- Failed drops.
- Safety rules.

DroneZ turns that problem into a reinforcement learning environment so policies can be evaluated and improved.

## 4. What An RL Environment Means

RL means reinforcement learning.

In simple language:

1. The agent observes the current situation.
2. The agent chooses an action.
3. The environment applies that action.
4. The environment gives a reward.
5. The agent tries to learn which actions get better rewards.

In DroneZ:

- Observation means the current fleet, orders, city, weather, chargers, and recent events.
- Action means one mission-level command such as assigning, rerouting, charging, or delivering.
- Reward means a score that says whether the decision helped or hurt.

## 5. Observation, Action, Reward

### Observation

An observation is the information the agent sees before deciding.

Example observation data:

- Drone FA-1 is at the hub.
- Drone HE-1 has 82 percent battery.
- Order O3 is urgent.
- Zone Z2 has a no-fly warning.
- Charging station C_HUB has one free slot.
- A recipient is unavailable.

### Action

An action is the one command the agent sends next.

Example actions:

```json
{"action": "assign_delivery", "params": {"drone_id": "FA-1", "order_id": "O3"}}
```

```json
{"action": "reroute", "params": {"drone_id": "FA-1", "corridor": "safe"}}
```

```json
{"action": "return_to_charge", "params": {"drone_id": "HE-1"}}
```

### Reward

Reward is the score after the action.

Good behavior gets positive reward:

- Successful delivery.
- Urgent delivery completed.
- Deadline met.
- Safe reroute.
- Battery-safe operation.
- Locker fallback after failed delivery.

Bad behavior gets negative reward:

- Invalid action.
- Missed deadline.
- Unsafe route.
- Critical battery.
- Unnecessary reroute.
- Looping without progress.

## 6. How A Drone Delivery Episode Works

An episode is one full simulation run.

Step by step:

1. The environment starts with a task such as `demo`, `easy`, `medium`, or `hard`.
2. Drones, orders, city sectors, weather, and chargers are created.
3. The agent gets an observation.
4. The agent chooses one action.
5. The simulation updates the world.
6. Reward is calculated.
7. The next observation is returned.
8. This repeats until the episode ends.

An episode can end when:

- All orders are resolved.
- The maximum number of steps is reached.
- Too many invalid actions happen.
- No useful drones remain.

## 7. What The Agent Does

The agent is a mission controller.

It is not controlling:

- Motor speed.
- Propeller thrust.
- Roll, pitch, and yaw directly.
- Real camera pixels.
- Real LiDAR point clouds.

It is controlling:

- Fleet assignment.
- Order priority.
- Route adaptation.
- Charging decisions.
- Delivery attempts.
- Fallback recovery.
- Safety holds.

That is why we call DroneZ mission-level control.

## 8. Why This Is Not Low-Level Motor Control

A real drone usually has a flight controller. That flight controller uses classical robotics methods:

- PID control for stability.
- Sensor fusion to combine GPS, IMU, camera, and other sensors.
- Kalman-style state estimation to guess the true position and motion.
- GPS navigation for waypoints.
- Rule-based safety logic for emergencies.

DroneZ sits above that layer.

Think of it like this:

- Low-level controller: keeps one drone stable and flying.
- DroneZ controller: decides what the fleet should do next.

## 9. Why Hybrid Drone Systems Matter

Pure RL is not usually the safest way to fly real delivery drones today.

The better industry idea is hybrid intelligence:

- Classical control handles stability.
- Sensor fusion handles state estimation.
- Safety rules handle emergencies.
- RL or AI handles mission-level decision making.

DroneZ is designed to demonstrate that hybrid architecture.

## 10. What Is Trained And What Is Not Trained

What we want to train:

- An LLM-style policy that reads DroneZ observations and outputs valid JSON actions.

What is already proven:

- The deterministic improved policy beats the baselines on the current benchmark and keeps safety violations and invalid actions at zero in the demo.

What is not yet proven:

- A trained GRPO model has not yet shown reward improvement in tracked results.

The old RTX 5060 run executed, but it did not improve. It produced invalid actions, reward stayed at `-90.0`, and loss stayed at `0.0`.

## 11. How Companies Would Use DroneZ

A company could use a platform like DroneZ to test drone fleet policies before real deployment.

Example company settings:

- Industry: medical logistics, e-commerce, inspection, emergency response.
- Drone type: light delivery, heavy payload, long range, surveillance.
- Payload capacity.
- Battery capacity.
- Sensor suite.
- Weather tolerance.
- Safety strictness.
- Objective weights for speed, safety, energy, cost, and urgent delivery.

The company would simulate many scenarios, compare policies, and only later connect to real drones after safety validation.

## 12. The One Big Idea

DroneZ is not "AI flies propellers."

DroneZ is "AI helps a control tower make better fleet decisions while classical drone systems keep the aircraft stable and safe."

