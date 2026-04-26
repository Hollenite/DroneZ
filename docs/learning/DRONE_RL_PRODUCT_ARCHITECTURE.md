# Drone RL Product Architecture

This document explains the product vision for DroneZ as a hybrid drone operations platform.

## 1. The Big Product Idea

DroneZ is a B2B platform for simulating and improving drone fleet operations.

It is not trying to replace the real flight controller inside a drone. It is trying to improve the higher-level control tower decisions.

Simple version:

- Real drone systems keep aircraft stable and safe.
- DroneZ trains and evaluates mission-level fleet decisions.

## 2. Real Drone System Layers

### Layer 1: Physical Drone

This includes:

- Motors.
- Propellers.
- Battery.
- Payload.
- Sensors.
- Flight controller hardware.

DroneZ does not simulate detailed motor physics yet.

### Layer 2: Low-Level Flight Control

This keeps the drone stable.

Common methods:

- PID control.
- State estimation.
- Sensor fusion.
- Kalman-style filters.
- GPS waypoint navigation.
- Rule-based emergency logic.

This layer is better handled by classical robotics and certified flight software.

### Layer 3: Mission Autonomy

This decides what the drone should do in the mission.

Examples:

- Go to zone Z3.
- Avoid a no-fly zone.
- Return to charge.
- Attempt delivery.
- Use locker fallback.

DroneZ focuses here.

### Layer 4: Fleet Control Tower

This coordinates many drones.

Examples:

- Assign the best drone to an urgent order.
- Watch all battery levels.
- Monitor weather and restrictions.
- Balance speed, safety, energy, and cost.
- Recommend operator overrides.

DroneZ also focuses here.

## 3. B2B Platform Architecture

A company using DroneZ would configure its operations.

Configuration examples:

- Organization name.
- Industry.
- Drone types.
- Payload capacity.
- Battery capacity.
- Sensor suite.
- Weather tolerance.
- Safety strictness.
- Objective weights.

Then the platform can simulate scenarios and evaluate policies.

## 4. Organization Configurator

The organization configurator represents how a business would customize the environment.

Example:

- Medical logistics may value urgent delivery and safety highly.
- Food delivery may value speed and cost.
- Inspection may value sensor coverage and stable routes.
- Emergency response may value fast dispatch and recovery.

The current demo shows this as a UI concept. It does not retrain a model live in the browser.

## 5. Simulation Layer

The simulation layer creates a safe fake world.

It includes:

- Drones.
- Orders.
- City zones.
- Weather.
- Wind.
- No-fly zones.
- Chargers.
- Failed deliveries.
- Deadlines.

The browser visualization uses real environment traces plus derived visualization metadata.

Important boundary:

The UI telemetry is simulated and derived from environment state. It is not real aircraft telemetry.

## 6. Training Layer

The training layer is where an LLM-style policy learns to choose better actions.

The model receives an observation and should output an action.

Two important training steps:

- SFT warm start teaches the action format.
- Candidate-choice GRPO lets the model choose from valid candidate actions instead of inventing JSON from scratch.

This matters because the old real training attempt failed mainly due invalid actions.

## 7. Monitoring Layer

The monitoring layer is what an operator sees.

It should show:

- Active drones.
- Battery.
- Health.
- GPS lock.
- Sensor status.
- Assigned order.
- ETA.
- Weather alerts.
- No-fly zones.
- Reward and policy state.

The current demo implements a lightweight version of this control tower.

## 8. Control Tower Or Parent Server

Some companies may use a parent server that coordinates all drones.

That server can:

- Dispatch missions.
- Monitor fleet health.
- React to emergencies.
- Recommend route changes.
- Apply organization-level safety policy.

DroneZ can be viewed as an environment for training and testing that parent control policy.

## 9. Where Machine Learning Is Useful

ML and RL are useful for:

- Complex route adaptation.
- Long-horizon scheduling.
- Prioritizing urgent missions.
- Learning tradeoffs between speed, safety, battery, and cost.
- Handling unusual combinations of events.

## 10. Where Rule-Based Control Is Better

Rule-based or classical systems are better for:

- Emergency failsafes.
- Hard safety constraints.
- Geofencing.
- Low-level stability.
- Certified aviation behavior.

In real drones, you usually do not want a black-box LLM directly deciding motor commands.

## 11. Sim-To-Real Limitations

Simulation is useful, but it is not the real world.

Real deployment would still need:

- Real flight testing.
- Hardware-in-the-loop testing.
- Sensor noise modeling.
- Weather validation.
- Regulatory approval.
- Certified safety systems.
- Integration with a real autopilot stack.

DroneZ is a research and product prototype, not an aviation-certified system.

## 12. Future Real-World Path

A future version could connect DroneZ to:

- Gazebo.
- Isaac Sim.
- AirSim.
- Real map data.
- Real autopilot telemetry.
- Computer vision models.
- Hardware-in-the-loop simulation.

The current version is intentionally reliable and lightweight for Hugging Face Spaces.

