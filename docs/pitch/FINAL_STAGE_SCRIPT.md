# DroneZ Final Stage Script

## 30-Second Explanation

DroneZ is an OpenEnv environment where an LLM can learn to act as a mission-level controller for delivery drones. It is not training propellers. Classical drone systems handle PID stability, sensor fusion, GPS navigation, and safety rules, while DroneZ trains and evaluates high-level fleet decisions under realistic disruption.

## 90-Second Explanation

Most drone demos focus on flying a drone. DroneZ focuses on the harder operational layer: deciding which drone should deliver which package, when to reroute, when to pause a risky zone, when to send drones to charge, and how to recover from failed drops.

The visual demo now looks like a hybrid-drone control tower. It shows a 2.5D mission map, curved route corridors, no-fly and weather overlays, simulated telemetry, control tower queues, RL recommendations, and the split between low-level flight systems and high-level RL/AI decisions.

The environment follows the OpenEnv loop: reset, observe, act, step, reward, state. We compare random, naive, heuristic, and improved policies. In the deterministic demo, the improved policy reaches reward `89.0` versus heuristic `32.0`, while reducing safety violations from `8` to `0`. A real local GRPO-style run was attempted, but it did not improve because the model failed the action-format step; we now show the fix path with action-format SFT data and candidate-choice prompts.

## 2-Minute Demo Narration

DroneZ is a fleet operations simulator for autonomous urban delivery. The agent is not flying propellers. It is running a control room.

At every step, it sees fleet battery, location, assigned order, target zone, ETA, city hazards, no-fly zones, charging pressure, and urgent orders. It then emits one JSON action such as assign a delivery, reroute a drone, hold a zone, attempt delivery, or fall back to a locker.

The important architecture is hybrid. PID, state estimation, sensor fusion, GPS, and safety rules represent what real drones already need for stable flight. The RL/AI layer is above that, acting like a parent server or control tower that coordinates all drones and optimizes mission decisions.

The reward is decomposed. We reward successful and urgent deliveries, deadline completion, safe reroutes, recovery, battery-safe operation, and compliance. We penalize missed deadlines, unsafe routing, battery critical states, invalid actions, unnecessary reroutes, and loops.

Our demo compares baseline behavior against an improved deterministic controller. On the demo scenario, the improved policy gets reward `89.0`, normalized score `0.5861`, zero invalid actions, and zero safety violations. The heuristic baseline gets reward `32.0` with eight safety violations. That is the core story: better fleet control is not just more deliveries; it is safer operation under disruption.

## Exact Live Demo Flow

1. Open `README.md` and show the one-line pitch.
2. Show the environment loop: `reset -> observation -> action -> step -> reward -> state`.
3. Open `artifacts/plots/reward_comparison.png`.
4. Open `demo_ui/index.html` locally or `https://krishna2521-dronez-openenv.hf.space/demo/index.html`.
5. Turn on Stage Demo Mode if presenting on a projector.
6. Load `demo` + `naive` or `random` and show poor reward/safety.
7. Switch to `heuristic` and show it is better but has safety violations.
8. Switch to `improved` and show reward `89.0` and safety violations `0`.
9. Point at the curved safe routes, telemetry, weather/no-fly overlays, and control tower panel.
10. Open `/docs` on the local server or HF Space to show OpenEnv-style API endpoints.
11. Open the Colab notebook link and explain dry-run versus real GRPO training.

## Judge Questions And Answers

### 1. What is the environment?

DroneZ is a simulated urban drone-delivery operations world for training LLM agents to control fleet-level decisions.

### 2. What does the agent observe?

It observes drones, orders, city sectors, no-fly zones, weather, charging stations, recent events, warnings, and reward feedback.

### 3. What actions can it take?

It can assign deliveries, reroute drones, reserve chargers, send drones to charge, prioritize orders, attempt delivery, use locker fallback, hold risky zones, and resume operations.

### 4. What is the reward?

The reward combines delivery success, urgent success, deadlines, safety, battery behavior, recovery, utilization, invalid actions, missed deadlines, unsafe zones, and loops.

### 4.1 Why are you showing PID, sensor fusion, and GPS if this is RL?

Because real drones are hybrid systems. PID, sensor fusion, GPS navigation, and safety rules handle stable flight. DroneZ uses RL/AI for the mission-control layer: assignment, route adaptation, charging, recovery, and control tower decisions.

### 5. What is actually trained?

The intended trained object is the LLM policy that maps observations to JSON actions. Real GRPO training is prepared but not claimed until the actual run is completed.

### 6. Is the improved policy trained or scripted?

It is scripted. It is a deterministic reference policy used for evaluation and demonstration. We are explicit about that.

### 7. Why is this not just a drone animation?

The animation replays real environment traces. The core artifact is the OpenEnv-compatible environment and reward loop.

### 8. What does OpenEnv add?

OpenEnv standardizes environment packaging, reset/step/state interaction, Docker deployment, and training integration.

### 9. How do you prevent reward hacking?

We use multiple reward components, invalid-action caps, action caps, safety checks, deterministic traces, and reward breakdown audits.

### 10. Why does heuristic complete more deliveries but score worse?

Because heuristic takes more unsafe routes. In drone delivery, safety and regulatory compliance matter as much as raw delivery count.

### 11. What will real GRPO training do?

It should optimize a model policy toward higher reward behavior by sampling actions in DroneZ, scoring rollouts, and updating model probabilities.

### 12. How does Unsloth help?

Unsloth can reduce memory and speed up LLM fine-tuning/RL workflows, which helps on Colab or hackathon GPU credits.

### 13. Why Google Colab?

The MacBook is good for environment development, but LLM RL needs GPU. Colab provides accessible GPU runtime for a small GRPO run.

### 14. How would a company customize drones?

Companies can adjust drone profiles: speed, payload, range, battery, charging rate, weather tolerance, failure risk, and operating cost.

### 15. What would you improve next?

Run action-format SFT, then candidate-choice GRPO, add more scenarios, wire deployment profiles into reset, and expand the replay UI into side-by-side comparison.

### 16. Is the new route and telemetry UI real GPS telemetry?

No. The replay uses real DroneZ environment traces. Extra map coordinates, route curves, wind, and sensor indicators are derived visualization metadata so judges can understand the simulated mission. They are labeled as simulated, not real aircraft telemetry.

## Non-Technical Explanation

DroneZ is like a control room for delivery drones. The AI decides which drone should take which package, when to avoid dangerous areas, when to charge, and how to recover if something goes wrong.
