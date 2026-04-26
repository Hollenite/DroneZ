# DroneZ Cinematic Hybrid Control-Tower Simulation

This is a separate standalone prototype. It does not replace the working Hugging Face demo and does not modify the production API.

## Goal

Show DroneZ as a professional hybrid drone operations control tower:

- low-level drone systems: PID, sensor fusion, Kalman-style state estimation, GPS, and safety rules
- high-level AI/RL system: assignment, rerouting, charging, recovery, and urgent-order decisions
- parent/control tower: fleet monitoring, dispatch queue, route optimization, alerts, and human override

## How To Run

From the repo root:

```bash
python -m http.server 8090
```

Then open:

```text
http://127.0.0.1:8090/cinematic_sim/index.html
```

The simulation tries to load:

```text
../artifacts/traces/demo_improved_enriched.json
```

If the trace cannot be loaded, it uses built-in fallback frames so the page still works.

## Important Honesty Boundary

This is a cinematic mission-control visualization. It is not real aircraft telemetry, not certified aviation software, and not a full Gazebo/Isaac Sim physics simulator.

