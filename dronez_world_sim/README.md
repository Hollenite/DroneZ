# DroneZ World Simulator

This is a separate premium 3D simulation layer for DroneZ. It does not replace the OpenEnv environment, FastAPI server, or Hugging Face demo. It is a standalone visual product demo that can read DroneZ trace artifacts and present them in a more game-like way.

## Why this exists

The normal Hugging Face demo is intentionally lightweight and reliable. This folder explores a more visually impressive direction:

- 3D drone world using Three.js
- 42-drone warehouse / drone-port fleet
- operations mode, follow mode, cinematic film mode, and warehouse mode
- 2D tactical minimap beside the 3D main world
- drone switching and live telemetry
- no-fly zones, weather zones, routes, buildings, terrain, hub, launch pads
- honest explanation that DroneZ controls mission-level fleet decisions, not propellers

## Stack chosen

I chose **Three.js** instead of Unity/Unreal because it runs directly in the browser, is easy to demo locally, can still look high-end, and can be hosted later without shipping a heavy engine project. React Three Fiber would be a good next step for a larger app, but plain Three.js is faster and cleaner for this hackathon-grade standalone prototype.

## Run locally

From the repo root:

```bash
python -m http.server 8090
```

Open:

```text
http://127.0.0.1:8090/dronez_world_sim/index.html
```

This page loads Three.js from a CDN, so it needs internet access the first time it runs.

## Best demo path

1. Open the simulator.
2. Click **Warehouse** to show the large drone-port fleet.
3. Select different drones from the dropdown.
4. Click **Launch Fleet**.
5. Switch to **Drone Follow** to show a game-like selected-drone camera.
6. Switch to **Cinematic Film** for product-video style camera movement.
7. Point to the minimap to explain routes, no-fly zones, storm zones, and drone positions.
8. Point to the Training Truth panel and say honestly: real GRPO improvement is not proven yet.

## Honest boundary

This is a cinematic visualization layer. It is not Gazebo, Isaac Sim, AirSim, certified drone physics, real camera/LiDAR data, or real aircraft telemetry. It is meant to visually communicate the DroneZ mission-control environment and hybrid architecture.

## Integration with DroneZ

The simulator tries to fetch:

```text
../artifacts/traces/demo_improved_enriched.json
```

If that file is present, the UI mentions the available trace frames. The current 3D world uses cinematic route geometry rather than real GPS coordinates.
