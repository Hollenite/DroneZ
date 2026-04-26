# DroneZ Fleet Delivery Environment

Showcase: https://drive.google.com/drive/folders/1EvQvEcNg9AasMICUYt_AkMDGtoTlMbJE?usp=sharing
Hugging Face Space Repository: https://huggingface.co/spaces/Krishna2521/dronez-openenv
Live Runtime: https://krishna2521-dronez-openenv.hf.space
Live Demo: https://krishna2521-dronez-openenv.hf.space/demo/index.html
API Docs: https://krishna2521-dronez-openenv.hf.space/docs
Health Check: https://krishna2521-dronez-openenv.hf.space/health
GitHub Team Repo: https://github.com/Hollenite/DroneZ.git

---

## Introduction

When we order something online through delivery apps, we generally expect another human to arrive at our doorstep with our parcel. Sometimes they arrive early, sometimes late—mainly due to traffic on the road.

Roughly a decade ago, drones also began to emerge as delivery agents—or should I say delivery bots—in countries like China and Japan. When we think of a drone delivering our parcel, we usually assume it would be quick and efficient as there is no traffic in the air and routes are more direct.

This is where I'd want you to take a pause. The drones used now generally has a fixed route. Not the best, not the quickest nor the most efficient but the most predictable. you see, Even though drones go through air-route they can be more late than a human simply cuz it can't make decisions, assess routes for unexpected mishaps such as weather or unexpected obstacle.

So, to make the drone efficient, the drone needs to make decision. but how? How does the drone know what is the right decision in various situation? That is where DroneZ environment is useful.

The environment trains the agent (drone) with uncertain and unexpected conditions such as weather, obstacles etc. which simulates the real world.

The drone makes decisions at every step. every step is evaluated. Decisions are rewarded for successful delivery, efficiency, avoiding crashes and it will be penalised for failed delivery, crashes, delayed delivery and unnecessary large routes. the drone will be likely to repeat steps which yields the most reward. So, it will be able to make decisions with atmost probability of them being right and

---

## Project Access

**Hugging Face Space (Repository)**: https://huggingface.co/spaces/Krishna2521/dronez-openenv

**Live Runtime**: https://krishna2521-dronez-openenv.hf.space

**API Documentation**: `/docs`

**Health Check Endpoint**: `/health`

---

## Interactive Demonstration Interface

The browser replay interface visualizes the environment using real JSON traces, presenting the system as a sophisticated hybrid-drone control tower. Rather than displaying a simple grid, the interface renders a 2.5D procedural city with curved route corridors that connect active drones to their delivery destinations. Charging stations, weather overlays, and no-fly zones are dynamically represented, while simulated telemetry tracks reward evolution and recent events. The control tower state monitoring provides real-time insights into fleet operations, creating an immersive visualization that reflects actual environment behavior.

**Note**: Route geometry, wind values, sensor indicators, and control-layer labels are derived visualization metadata, not real-world GPS or sensor streams.

---

## Running the Environment Locally

```bash
python -m http.server 8080
```

Open in your browser:
```
http://localhost:8080/demo_ui/index.html
```

---

## Running on Hugging Face

```
https://krishna2521-dronez-openenv.hf.space/demo/index.html
```

---

## Reward System

The DroneZ environment encourages intelligent behaviour through an efficient reward system. Successful delivery yields higher rewards, while for urgent delivery additional reward points are awarded. The system is quite adaptive, it rewards for safe rerouting decisions and recovery from disruptions. It shows flexibility is also valued alongside reliability. Battery-safe operation and efficient fleet utilization are rewarded to promote sustainable operations. While regulatory compliance and lockout fallback mechanisms ensure the system operates within established safety and operational parameters.

### Negative Penalties

To avoid inefficient and unsafe behaviors, the environment implements a structured penalty system. Invalid actions, missed deadlines, failed delivery attempts and critical battery events will be penalised. The system penalizes unsafe zone entry and unnecessary reroutes that waste time and energy. Abandoned urgent orders, idle fleet behavior, and charging misuse represent operational inefficiencies that are actively discouraged and penalized. Additionally, loop or no progress behaviors is penalized to prevent agents from getting stuck in unproductive patterns, ensuring continuous learning toward meaningful actions.

---

## Conclusion

Drone delivery isn't just about flying – it's about decision-making under uncertain conditions. DroneZ shifts focus from fixed routes to adaptive intelligence, where every action is learned, evaluated, and improved. The result is a system that is efficient and quick.
