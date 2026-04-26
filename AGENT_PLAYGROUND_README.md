# DroneZ Agent Playground

This document explains the Agent Playground for DroneZ.

Official runtime:

```text
https://krishna2521-dronez-openenv.hf.space/agent/
```

Official Hugging Face Space:

```text
https://huggingface.co/spaces/Krishna2521/dronez-openenv
```

## What `/agent` Is

`/agent` is a lightweight browser playground for the actual RL/OpenEnv loop. It is not just a replay dashboard. It lets a user reset the environment, inspect the observation JSON, choose a valid action, send that action to `/step`, and see the returned reward, done flag, and info.

The loop is:

```text
reset -> observation -> action -> step -> reward/done/info -> repeat
```

## How Reset And Step Work

`POST /reset` starts a new task episode.

Example:

```bash
curl -X POST https://krishna2521-dronez-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'
```

The response contains a `session_id`, an `observation`, and `info`.

`POST /step` applies one JSON action to the active environment state.

Example:

```bash
curl -X POST https://krishna2521-dronez-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action":{"action":"assign_delivery","params":{"drone_id":"FA-1","order_id":"O1"}}}'
```

The response contains the next `observation`, `reward`, `done`, and `info`.

## How To Use Valid Actions

`POST /valid-actions` returns candidate actions that are valid for the current state.

Example:

```bash
curl -X POST https://krishna2521-dronez-openenv.hf.space/valid-actions \
  -H "Content-Type: application/json" \
  -d '{}'
```

The Agent Playground displays these candidates as clickable cards. Clicking a card fills the action editor with valid JSON. This is useful for humans, scripted agents, and candidate-choice training where a model selects one legal action instead of inventing free-form JSON.

## Train An External Agent

A minimal Python loop looks like this:

```python
import requests

BASE = "https://krishna2521-dronez-openenv.hf.space"

reset = requests.post(f"{BASE}/reset", json={"task_id": "easy"}).json()
for step in range(20):
    candidates = requests.post(f"{BASE}/valid-actions", json={}).json()["actions"]
    action = candidates[0]["action"]
    result = requests.post(f"{BASE}/step", json={"action": action}).json()
    print(step, result["reward"], result["done"], result["info"].get("done_reason"))
    if result["done"]:
        break
```

Repo examples:

```bash
BASE_URL=https://krishna2521-dronez-openenv.hf.space python examples/remote_agent_quickstart.py
BASE_URL=https://krishna2521-dronez-openenv.hf.space python examples/random_remote_agent.py
```

## Official Space

The official DroneZ Space is:

```text
https://huggingface.co/spaces/Krishna2521/dronez-openenv
```
