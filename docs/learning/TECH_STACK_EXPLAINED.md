# Tech Stack Explained

This guide explains each technology in DroneZ in simple language.

## Python

Python is the main backend language.

DroneZ uses Python for:

- The RL environment.
- The simulation engine.
- Reward calculation.
- Evaluation scripts.
- Training scripts.
- FastAPI server.

## FastAPI

FastAPI is the web server framework.

It lets DroneZ expose endpoints such as:

- `/health`
- `/tasks`
- `/reset`
- `/step`
- `/state`
- `/api`
- `/docs`

The browser demo and external tools can call these endpoints.

## OpenEnv

OpenEnv is the environment packaging idea.

It standardizes how an AI environment exposes:

- Reset.
- Step.
- State.
- Health.
- Manifest metadata.

DroneZ is built to fit that pattern.

## Docker

Docker packages the app so it can run the same way on different machines.

Hugging Face Spaces reads the Dockerfile, builds an image, and runs the server inside a container.

## Hugging Face Spaces

Hugging Face Spaces hosts the live DroneZ app.

In this project, the Space runs the FastAPI Docker app and serves:

- The API.
- The demo UI.
- The artifacts.
- The docs page.

The Space UI can be modified because it is just serving files from the repo, especially `demo_ui/index.html`, `demo_ui/app.js`, and `demo_ui/styles.css`.

## Hugging Face Models

Hugging Face also hosts pretrained AI models.

DroneZ training scripts use a model name such as:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

That model is asked to read an observation and output a DroneZ action.

## PyTorch

PyTorch is the machine learning library used for model training.

It handles tensors, GPU execution, backpropagation, and optimizer updates.

## Qwen Model

Qwen is the LLM family used in the training attempt.

In DroneZ, the Qwen model is not controlling real drone motors. It is used as an LLM-style mission policy that should output JSON actions.

## TRL

TRL means Transformer Reinforcement Learning.

It is a Hugging Face library for training language models with methods such as SFT, DPO, PPO, and GRPO.

DroneZ includes scripts that prepare a GRPO-style training path.

## Unsloth

Unsloth is a tool that can make LLM fine-tuning faster and lighter on GPU memory.

It is useful for hackathon training on limited GPUs or Colab, but the DroneZ environment itself does not depend on Unsloth to run.

## Google Colab

Google Colab is a cloud notebook environment.

It can provide a GPU when a local laptop does not have enough compute.

In DroneZ, Colab is useful for:

- Running model training.
- Testing training scripts.
- Generating real training artifacts.

## GPU

GPU means graphics processing unit.

LLM training usually needs a GPU because model updates are expensive. The old RTX 5060 run used a GPU, but the model output invalid actions and did not improve reward.

This local Mac environment can validate code and run the simulation, but real GRPO training requires CUDA.

## HTML, CSS, JavaScript

These are the browser technologies for the demo UI.

- HTML defines the page structure.
- CSS controls the professional visual style.
- JavaScript loads traces and animates the simulation.

The current UI is intentionally static and lightweight so Hugging Face Spaces can run it reliably.

## SVG

SVG is used to draw the simulation map.

It draws:

- City zones.
- Drone icons.
- Curved route paths.
- Chargers.
- Orders.
- Weather/no-fly overlays.

SVG is more reliable than a heavy 3D engine inside a hackathon Space.

## Canvas And Three.js

Canvas and Three.js are possible future options.

- Canvas is good for high-performance 2D rendering.
- Three.js is good for real 3D/WebGL scenes.

DroneZ currently chooses reliable SVG/2.5D visuals for Hugging Face. Full WebGL 3D is possible in a browser, but it adds risk for judge demos because hardware, browser, and network conditions can vary.

## GitHub Team Repo

GitHub stores the source code.

The team repo is:

```text
https://github.com/Hollenite/DroneZ.git
```

GitHub is where teammates review, clone, and version the project.

## How Everything Connects

The full flow is:

1. Python simulation creates the drone operations world.
2. FastAPI exposes the environment through HTTP.
3. OpenEnv manifest describes how tools can call it.
4. Evaluation scripts run policies and save results.
5. Trace scripts save frame-by-frame replays.
6. Enrichment script adds visualization metadata.
7. HTML/CSS/JS demo loads the traces and shows the control tower.
8. Training scripts prepare or run model optimization.
9. Docker packages everything.
10. Hugging Face Space hosts it live.

