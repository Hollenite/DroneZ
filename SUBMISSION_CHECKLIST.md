# DroneZ Final Submission Checklist

Final submission URL to use:

```text
https://huggingface.co/spaces/Krishna2521/dronez-openenv
```

Only the team leader should submit. Submit one URL only, and avoid changing the repo after the official deadline.

## Required Hackathon Items

| Item | Status | Notes |
| --- | --- | --- |
| OpenEnv environment | `PASS` | `openenv validate` has passed. |
| Hugging Face Space pushed | `PASS` | Space repo and live runtime are available. |
| README includes HF Space link | `PASS` | README includes Space repo, runtime, demo, docs, and health links. |
| Working API endpoints | `PASS` | `/health`, `/tasks`, `/api`, `/reset`, `/step`, `/state`, `/docs`, and `/demo/index.html` are supported. |
| Training script | `PASS` | `scripts/train_grpo_local.py`, `scripts/train_grpo_colab.py`, and `scripts/train_grpo.py` are included. |
| Colab notebook | `PASS` | `notebooks/train_dronez_grpo_colab.ipynb` is present. |
| Public Colab link | `PASS` | README includes the public Colab URL currently provided by the team. |
| Training logs/artifacts | `PASS` | Small diagnostic artifacts exist in `artifacts/training/` and plots exist in `artifacts/plots/`. |
| Real trained reward improvement | `NOT PROVEN` | Do not claim GRPO model improvement unless a future run produces better `eval_after` than `eval_before`. |
| Evaluation results | `PASS` | README and BLOG include actual metrics from `artifacts/results/policy_comparison.json`. |
| Demo traces | `PASS` | Raw and enriched traces exist for `easy`, `medium`, `hard`, and `demo` across all four policies. |
| Blog/writeup | `PASS / TODO PUBLIC URL` | `BLOG.md` is ready locally. Add an external blog URL if published. |
| YouTube/video | `TODO` | Add public URL before final form submission if available. Do not commit video files. |
| Slides | `TODO` | Add public URL before final form submission if available. Do not commit large slide exports unless small. |
| No secrets | `PASS AFTER SCAN` | Keep all tokens in environment variables only. |
| No large checkpoints/videos | `PASS` | Do not commit model checkpoints, videos, or large generated assets. |
| Team GitHub repo | `PASS AFTER PUSH` | Push normally to `https://github.com/Hollenite/DroneZ.git`. Do not force push GitHub. |
| Hugging Face repo | `PASS AFTER PUSH` | Push final files to the HF Space repo. Force push is acceptable only for the Space. |

## Validation Commands

```bash
python -m pip install -e .
pytest -q
python scripts/evaluate_policies.py
python scripts/generate_demo_trace.py --task easy --policy all
python scripts/generate_demo_trace.py --task medium --policy all
python scripts/generate_demo_trace.py --task hard --policy all
python scripts/generate_demo_trace.py --task demo --policy all
python scripts/enrich_demo_trace.py --task easy --policy all
python scripts/enrich_demo_trace.py --task medium --policy all
python scripts/enrich_demo_trace.py --task hard --policy all
python scripts/enrich_demo_trace.py --task demo --policy all
python scripts/generate_plots.py
openenv validate
```

Training diagnostics:

```bash
python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity
python scripts/train_grpo_local.py --format-check --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,demo --output-dir artifacts/training/format_check
python scripts/generate_sft_action_data.py --tasks easy,medium,demo,hard --output artifacts/training/sft_action_data.jsonl
```

Local server:

```bash
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Docker:

```bash
docker build -t dronez .
docker run --rm -p 8000:7860 dronez
```

## Honest Claims To Use

- DroneZ is a mission-level drone fleet operations OpenEnv environment.
- It trains/evaluates assignment, routing, charging, recovery, and priority decisions.
- It does not train real propeller control.
- The deterministic improved policy beats baselines on reward and safety.
- The real GRPO attempt was run but did not improve because the model generated invalid actions.
- Candidate-choice formatting and SFT warm-start data were added to address that bottleneck.

## Claims Not To Use

- Do not say the GRPO-trained model improved unless a real future artifact proves it.
- Do not say the visualization is real aircraft telemetry.
- Do not say DroneZ is a full Isaac/Gazebo physics simulator.
- Do not say it is ready for real-world aviation deployment.

## Final Human Actions

1. Add YouTube/video link if the team publishes one.
2. Add slides link if the team publishes one.
3. Add public blog URL if published externally.
4. Confirm the team leader submits the Hugging Face Space URL.
5. Submit only once from the team leader email.
6. Revoke any token that was ever pasted into a chat or screenshot.
