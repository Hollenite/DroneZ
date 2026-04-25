# DroneZ Submission Checklist

| Item | Command | Expected Output | Current Status |
| --- | --- | --- | --- |
| Editable install | `python -m pip install -e .` | Editable package installs successfully | `PASS` |
| Local tests | `pytest -q` | All tests pass | `PASS` |
| Evaluation artifacts | `python scripts/evaluate_policies.py` | Regenerates `artifacts/results/policy_comparison.json` and `.csv` | `PASS` |
| Demo traces | `python scripts/generate_demo_trace.py --task demo --policy all` | Regenerates all four demo trace files | `PASS` |
| Plot generation | `python scripts/generate_plots.py` | Regenerates the three comparison plots | `PASS` |
| Smoke training | `python scripts/train_grpo.py --mode smoke` | Writes `artifacts/results/training_smoke_metrics.json` | `PASS` |
| Dry-run training prep | `python scripts/train_grpo.py --mode dry-run` | Writes `artifacts/training/*.json` without claiming training happened | `PASS` |
| Colab entrypoint prep | `python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training` | Writes dry-run training artifacts through the Colab entrypoint | `PASS` |
| OpenEnv validation | `openenv validate` | Ready for multi-mode deployment | `PASS` |
| Local server endpoints | `python -m uvicorn server.app:app --host 127.0.0.1 --port 8000` plus `/health`, `/tasks`, `/reset`, `/state`, `/step` curls | Server binds and serves the OpenEnv-compatible API | `PASS` |
| Docker build | `docker build -t dronez .` | Docker image builds successfully | `PASS` |
| Docker run / local API | `docker run --rm -p 8000:7860 dronez` plus `/health`, `/tasks`, `/api`, `/artifacts/traces/demo_improved_trace.json` curls | Container serves the judge demo and OpenEnv-compatible API | `PASS` |
| Local GRPO sanity check | `python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity` | Honest dependency/GPU metadata without claiming training happened | `PASS` |
| Real local GRPO run | `python scripts/train_grpo_local.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --eval-tasks easy,medium,demo,hard --output-dir artifacts/training` on a local GPU machine | Real `eval_before`, `eval_after`, and training metrics | `NOT RUN` |
| Colab template check | `python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training` on Colab / hackathon GPU | Dependency/template validation and `grpo_template.json`, without claiming training happened | `NOT RUN` |
| HF Space deployment | Follow `HF_SPACE_DEPLOYMENT.md` | Live public Space with `/health`, `/tasks`, `/reset`, `/step`, `/state`, `/demo/index.html`, `/api` | `PASS` |
| HF link in README | Edit `README.md` Submission Links section | Real Space URL added | `PASS` |
| Video / blog / slides links | Edit `README.md` Submission Links section | Real supporting links added | `TODO` |
| Colab link in README | Edit `README.md` Submission Links section | Public Colab URL added | `PASS` |
| README review | Open `README.md` | Final judge-facing story is correct and honest | `PASS` |
| Pitch review | Open `PITCH.md` | Team stage answers are rehearsed | `PASS` |
| Junk-file cleanup | `git status --short` | No cache junk or accidental duplicate files | `PASS` |
| Team GitHub push | `git pull --rebase origin main && git push origin main` | Final code is pushed to `https://github.com/Hollenite/DroneZ.git` without force pushing | `PASS` |
| Clean git status before submission | `git status --short` | Only intended tracked changes remain before commit | `PASS` |

## Current Remaining Work

- Real GRPO / TRL / Unsloth training is still `NOT RUN`. Smoke, dry-run, Colab dry-run, and local GPU sanity checks are validated, but no trained-model improvement should be claimed yet.
- Video and slides links are still `TODO` unless the team publishes them before final submission.
- Large checkpoints and videos should stay out of the repo; link them externally.

## Final Human Actions Before Submission

1. Run the real GRPO/TRL job on Colab, Hugging Face Jobs, or hackathon compute if time/credits allow.
2. Replace video/slides and public blog links in README once they exist.
3. Rehearse the live demo flow with `https://krishna2521-dronez-openenv.hf.space`.
