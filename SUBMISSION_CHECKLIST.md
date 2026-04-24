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
| Docker build | `docker build -t dronez .` | Docker image builds successfully | `BLOCKED` |
| Real GRPO run | `python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training` on Colab / hackathon GPU | Real `eval_before`, `eval_after`, and training metrics | `NOT RUN` |
| HF Space deployment | Follow `HF_SPACE_DEPLOYMENT.md` | Live public Space with `/health`, `/tasks`, `/reset`, `/step`, `/state` | `NOT RUN` |
| HF link in README | Edit `README.md` Submission Links section | Real Space URL added | `PASS` |
| Video / blog / slides links | Edit `README.md` Submission Links section | Real supporting links added | `TODO` |
| Colab link in README | Edit `README.md` Submission Links section | Public Colab URL added | `PASS` |
| README review | Open `README.md` | Final judge-facing story is correct and honest | `PASS` |
| Pitch review | Open `PITCH.md` | Team stage answers are rehearsed | `PASS` |
| Junk-file cleanup | `git status --short` | No cache junk or accidental duplicate files | `PASS` |
| Clean git status before submission | `git status --short` | Only intended tracked changes remain before commit | `TODO` |

## Current Blocked Details

- Local server bind: blocked by sandbox with `error while attempting to bind on address ('127.0.0.1', 8000): operation not permitted`
- Docker build: blocked because Docker/Colima daemon is not running at `/Users/saichaitu/.colima/default/docker.sock`

## Final Human Actions Before Submission

1. Start Docker or Colima and rerun the Docker build.
2. Run the real GRPO/TRL job on Colab or hackathon compute.
3. Deploy the environment to a Hugging Face Space.
4. Test the live Space at `https://krishna2521-dronez-openenv.hf.space/health`.
5. Replace video/slides and public blog links in README once they exist.
6. Commit the final tree once you are satisfied with the generated artifacts.
