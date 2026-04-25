# DroneZ Colab Training Guide

Colab notebook link:

- `https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV`

Repo notebook path:

- `notebooks/train_dronez_grpo_colab.ipynb`

This repo does **not** claim that a GRPO run already happened. The local repo provides a real environment-connected smoke harness, dry-run training prep, a dedicated local GPU training script, and a Colab-ready template path for actual TRL/Unsloth work.

If you have a laptop or workstation GPU, prefer `scripts/train_grpo_local.py` for the first real run. Use the Colab path when you need remote GPU compute.

## 1. Recommended Small Model

- `Qwen/Qwen2.5-0.5B-Instruct`

Larger options if GPU memory allows:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `google/gemma-2-2b-it`

## 2. Local Sanity Before Any GPU Run

```bash
python scripts/train_grpo.py --mode smoke
python scripts/train_grpo.py --mode dry-run
python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity
python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

These commands should write:

- `artifacts/results/training_smoke_metrics.json`
- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`
- `artifacts/training/local_sanity/sanity_check.json`

## 3. Local GPU Run

On a machine with a CUDA GPU:

```bash
pip install -e .[train]
python scripts/train_grpo_local.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --eval-tasks easy,medium,demo,hard --output-dir artifacts/training
```

This path is the repo's dedicated real-training entrypoint. It is the only tracked script intended to set `training_executed: true` after a real run completes.

## 4. Colab Setup

In Colab:

```bash
git clone https://github.com/SAICHAITU2012/Meta-Drone-Environment.git
cd Meta-Drone-Environment
pip install -e .[train]
python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

To check the dependency/template path:

```bash
python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

## 5. What The Training Path Uses

- Prompt source: DroneZ observation summary plus supported action list
- Action format: strict JSON with keys `action` and `params`
- Reward source: the real reward returned by `DroneZEnvironment.step(...)`
- Curriculum: `easy -> medium -> demo -> hard`
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`

## 6. Honest Expected Outputs

When you run only `--dry-run`, you should **not** claim training happened.

When you run `--sanity-check`, you should **not** claim training happened.

When a real local-GPU GRPO job is run, save:

- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/training_loss_curve.png` if available

Today, the Colab wrapper supports dry-run prep and dependency/template validation. It does not yet write those real-training artifacts by itself.

## 7. Creating A Public Colab Link

1. Open `notebooks/train_dronez_grpo_colab.ipynb` in Google Colab.
2. Save a copy to Drive.
3. Set sharing to anyone with the link can view.
4. Paste the public link into the README section `Submission Links To Fill Before Deadline`.

## 8. Recommended Judge Story

- Current measured improvement in this repo: deterministic `improved` policy vs baselines
- Local GPU path: implemented and sanity-checkable
- Colab path: template/dry-run validated
- Real trained-model claims should be added only after a real run finishes and artifacts are saved

## 9. Dependency Notes

- `pip install -e .[train]` now includes `peft` alongside `accelerate`, `datasets`, `transformers`, and `trl`.
- Keep `HF_TOKEN` in the shell environment if your model download requires it.
- Do not paste secrets directly into notebooks or committed scripts.

## 10. Commit Hygiene

- Do not commit large local checkpoints or generated model weights unless you intentionally want them in the repository.
- Commit the script/docs/tests changes first, then run training on your own machine and review the generated artifacts separately.

## 11. One-Cell Colab Preference

For linear notebook flows, prefer a single runnable cell for setup/validation commands where practical.

## 12. Handoff Summary

- local smoke and dry-run paths remain honest scaffolding
- `scripts/train_grpo_local.py` is the repo’s real local training entrypoint
- `scripts/train_grpo_colab.py` remains the safer dependency/template path until a full shared trainer is warranted
- real evidence still requires an actual GPU run

## 13. Keep Claims Narrow

Until a real run is completed, the correct statement is:

- the repo now contains a local GPU training script
- the repo does not yet contain committed evidence from a completed trained-model run

That is the claim boundary to preserve.

## 14. Source Of Truth For Training Outputs

The canonical judge-facing files remain:

- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`

Update any README or pitch claims only after those files come from a real run.

## 15. Practical Order Of Operations

1. Run smoke.
2. Run dry-run.
3. Run local sanity check.
4. Run local GPU training.
5. Review before/after metrics.
6. Update submission claims only if the real artifacts support them.

This keeps the training story honest and reproducible.

## 16. Existing Colab Notebook

The existing notebook remains useful for setup and template validation, but the new local script is the fastest path if you already have a GPU machine.

## 17. Final Reminder

No fake training claims. Only real artifacts count.

## 18. Original Colab Notes

Colab notebook link:

- `https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV`

Repo notebook path:

- `notebooks/train_dronez_grpo_colab.ipynb`

## 19. Appendix

The rest of this guide remains compatible with the updated flow.

## 20. Existing guidance below


