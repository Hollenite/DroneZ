# DroneZ Colab Training Guide

## Public Colab Link

[Open in Google Colab](https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV)

**Repo notebook path:** `notebooks/train_dronez_grpo_colab.ipynb`

## What This Notebook Does

The notebook is designed to be **judge-runnable** and **beginner-friendly**. It does not fake training success.

It runs:

1. Runtime and GPU checks
2. Repo clone / update
3. Safe dependency installation
4. Environment smoke test
5. Dry-run training setup
6. Action-format diagnostics
7. SFT warm-start data generation
8. Optional candidate-choice GRPO training (if GPU is available)
9. Plot and metrics summary
10. `artifacts/training/TRAINING_SUMMARY.md`
11. `dronez_training_artifacts.zip`

## How To Run In Colab

1. Open the [Colab link](https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV)
2. Go to **Runtime → Change runtime type**
3. Select a **GPU** runtime (T4 is fine)
4. Run cells from top to bottom
5. If a cell fails, copy the full error back into the repo issue/chat so it can be fixed

## Security

**Do not paste Hugging Face tokens** into the notebook, README, logs, screenshots, or chat.

If a model download needs authentication, use one of these:

- **Colab Secrets** (🔑 icon in the left sidebar → add `HF_TOKEN`)
- `os.environ["HF_TOKEN"]` (set before running)
- Manual `huggingface-cli login`

**Never print the token.**

## Commands Used By The Notebook

### Dry-run training setup
```bash
python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

### Sanity check
```bash
python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity
```

### Action-format diagnostics
```bash
python scripts/train_grpo_local.py --format-check --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,demo --output-dir artifacts/training/format_check
```

### SFT warm-start data
```bash
python scripts/generate_sft_action_data.py --tasks easy,medium,demo,hard --output artifacts/training/sft_action_data.jsonl
```

### Real Training (GPU only)

Primary:
```bash
python scripts/train_grpo_local.py --real-train --candidate-choice \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks easy,medium,demo \
  --eval-tasks easy,medium,demo,hard \
  --episodes 20 --group-size 4 \
  --output-dir artifacts/training/candidate_grpo
```

Smaller fallback (low VRAM):
```bash
python scripts/train_grpo_local.py --real-train --candidate-choice \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --tasks easy,demo \
  --eval-tasks easy,demo \
  --episodes 5 --group-size 2 \
  --output-dir artifacts/training/candidate_grpo_small
```

If CUDA/GPU is unavailable, the notebook skips real training gracefully and still produces dry-run, diagnostics, SFT data, plots, and a summary.

## Expected Outputs

- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`
- `artifacts/training/local_sanity/sanity_check.json`
- `artifacts/training/format_check/format_check.json`
- `artifacts/training/sft_action_data.jsonl`
- `artifacts/training/TRAINING_SUMMARY.md`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/training_loss_curve.png`
- `artifacts/plots/valid_action_rate.png`
- `artifacts/plots/eval_before_after.png`
- `dronez_training_artifacts.zip`

## How To Interpret Results

**You may claim:**

- DroneZ environment works
- Dry-run training setup works
- Action-format diagnostics work
- SFT warm-start data exists
- Candidate-choice training path is implemented

**Do not claim:**

- A GRPO-trained model improved reward, unless `eval_after` is better than `eval_before`
- Real drone flight control is solved
- Visual telemetry is real aircraft telemetry

## Honest Training Status

> Real trained-model reward improvement is not claimed unless `eval_after` improves over `eval_before`. Otherwise, the run is reported as an attempted training run with diagnostics.
