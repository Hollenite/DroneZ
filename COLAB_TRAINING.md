# DroneZ Colab Training Guide

Colab notebook link:

- `https://colab.research.google.com/drive/1ge0s9eYcbeE25oEXh6t-wySGh3ZCR9AV`

Repo notebook path:

- `notebooks/train_dronez_grpo_colab.ipynb`

This repo does **not** claim that a GRPO run already happened. The local repo provides a real environment-connected smoke harness, dry-run training prep, and a Colab-ready path for actual TRL/Unsloth work.

## 1. Recommended Small Model

- `Qwen/Qwen2.5-0.5B-Instruct`

Larger options if GPU memory allows:

- `Qwen/Qwen2.5-1.5B-Instruct`
- `google/gemma-2-2b-it`

## 2. Local Sanity Before Colab

```bash
python scripts/train_grpo.py --mode smoke
python scripts/train_grpo.py --mode dry-run
python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

These commands should write:

- `artifacts/results/training_smoke_metrics.json`
- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`

## 3. Colab Setup

In Colab:

```bash
git clone https://github.com/SAICHAITU2012/Meta-Drone-Environment.git
cd Meta-Drone-Environment
pip install -e .[train]
python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

To check the TRL dependency path:

```bash
python scripts/train_grpo_colab.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training
```

## 4. What The Training Path Uses

- Prompt source: DroneZ observation summary plus supported action list
- Action format: strict JSON with keys `action` and `params`
- Reward source: the real reward returned by `DroneZEnvironment.step(...)`
- Curriculum: `easy -> medium -> demo -> hard`
- Default model: `Qwen/Qwen2.5-0.5B-Instruct`

## 5. Honest Expected Outputs

When you run only `--dry-run`, you should **not** claim training happened.

When a real TRL/GRPO job is run on Colab or hackathon compute, save:

- `artifacts/training/training_metrics.json`
- `artifacts/training/eval_before.json`
- `artifacts/training/eval_after.json`
- `artifacts/plots/training_reward_curve.png`
- `artifacts/plots/training_loss_curve.png` if available

## 6. Creating A Public Colab Link

1. Open `notebooks/train_dronez_grpo_colab.ipynb` in Google Colab.
2. Save a copy to Drive.
3. Set sharing to anyone with the link can view.
4. Paste the public link into the README section `Submission Links To Fill Before Deadline`.

## 7. Recommended Judge Story

- Current measured improvement in this repo: deterministic `improved` policy vs baselines
- Real GRPO path: prepared and dry-run validated
- Real trained-model claims should be added only after a real run finishes and artifacts are saved
