# DroneZ Training Summary

This summary is intentionally conservative. It documents the runnable training path and diagnostics currently included in the repo. It does not claim trained-model reward improvement.

## Commands Validated

- `python scripts/train_grpo_colab.py --dry-run --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --output-dir artifacts/training`
- `python scripts/train_grpo_local.py --sanity-check --model Qwen/Qwen2.5-0.5B-Instruct --output-dir artifacts/training/local_sanity`
- `python scripts/train_grpo_local.py --format-check --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,demo --output-dir artifacts/training/format_check`
- `python scripts/generate_sft_action_data.py --tasks easy,medium,demo,hard --output artifacts/training/sft_action_data.jsonl`

## Current Artifacts

- Dry-run metrics: `artifacts/training/training_metrics.json`
- Dry-run evaluation placeholders: `artifacts/training/eval_before.json`, `artifacts/training/eval_after.json`
- Sanity check: `artifacts/training/local_sanity/sanity_check.json`
- Format check: `artifacts/training/format_check/format_check.json`
- SFT warm-start examples: `artifacts/training/sft_action_data.jsonl`

## Current Diagnostic Results

- `valid_json_rate`: `0.875`
- `valid_action_rate`: `0.875`
- SFT examples: `108`

## Real Training Status

A real GRPO-style run was attempted earlier, but reward improvement was not proven. The bottleneck was invalid model actions. The current Colab notebook and scripts now focus on candidate-choice prompting, robust action parsing, action repair, and SFT warm-start data.

Only claim trained-model improvement if a future run writes real `eval_before`, `eval_after`, and `training_metrics` artifacts where `eval_after` improves over `eval_before`.
