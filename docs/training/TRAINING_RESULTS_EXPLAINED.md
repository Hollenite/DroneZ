# Training Results Explained

This document explains the DroneZ training status honestly and in beginner-friendly language.

## 1. The Most Important Truth

DroneZ has a working environment and a strong deterministic improved policy.

But the previous real GRPO-style model training run did not prove trained-model improvement.

That is not hidden. It is part of the technical story.

## 2. What Was Run Before

A real local GRPO-style run was attempted on a laptop with:

- GPU: NVIDIA GeForce RTX 5060 Laptop GPU.
- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Goal: train an LLM-style policy to output DroneZ actions.

The run executed, but it did not improve.

## 3. What Happened

The old run showed:

- Reward history stayed at `-90.0`.
- Loss history stayed at `0.0`.
- Every episode ended with `invalid_action_cap_reached`.
- `mean_reward_delta` was `0.0`.
- `eval_before` and `eval_after` were identical.
- The model produced invalid actions before and after training.

In simple language:

The model did not learn useful DroneZ behavior because it could not reliably speak the action language of the environment.

## 4. Why Reward Stayed At -90

The environment penalizes invalid actions.

If the model keeps producing invalid actions, the episode hits the invalid action cap. That means the environment ends early with bad reward.

If every rollout gets the same bad reward, GRPO has no useful comparison signal.

GRPO needs differences between candidates. If every candidate is equally bad, the optimizer cannot know what to increase or decrease.

## 5. Why Loss Stayed At 0

Loss can stay near zero when the group-normalized advantage is zero.

That happens when all sampled rollouts receive identical reward.

Simple analogy:

If a teacher gives every student the same score, you cannot tell who did better or worse.

## 6. Why Invalid Actions Happened

The base model was asked to generate structured DroneZ JSON actions.

This is harder than normal text generation because the output must match the environment exactly.

Common mistakes include:

- Extra text around JSON.
- Wrong action names.
- Missing parameters.
- Invalid drone IDs.
- Invalid order IDs.
- Markdown code fences.
- Single quotes instead of JSON double quotes.
- Nested action objects.

## 7. What We Added To Fix This

DroneZ now includes an action-format repair pipeline:

- Compact action prompts.
- Candidate-choice mode.
- Robust parser.
- Safe action repair.
- Format-check mode.
- SFT action-data generator.

## 8. What Candidate-Choice Mode Means

Instead of asking the model to invent an action from scratch, the environment generates valid candidate actions.

Then the model can output:

```json
{"choice": 2}
```

This is easier because the model only needs to choose among legal options.

That should reduce invalid actions and make the reward signal more useful.

## 9. What SFT Warm Start Means

SFT means supervised fine-tuning.

Warm start means teaching the model the action format before reinforcement learning.

DroneZ can generate examples from the improved policy:

- Observation summary.
- Valid action JSON.
- Task ID.
- Reward context.

The goal is simple:

First teach the model to speak valid DroneZ action JSON. Then use GRPO to improve the decisions.

## 10. Current Format Reliability

The current action-format diagnostics show:

- `valid_json_rate`: `0.875`
- `valid_action_rate`: `0.875`
- SFT examples generated: `108`

This is a major improvement over the old invalid-action failure, but it is not the same as proven reward improvement.

## 11. What Still Needs A Real GPU Run

The next real training command should be run on a CUDA GPU machine:

```bash
python scripts/train_grpo_local.py --real-train --candidate-choice --model Qwen/Qwen2.5-0.5B-Instruct --tasks easy,medium,demo --eval-tasks easy,medium,demo,hard --episodes 20 --group-size 4 --output-dir artifacts/training/candidate_grpo
```

If it improves, we can report:

- `eval_before`
- `eval_after`
- `mean_reward_delta`
- valid action rate
- reward curve
- loss curve

If it does not improve, we should say that honestly and inspect the next bottleneck.

## 12. What This Local Final Pass Could Run

This local machine does not have a CUDA GPU available.

The candidate-choice GRPO command was attempted with fewer episodes, but it stopped before model loading or optimizer updates because CUDA was unavailable. The repo writes blocked-run artifacts under:

```text
artifacts/training/candidate_grpo/
```

Those files are not training success evidence. They simply record:

- training did not execute
- the reason was missing CUDA
- no reward improvement was measured
- no `eval_before` versus `eval_after` comparison exists for that blocked run

## 13. What We Can Honestly Say Today

We can say:

- The environment works.
- The deterministic improved policy beats baselines.
- The old real GRPO attempt did not improve.
- The failure was diagnosed as an action-format bottleneck.
- Candidate-choice mode and SFT data were added to address it.
- A new CUDA training run is the next step to prove trained-model improvement.

We should not say:

- "The GRPO model improved" unless `eval_after` is actually better than `eval_before`.
- "Reward increased during training" unless the saved metrics prove it.
- "The model controls real drones" because DroneZ is mission-level simulation.
