# Training Status For The 3D Simulator

The simulator does not solve training by itself. It visualizes fleet behavior.

## Current honest status

- Deterministic improved policy is the proven improvement.
- Earlier real GRPO-style training did run, but it did not improve reward.
- The failure reason was invalid actions: the model produced actions that DroneZ could not execute.
- Therefore reward stayed flat and the trained model should not be claimed as improved.

## How to truly resolve the GRPO issue

The only honest ways to resolve `Real GRPO reward improvement is not proven` are:

1. Run real training with candidate-choice actions and SFT warm start.
2. Save real `eval_before.json`, `eval_after.json`, and `training_metrics.json`.
3. Show that `eval_after` is better than `eval_before`.
4. Only then update claims.

If training still does not improve, keep the pitch honest:

> DroneZ has a strong environment and deterministic improved policy. A real GRPO attempt exposed an action-format bottleneck. We added candidate-choice and SFT data so the next GPU run can train from valid actions first.
