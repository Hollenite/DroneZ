from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent

for candidate in (SCRIPT_DIR, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from train_grpo import build_prompt, dependency_status, parse_action_text, write_json
from urbanair.env.environment import DroneZEnvironment
from urbanair.eval.benchmark import benchmark_task_sweep
from urbanair.policies.base import Policy
from urbanair.policies.baseline import ImprovedPolicy

TRAINING_DIR = ROOT / "artifacts" / "training"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TASKS = ["easy", "medium", "demo"]
DEFAULT_EVAL_TASKS = ["easy", "medium", "demo", "hard"]


class LocalModelPolicy(Policy):
    def __init__(self, policy_id: str, trainer: "LocalGRPOTrainer", do_sample: bool = False) -> None:
        self.policy_id = policy_id
        self._trainer = trainer
        self._do_sample = do_sample

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return self._trainer.choose_action(observation, do_sample=self._do_sample)


class LocalGRPOTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.tasks = _parse_csv_list(args.tasks, fallback=DEFAULT_TASKS)
        self.eval_tasks = _parse_csv_list(args.eval_tasks, fallback=DEFAULT_EVAL_TASKS)
        self.dependencies = extended_dependency_status()
        self._torch = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.dtype_name = None
        self.optimizer = None
        self.continuation_policy = ImprovedPolicy()
        self.loss_history: list[float] = []
        self.reward_history: list[float] = []
        self.training_log: list[dict[str, Any]] = []

    def run(self) -> dict[str, Path]:
        self._validate_training_environment()
        self._seed_everything()
        self._load_model()

        eval_before = self._evaluate_model(policy_id="local_model_pre")
        baseline_reference = serialize_benchmark(
            benchmark_task_sweep([ImprovedPolicy()], tasks=self.eval_tasks, max_actions=self.args.max_actions)
        )

        overall_start = time.time()
        for episode_index in range(self.args.episodes):
            task_id = self.tasks[episode_index % len(self.tasks)]
            self.training_log.append(self._run_training_episode(task_id, episode_index + 1))

        eval_after = self._evaluate_model(policy_id="local_model_post")
        duration_seconds = round(time.time() - overall_start, 2)

        training_metrics = {
            "mode": "local-grpo",
            "training_executed": True,
            "note": (
                "This script ran a real local GPU GRPO-style online optimization loop against "
                "the DroneZ environment using group-normalized rollout rewards."
            ),
            "selected_model": self.args.model,
            "curriculum": self.tasks,
            "eval_tasks": self.eval_tasks,
            "dependency_status": self.dependencies,
            "device": self._device_summary(),
            "hyperparameters": {
                "seed": self.args.seed,
                "episodes": self.args.episodes,
                "group_size": self.args.group_size,
                "learning_rate": self.args.learning_rate,
                "max_new_tokens": self.args.max_new_tokens,
                "max_prompt_tokens": self.args.max_prompt_tokens,
                "temperature": self.args.temperature,
                "top_p": self.args.top_p,
                "max_continuation_steps": self.args.max_continuation_steps,
                "max_actions": self.args.max_actions,
            },
            "duration_seconds": duration_seconds,
            "training_log": self.training_log,
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "baseline_reference": baseline_reference,
            "pre_training_mean_reward": _aggregate_mean_reward(eval_before),
            "post_training_mean_reward": _aggregate_mean_reward(eval_after),
            "mean_reward_delta": round(_aggregate_mean_reward(eval_after) - _aggregate_mean_reward(eval_before), 4),
            "output_dir": str(self.output_dir),
        }

        eval_before_payload = {
            "status": "completed",
            "note": "Pre-training evaluation for the local generative policy before optimizer updates.",
            "baseline_reference": baseline_reference,
            "payload": eval_before,
        }
        eval_after_payload = {
            "status": "completed",
            "note": "Post-training evaluation for the local generative policy after optimizer updates.",
            "baseline_reference": baseline_reference,
            "payload": eval_after,
            "mean_reward_delta": training_metrics["mean_reward_delta"],
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_paths = {
            "training_metrics": write_json(self.output_dir / "training_metrics.json", training_metrics),
            "eval_before": write_json(self.output_dir / "eval_before.json", eval_before_payload),
            "eval_after": write_json(self.output_dir / "eval_after.json", eval_after_payload),
        }

        if self.args.save_model_dir:
            save_path = Path(self.args.save_model_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            result_paths["saved_model"] = save_path

        return result_paths

    def write_sanity_check(self) -> Path:
        payload = {
            "mode": "local-grpo-sanity",
            "training_executed": False,
            "note": (
                "This is a dependency and runtime validation path for the local GPU training script. "
                "It does not download a model or claim that training happened."
            ),
            "selected_model": self.args.model,
            "curriculum": self.tasks,
            "eval_tasks": self.eval_tasks,
            "dependency_status": self.dependencies,
            "device": self._device_summary(),
            "output_dir": str(self.output_dir),
        }
        destination = write_json(self.output_dir / "sanity_check.json", payload)
        return destination

    def choose_action(self, observation: dict[str, Any], *, do_sample: bool) -> dict[str, Any]:
        prompt = build_prompt(observation)
        text, _, _ = self._generate_completion(prompt, do_sample=do_sample)
        return safe_parse_action(text)

    def _run_training_episode(self, task_id: str, episode_number: int) -> dict[str, Any]:
        env = DroneZEnvironment(default_task_id=task_id, max_episode_actions=self.args.max_actions)
        observation, info = env.reset(task_id)

        episode_return = 0.0
        episode_loss_total = 0.0
        update_steps = 0
        invalid_action_count = 0
        done = False

        while not done:
            prompt = build_prompt(observation)
            candidates = []
            for _ in range(self.args.group_size):
                raw_text, prompt_ids, completion_ids = self._generate_completion(prompt, do_sample=True)
                action = safe_parse_action(raw_text)
                estimate = self._evaluate_candidate(env, action)
                candidates.append(
                    {
                        "raw_text": raw_text,
                        "action": action,
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion_ids,
                        **estimate,
                    }
                )

            rewards = self._torch.tensor(
                [candidate["estimated_total_return"] for candidate in candidates],
                dtype=self._torch.float32,
                device=self.device,
            )
            if rewards.numel() > 1:
                advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            else:
                advantages = self._torch.zeros_like(rewards)

            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for advantage, candidate in zip(advantages, candidates):
                logprob = self._mean_logprob_of_completion(candidate["prompt_ids"], candidate["completion_ids"])
                losses.append(-advantage.detach() * logprob)
            loss = self._torch.stack(losses).mean()
            loss.backward()
            self._torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            best_candidate = candidates[int(self._torch.argmax(rewards).item())]
            observation, reward, done, info = env.step(best_candidate["action"])
            episode_return += float(reward)
            invalid_action_count = int(info.get("invalid_action_count", invalid_action_count))
            update_steps += 1
            loss_value = float(loss.detach().cpu())
            episode_loss_total += loss_value
            self.loss_history.append(round(loss_value, 6))

        self.reward_history.append(round(episode_return, 4))
        return {
            "episode": episode_number,
            "task_id": task_id,
            "episode_return": round(episode_return, 4),
            "mean_loss": round(episode_loss_total / max(update_steps, 1), 6),
            "update_steps": update_steps,
            "invalid_action_count": invalid_action_count,
            "done_reason": info.get("done_reason", "unknown"),
            "terminated_by": info.get("terminated_by", "unknown"),
            "final_cumulative_reward": info.get("cumulative_reward", {}),
        }

    def _evaluate_candidate(self, base_env: DroneZEnvironment, action: dict[str, Any]) -> dict[str, Any]:
        env_copy = copy.deepcopy(base_env)
        observation, reward, done, info = env_copy.step(action)
        total_return = float(reward)
        if not done:
            total_return += self._rollout_with_baseline_policy(env_copy, observation, info)
        return {
            "estimated_total_return": total_return,
            "immediate_reward": float(reward),
            "done": bool(done),
            "invalid_action": bool(info.get("invalid_action", False)),
            "done_reason": info.get("done_reason", "unknown"),
        }

    def _rollout_with_baseline_policy(
        self,
        env: DroneZEnvironment,
        observation: dict[str, Any],
        info: dict[str, Any],
    ) -> float:
        total = 0.0
        done = False
        steps = 0
        while not done and steps < self.args.max_continuation_steps:
            action = self.continuation_policy.choose_action(observation, info)
            observation, reward, done, info = env.step(action)
            total += float(reward)
            steps += 1
        return total

    def _evaluate_model(self, *, policy_id: str) -> dict[str, Any]:
        self.model.eval()
        try:
            policy = LocalModelPolicy(policy_id=policy_id, trainer=self, do_sample=False)
            return serialize_benchmark(
                benchmark_task_sweep([policy], tasks=self.eval_tasks, max_actions=self.args.max_actions)
            )
        finally:
            self.model.train()

    def _generate_completion(self, prompt: str, *, do_sample: bool):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_prompt_tokens,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": do_sample,
            "max_new_tokens": self.args.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.args.temperature
            generation_kwargs["top_p"] = self.args.top_p

        with self._torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)

        prompt_length = input_ids.shape[1]
        completion_ids = outputs[0, prompt_length:].detach().cpu()
        if completion_ids.numel() == 0:
            completion_ids = self._torch.tensor([self.tokenizer.eos_token_id], dtype=self._torch.long)
        text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return text, input_ids[0].detach().cpu(), completion_ids

    def _mean_logprob_of_completion(self, prompt_ids_cpu, completion_ids_cpu):
        prompt_ids = prompt_ids_cpu.to(self.device)
        completion_ids = completion_ids_cpu.to(self.device)
        full_ids = self._torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0)
        attention_mask = self._torch.ones_like(full_ids, device=self.device)
        outputs = self.model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        targets = full_ids[:, 1:]

        prompt_length = prompt_ids.shape[0]
        completion_length = completion_ids.shape[0]
        start = prompt_length - 1
        completion_logits = logits[:, start : start + completion_length, :]
        completion_targets = targets[:, start : start + completion_length]
        token_logprobs = self._torch.log_softmax(completion_logits, dim=-1).gather(
            -1, completion_targets.unsqueeze(-1)
        ).squeeze(-1)
        return token_logprobs.mean()

    def _validate_training_environment(self) -> None:
        missing = [name for name in ("torch", "transformers") if not self.dependencies.get(name, False)]
        if missing:
            raise RuntimeError(
                "Local GRPO training requires the following Python packages to be installed: "
                + ", ".join(missing)
            )
        torch = self._import_torch()
        if not torch.cuda.is_available():
            raise RuntimeError("Local GRPO training requires a CUDA-enabled GPU runtime.")

    def _seed_everything(self) -> None:
        torch = self._import_torch()
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)
            torch.backends.cuda.matmul.allow_tf32 = True

    def _load_model(self) -> None:
        torch = self._import_torch()
        transformers = self._import_transformers()
        self.device = torch.device("cuda")
        capability_major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if capability_major >= 8 else torch.float16
        self.dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.train()
        self.model.config.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

    def _import_torch(self):
        if self._torch is None:
            import torch

            self._torch = torch
        return self._torch

    def _import_transformers(self):
        import transformers

        return transformers

    def _device_summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "gpu_available": False,
            "dtype": self.dtype_name,
        }
        if not self.dependencies.get("torch", False):
            return summary
        try:
            torch = self._import_torch()
        except Exception:
            return summary

        summary["gpu_available"] = bool(torch.cuda.is_available())
        summary["torch_version"] = getattr(torch, "__version__", None)
        if torch.cuda.is_available():
            summary["device_name"] = torch.cuda.get_device_name(0)
            summary["device_capability"] = list(torch.cuda.get_device_capability(0))
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local GPU GRPO-style training session for DroneZ.")
    parser.add_argument("--output-dir", default=str(TRAINING_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--eval-tasks", default=",".join(DEFAULT_EVAL_TASKS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompt-tokens", type=int, default=1536)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-continuation-steps", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--save-model-dir", default="")
    parser.add_argument("--sanity-check", action="store_true", help="Validate dependencies/GPU reporting without running training.")
    return parser


def extended_dependency_status() -> dict[str, bool]:
    status = dependency_status()
    for module_name in ("torch", "peft"):
        status[module_name] = importlib.util.find_spec(module_name) is not None
    return status


def safe_parse_action(text: str) -> dict[str, Any]:
    try:
        candidate = parse_action_text(text)
    except Exception:
        return {"action": "__invalid__", "params": {}}
    if not isinstance(candidate, dict):
        return {"action": "__invalid__", "params": {}}
    params = candidate.get("params")
    if not isinstance(params, dict):
        params = {}
    return {"action": candidate.get("action", "__invalid__"), "params": params}


def serialize_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "tasks": payload["tasks"],
        "ranking": payload["ranking"],
        "aggregate": payload["aggregate"],
        "policy_results": {
            policy_id: {task_id: summary.model_dump() for task_id, summary in summaries.items()}
            for policy_id, summaries in payload["policy_results"].items()
        },
    }


def _aggregate_mean_reward(payload: dict[str, Any]) -> float:
    aggregate = payload.get("aggregate", {})
    if not aggregate:
        return 0.0
    first_value = next(iter(aggregate.values()))
    return float(first_value.get("mean_total_reward", 0.0))


def _parse_csv_list(raw: str, *, fallback: list[str]) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or fallback


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    trainer = LocalGRPOTrainer(args)

    try:
        if args.sanity_check:
            destination = trainer.write_sanity_check()
            print(f"Wrote {destination}")
            print(destination.read_text())
            return 0

        result_paths = trainer.run()
        print(f"Wrote {result_paths['training_metrics']}")
        print(f"Wrote {result_paths['eval_before']}")
        print(f"Wrote {result_paths['eval_after']}")
        return 0
    except RuntimeError as exc:
        print(f"Local GRPO training is not ready: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
