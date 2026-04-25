from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import os
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

from train_grpo import dependency_status, write_json
from urbanair.env.environment import DroneZEnvironment
from urbanair.eval.benchmark import benchmark_task_sweep
from urbanair.policies.base import Policy
from urbanair.policies.baseline import ImprovedPolicy
from urbanair.training.action_format import build_action_prompt, build_candidate_actions, parse_llm_action

TRAINING_DIR = ROOT / "artifacts" / "training"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TASKS = ["easy", "medium", "demo"]
DEFAULT_EVAL_TASKS = ["easy", "medium", "demo", "hard"]
HF_JOB_ENV_VARS = (
    "HF_JOB_ID",
    "HF_SPACE_ID",
    "SPACE_ID",
    "SPACE_AUTHOR_NAME",
    "SPACE_REPO_NAME",
)
SAFE_TOP_K = 50
SAFE_REPETITION_PENALTY = 1.05
TRAINING_DEFAULT_CANDIDATE_CHOICE = True
ACTION_GENERATION_TEMPERATURE = 0.35
ACTION_GENERATION_TOP_P = 0.9
ACTION_GENERATION_MAX_NEW_TOKENS = 48
PROMPT_PREVIEW_CHARS = 512
FAILED_EXAMPLE_LIMIT = 3


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
        self.hf_safe_generation = _is_hf_job_environment()
        self.use_safe_sampling = self.hf_safe_generation
        self.remove_invalid_values_supported = False
        self.renormalize_logits_supported = False
        self.generation_safety_flags: dict[str, Any] = {
            "hf_safe_generation": self.hf_safe_generation,
            "use_safe_sampling": self.use_safe_sampling,
            "remove_invalid_values": False,
            "renormalize_logits": False,
        }
        self.detected_transformers_version: str | None = None
        self.device_capability_major: int | None = None
        self.device_name: str | None = None
        self.training_dtype_name: str | None = None
        self.sampling_dtype_name: str | None = None
        self.sampling_model = None
        self.sampling_tokenizer = None
        self.sampling_device = None
        self.skipped_nonfinite_updates = 0
        self.latest_generation_kwargs: dict[str, Any] = {}
        self.latest_generation_used_sampling_model = False
        self.latest_generation_hf_safe = self.hf_safe_generation
        self.latest_generation_model_kind = "training"
        self.latest_generation_support_flags: dict[str, bool] = {}
        self.latest_generation_transformers_version: str | None = None
        self.latest_generation_dtype_name: str | None = None
        self.latest_generation_device_type: str | None = None
        self.latest_generation_temperature: float | None = None
        self.latest_generation_top_p: float | None = None
        self.latest_generation_top_k: int | None = None
        self.latest_generation_repetition_penalty: float | None = None
        self.latest_generation_removed_invalid_values = False
        self.latest_generation_renormalize_logits = False
        self.latest_generation_pad_token_id: int | None = None
        self.latest_generation_eos_token_id: int | None = None
        self.latest_generation_max_new_tokens: int | None = None
        self.latest_generation_device_capability_major: int | None = None
        self.latest_generation_device_name: str | None = None
        self.latest_generation_sampling_device_name: str | None = None
        self.latest_generation_sampling_dtype_name: str | None = None
        self.latest_generation_input_length: int | None = None
        self.latest_generation_completion_length: int | None = None
        self.latest_generation_use_cache = False
        self.latest_generation_do_sample = False
        self.training_candidate_choice = False if args.disable_training_candidate_choice else (TRAINING_DEFAULT_CANDIDATE_CHOICE if args.candidate_choice is None else bool(args.candidate_choice))
        self.training_generation_temperature = min(self.args.temperature, self.args.action_temperature)
        self.training_generation_top_p = min(self.args.top_p, self.args.action_top_p)
        self.training_generation_max_new_tokens = min(self.args.max_new_tokens, self.args.action_max_new_tokens)
        self.last_prompt_preview: str | None = None
        self.last_prompt_hash: str | None = None
        self.last_prompt_char_length = 0
        self.last_candidate_count = 0
        self.last_prompt_token_length: int | None = None
        self.last_completion_token_length: int | None = None
        self.last_generation_stop_reason: str | None = None
        self.last_parser_error_counts: dict[str, int] = {}
        self.last_failed_examples: list[dict[str, Any]] = []
        self.last_candidate_choice_rate = 0.0
        self.last_repair_rate = 0.0
        self.last_invalid_action_cap_hits = 0

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
            "generation_safety": self.generation_safety_flags,
            "sampling_model": {
                "enabled": self.use_safe_sampling,
                "dtype": self.sampling_dtype_name,
            },
            "action_generation": {
                "candidate_choice_default": self.training_candidate_choice,
                "temperature": self.training_generation_temperature,
                "top_p": self.training_generation_top_p,
                "max_new_tokens": self.training_generation_max_new_tokens,
            },
            "parser_error_code_counts": _parser_error_code_counts(self.training_log),
            "candidate_choice_rate": _training_rate(self.training_log, "used_candidate_choice"),
            "repair_rate": _training_rate(self.training_log, "repaired"),
            "invalid_action_cap_hits": sum(1 for item in self.training_log if item.get("done_reason") == "invalid_action_cap_reached"),
            "failed_parse_examples": _failed_parse_examples(self.training_log),
            "prompt_length_stats": _prompt_length_stats(self.training_log),
            "completion_length_stats": _completion_length_stats(self.training_log),
            "skipped_nonfinite_updates": self.skipped_nonfinite_updates,
            "duration_seconds": duration_seconds,
            "training_log": self.training_log,
            "loss_history": self.loss_history,
            "reward_history": self.reward_history,
            "valid_json_rate": _training_rate(self.training_log, "valid_json"),
            "valid_action_rate": _training_rate(self.training_log, "valid_action_shape"),
            "invalid_action_count": sum(int(item.get("invalid_action_count", 0)) for item in self.training_log),
            "completed_deliveries": _aggregate_metric(eval_after, "completed_deliveries"),
            "urgent_successes": _aggregate_metric(eval_after, "urgent_successes"),
            "safety_violations": _aggregate_metric(eval_after, "safety_violations"),
            "warnings": _training_warnings(self.reward_history, self.training_log),
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
            "candidate_choice_default": self.training_candidate_choice,
            "dependency_status": self.dependencies,
            "device": self._device_summary(),
            "generation_safety": self.generation_safety_flags,
            "sampling_model": {
                "enabled": self.use_safe_sampling,
                "dtype": self.sampling_dtype_name,
            },
            "output_dir": str(self.output_dir),
        }
        destination = write_json(self.output_dir / "sanity_check.json", payload)
        return destination

    def choose_action(self, observation: dict[str, Any], *, do_sample: bool) -> dict[str, Any]:
        candidate_actions = build_candidate_actions(observation)
        prompt = build_action_prompt(observation, candidate_choice=self.training_candidate_choice)
        text, _, _ = self._generate_completion(prompt, do_sample=do_sample)
        return safe_parse_action(text, observation=observation, candidate_actions=candidate_actions)

    def _run_training_episode(self, task_id: str, episode_number: int) -> dict[str, Any]:
        env = DroneZEnvironment(default_task_id=task_id, max_episode_actions=self.args.max_actions)
        observation, info = env.reset(task_id)

        episode_return = 0.0
        episode_loss_total = 0.0
        update_steps = 0
        invalid_action_count = 0
        done = False
        no_learning_signal = False
        candidates: list[dict[str, Any]] = []

        while not done:
            candidate_actions = build_candidate_actions(observation)
            prompt = build_action_prompt(observation, candidate_choice=self.training_candidate_choice)
            candidates = []
            prompt_preview = prompt[:PROMPT_PREVIEW_CHARS]
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            for _ in range(self.args.group_size):
                raw_text, prompt_ids, completion_ids = self._generate_completion(prompt, do_sample=True)
                parse_result = parse_llm_action(raw_text, observation=observation, candidate_actions=candidate_actions)
                action = parse_result.action
                estimate = self._evaluate_candidate(env, action)
                candidates.append(
                    {
                        "raw_text": raw_text,
                        "action": action,
                        "parse": {
                            "valid_json": parse_result.valid_json,
                            "valid_action_shape": parse_result.valid_action_shape,
                            "repaired": parse_result.repaired,
                            "used_candidate_choice": parse_result.used_candidate_choice,
                            "error_code": parse_result.error_code,
                            "notes": parse_result.notes,
                        },
                        "prompt_hash": prompt_hash,
                        "prompt_preview": prompt_preview,
                        "prompt_char_length": len(prompt),
                        "candidate_count": len(candidate_actions),
                        "prompt_token_length": int(prompt_ids.shape[0]),
                        "completion_token_length": int(completion_ids.shape[0]),
                        "generation": {
                            "temperature": self.latest_generation_temperature,
                            "top_p": self.latest_generation_top_p,
                            "max_new_tokens": self.latest_generation_max_new_tokens,
                            "stop_reason": self.last_generation_stop_reason,
                            "do_sample": self.latest_generation_do_sample,
                        },
                        "prompt_ids": prompt_ids,
                        "completion_ids": completion_ids,
                        **estimate,
                    }
                )
            self.last_parser_error_counts = _error_code_counts(candidates)
            self.last_failed_examples = _candidate_failure_examples(candidates)
            self.last_candidate_choice_rate = _candidate_flag_rate(candidates, "used_candidate_choice")
            self.last_repair_rate = _candidate_flag_rate(candidates, "repaired")
            self.last_prompt_preview = prompt_preview
            self.last_prompt_hash = prompt_hash
            self.last_prompt_char_length = len(prompt)
            self.last_candidate_count = len(candidate_actions)
            self.last_prompt_token_length = max((candidate.get("prompt_token_length", 0) for candidate in candidates), default=0)
            self.last_completion_token_length = max((candidate.get("completion_token_length", 0) for candidate in candidates), default=0)

            rewards = self._torch.tensor(
                [candidate["estimated_total_return"] for candidate in candidates],
                dtype=self._torch.float32,
                device=self.device,
            )
            if rewards.numel() > 1:
                advantages = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
            else:
                advantages = self._torch.zeros_like(rewards)
            no_learning_signal = bool(rewards.numel() > 1 and float(rewards.std(unbiased=False).detach().cpu()) < 1e-6)

            self.optimizer.zero_grad(set_to_none=True)
            losses = []
            for advantage, candidate in zip(advantages, candidates):
                logprob = self._mean_logprob_of_completion(candidate["prompt_ids"], candidate["completion_ids"])
                losses.append(-advantage.detach() * logprob)
            loss = self._torch.stack(losses).mean()
            should_step = True
            if self.hf_safe_generation and not bool(self._torch.isfinite(loss).all().item()):
                should_step = False
                self.skipped_nonfinite_updates += 1
                self.optimizer.zero_grad(set_to_none=True)
            else:
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.hf_safe_generation and self._has_nonfinite_gradients():
                    should_step = False
                    self.skipped_nonfinite_updates += 1
                    self.optimizer.zero_grad(set_to_none=True)
            if should_step:
                self.optimizer.step()

            best_candidate = candidates[int(self._torch.argmax(rewards).item())]
            observation, reward, done, info = env.step(best_candidate["action"])
            episode_return += float(reward)
            invalid_action_count = int(info.get("invalid_action_count", invalid_action_count))
            self.last_invalid_action_cap_hits = 1 if info.get("done_reason") == "invalid_action_cap_reached" else 0
            update_steps += 1
            loss_value = float(loss.detach().cpu())
            episode_loss_total += loss_value
            self.loss_history.append(round(loss_value, 6))

        self.reward_history.append(round(episode_return, 4))
        current_samples = [
            {
                "valid_json": candidate["parse"]["valid_json"],
                "valid_action_shape": candidate["parse"]["valid_action_shape"],
                "repaired": candidate["parse"]["repaired"],
                "used_candidate_choice": candidate["parse"]["used_candidate_choice"],
                "error_code": candidate["parse"]["error_code"],
                "notes": candidate["parse"]["notes"],
                "raw_text": candidate["raw_text"],
                "prompt_hash": candidate["prompt_hash"],
                "prompt_preview": candidate["prompt_preview"],
                "prompt_char_length": candidate["prompt_char_length"],
                "prompt_token_length": candidate["prompt_token_length"],
                "completion_token_length": candidate["completion_token_length"],
                "candidate_count": candidate["candidate_count"],
                "generation": candidate["generation"],
                "invalid_action": candidate["invalid_action"],
                "estimated_total_return": round(float(candidate["estimated_total_return"]), 4),
            }
            for candidate in candidates
        ][: self.args.group_size]
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
            "candidate_choice": self.training_candidate_choice,
            "parser_error_code_counts": self.last_parser_error_counts,
            "candidate_choice_rate": self.last_candidate_choice_rate,
            "repair_rate": self.last_repair_rate,
            "failed_parse_examples": self.last_failed_examples,
            "prompt_preview": self.last_prompt_preview,
            "prompt_hash": self.last_prompt_hash,
            "prompt_char_length": self.last_prompt_char_length,
            "candidate_count": self.last_candidate_count,
            "prompt_token_length": self.last_prompt_token_length,
            "completion_token_length": self.last_completion_token_length,
            "invalid_action_cap_hits": self.last_invalid_action_cap_hits,
            "no_learning_signal_warning": no_learning_signal,
            "sample_parse_results": current_samples,
            "valid_json_rate": _rate(sample["valid_json"] for sample in current_samples),
            "valid_action_rate": _rate(sample["valid_action_shape"] for sample in current_samples),
        }

    def write_format_check(self) -> Path:
        rows: list[dict[str, Any]] = []
        for task_id in self.tasks:
            env = DroneZEnvironment(default_task_id=task_id, max_episode_actions=self.args.max_actions)
            observation, _ = env.reset(task_id)
            candidate_actions = build_candidate_actions(observation)
            examples = _format_check_examples(candidate_actions)
            for label, text in examples:
                result = parse_llm_action(text, observation=observation, candidate_actions=candidate_actions)
                rows.append(
                    {
                        "task_id": task_id,
                        "label": label,
                        "raw_text": text,
                        "parsed_action": result.action,
                        "valid_json": result.valid_json,
                        "valid_action_shape": result.valid_action_shape,
                        "repaired": result.repaired,
                        "used_candidate_choice": result.used_candidate_choice,
                        "error_code": result.error_code,
                        "notes": result.notes,
                    }
                )

        model_sampling = {
            "executed": False,
            "reason": "Pass --sample-model-actions on a CUDA machine to sample the selected model without training.",
            "samples": [],
        }
        if self.args.sample_model_actions:
            try:
                self._validate_training_environment()
                self._seed_everything()
                self._load_model()
                model_sampling = self._sample_model_actions_for_format_check()
            except RuntimeError as exc:
                model_sampling = {"executed": False, "reason": str(exc), "samples": []}

        valid_json_rate = _rate(row["valid_json"] for row in rows)
        valid_action_rate = _rate(row["valid_action_shape"] for row in rows)
        invalid_reasons: dict[str, int] = {}
        for row in rows:
            reason = row["error_code"] or "valid"
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
        payload = {
            "mode": "format-check",
            "training_executed": False,
            "note": (
                "Validates DroneZ action JSON extraction, repair, and candidate-choice scaffolding. "
                "This is not a model improvement claim."
            ),
            "selected_model": self.args.model,
            "curriculum": self.tasks,
            "candidate_choice_supported": True,
            "valid_json_rate": valid_json_rate,
            "valid_action_rate": valid_action_rate,
            "top_invalid_reasons": dict(sorted(invalid_reasons.items(), key=lambda item: (-item[1], item[0]))),
            "model_sampling": model_sampling,
            "rows": rows,
        }
        return write_json(self.output_dir / "format_check.json", payload)

    def _sample_model_actions_for_format_check(self) -> dict[str, Any]:
        self.model.eval()
        samples = []
        try:
            for task_id in self.tasks[:2]:
                env = DroneZEnvironment(default_task_id=task_id, max_episode_actions=self.args.max_actions)
                observation, _ = env.reset(task_id)
                candidate_actions = build_candidate_actions(observation)
                prompt = build_action_prompt(observation, candidate_choice=self.training_candidate_choice)
                raw_text, _, _ = self._generate_completion(prompt, do_sample=False)
                result = parse_llm_action(raw_text, observation=observation, candidate_actions=candidate_actions)
                samples.append(
                    {
                        "task_id": task_id,
                        "raw_text": raw_text,
                        "parsed_action": result.action,
                        "valid_json": result.valid_json,
                        "valid_action_shape": result.valid_action_shape,
                        "repaired": result.repaired,
                        "used_candidate_choice": result.used_candidate_choice,
                        "error_code": result.error_code,
                        "notes": result.notes,
                    }
                )
        finally:
            self.model.train()
        return {
            "executed": True,
            "reason": "Sampled selected model without optimizer updates.",
            "samples": samples,
            "valid_json_rate": _rate(sample["valid_json"] for sample in samples),
            "valid_action_rate": _rate(sample["valid_action_shape"] for sample in samples),
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
        generator_model = self.sampling_model if do_sample and self.use_safe_sampling else self.model
        generator_tokenizer = self.sampling_tokenizer if do_sample and self.use_safe_sampling else self.tokenizer
        generator_device = self.sampling_device if do_sample and self.use_safe_sampling else self.device
        generator_kind = "sampling" if do_sample and self.use_safe_sampling else "training"
        generator_dtype_name = self.sampling_dtype_name if do_sample and self.use_safe_sampling else self.dtype_name

        inputs = generator_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.args.max_prompt_tokens,
        )
        input_ids = inputs["input_ids"].to(generator_device)
        attention_mask = inputs["attention_mask"].to(generator_device)

        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_sample": do_sample,
            "max_new_tokens": self.training_generation_max_new_tokens if do_sample else self.args.max_new_tokens,
            "pad_token_id": generator_tokenizer.pad_token_id,
            "eos_token_id": generator_tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = self.training_generation_temperature
            generation_kwargs["top_p"] = self.training_generation_top_p
            generation_kwargs["top_k"] = SAFE_TOP_K
            generation_kwargs["repetition_penalty"] = SAFE_REPETITION_PENALTY
            if self.remove_invalid_values_supported:
                generation_kwargs["remove_invalid_values"] = True
            if self.renormalize_logits_supported:
                generation_kwargs["renormalize_logits"] = True
            if self.hf_safe_generation:
                generation_kwargs["top_k"] = SAFE_TOP_K
                generation_kwargs["repetition_penalty"] = SAFE_REPETITION_PENALTY
                if self.remove_invalid_values_supported:
                    generation_kwargs["remove_invalid_values"] = True
                if self.renormalize_logits_supported:
                    generation_kwargs["renormalize_logits"] = True
            self.last_generation_stop_reason = "max_new_tokens"
        else:
            self.last_generation_stop_reason = "eos_or_length"

        self.latest_generation_kwargs = {
            key: value for key, value in generation_kwargs.items() if key not in {"input_ids", "attention_mask"}
        }
        self.latest_generation_dtype_name = generator_dtype_name
        self.latest_generation_device_type = getattr(generator_device, "type", str(generator_device))
        self.latest_generation_used_sampling_model = bool(do_sample and self.use_safe_sampling)
        self.latest_generation_support_flags = {
            "remove_invalid_values": self.remove_invalid_values_supported,
            "renormalize_logits": self.renormalize_logits_supported,
        }
        self.latest_generation_hf_safe = self.hf_safe_generation
        self.latest_generation_do_sample = do_sample
        self.latest_generation_temperature = generation_kwargs.get("temperature")
        self.latest_generation_top_p = generation_kwargs.get("top_p")
        self.latest_generation_top_k = generation_kwargs.get("top_k")
        self.latest_generation_repetition_penalty = generation_kwargs.get("repetition_penalty")
        self.latest_generation_removed_invalid_values = bool(generation_kwargs.get("remove_invalid_values", False))
        self.latest_generation_renormalize_logits = bool(generation_kwargs.get("renormalize_logits", False))
        self.latest_generation_model_kind = generator_kind
        self.latest_generation_transformers_version = self.detected_transformers_version
        self.latest_generation_use_cache = bool(getattr(generator_model.config, "use_cache", False))
        self.latest_generation_pad_token_id = generation_kwargs["pad_token_id"]
        self.latest_generation_eos_token_id = generation_kwargs["eos_token_id"]
        self.latest_generation_max_new_tokens = generation_kwargs["max_new_tokens"]
        self.latest_generation_device_capability_major = self.device_capability_major
        self.latest_generation_device_name = self.device_name
        self.latest_generation_sampling_device_name = self.sampling_device.type if self.sampling_device is not None else None
        self.latest_generation_sampling_dtype_name = self.sampling_dtype_name
        self.latest_generation_input_length = int(input_ids.shape[1])

        with self._torch.no_grad():
            self.last_prompt_token_length = int(input_ids.shape[1])
            self.last_prompt_preview = prompt[:PROMPT_PREVIEW_CHARS]
            self.last_prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
            self.last_prompt_char_length = len(prompt)
            outputs = generator_model.generate(**generation_kwargs)

        prompt_length = input_ids.shape[1]
        completion_ids = outputs[0, prompt_length:].detach().cpu()
        self.latest_generation_completion_length = int(completion_ids.shape[0])
        self.last_completion_token_length = self.latest_generation_completion_length
        if completion_ids.numel() == 0:
            completion_ids = self._torch.tensor([generator_tokenizer.eos_token_id], dtype=self._torch.long)
        text = generator_tokenizer.decode(completion_ids, skip_special_tokens=True)
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
        if self.hf_safe_generation:
            completion_logits = completion_logits.float()
        token_logprobs = self._torch.log_softmax(completion_logits, dim=-1).gather(
            -1, completion_targets.unsqueeze(-1)
        ).squeeze(-1)
        return token_logprobs.mean()

    def _has_nonfinite_gradients(self) -> bool:
        for parameter in self.model.parameters():
            grad = getattr(parameter, "grad", None)
            if grad is None:
                continue
            if not bool(self._torch.isfinite(grad).all().item()):
                return True
        return False

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
        self.detected_transformers_version = getattr(transformers, "__version__", None)
        generation_signature = inspect.signature(transformers.GenerationConfig.__init__).parameters
        self.remove_invalid_values_supported = "remove_invalid_values" in generation_signature
        self.renormalize_logits_supported = "renormalize_logits" in generation_signature
        self.device = torch.device("cuda")
        self.sampling_device = self.device
        self.device_capability_major = torch.cuda.get_device_capability()[0]
        self.device_name = torch.cuda.get_device_name(0)
        dtype = torch.bfloat16 if self.device_capability_major >= 8 else torch.float16
        self.dtype_name = "bfloat16" if dtype == torch.bfloat16 else "float16"
        self.training_dtype_name = self.dtype_name
        self.sampling_dtype_name = self.dtype_name

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.args.model, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.sampling_tokenizer = self.tokenizer

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.args.model,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.sampling_model = self.model
        self.model.train()
        self.model.config.use_cache = False
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.generation_safety_flags = {
            "hf_safe_generation": self.hf_safe_generation,
            "use_safe_sampling": self.use_safe_sampling,
            "remove_invalid_values": self.remove_invalid_values_supported,
            "renormalize_logits": self.renormalize_logits_supported,
            "training_dtype": self.training_dtype_name,
            "sampling_dtype": self.sampling_dtype_name,
            "dedicated_sampling_model": False,
        }
        self.latest_generation_support_flags = {
            "remove_invalid_values": self.remove_invalid_values_supported,
            "renormalize_logits": self.renormalize_logits_supported,
        }
        self.latest_generation_sampling_dtype_name = self.sampling_dtype_name
        self.latest_generation_sampling_device_name = self.sampling_device.type if self.sampling_device is not None else None
        self.latest_generation_device_capability_major = self.device_capability_major
        self.latest_generation_device_name = self.device_name
        self.latest_generation_hf_safe = self.hf_safe_generation
        self.latest_generation_transformers_version = self.detected_transformers_version
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
    parser.add_argument("--action-temperature", type=float, default=ACTION_GENERATION_TEMPERATURE)
    parser.add_argument("--action-top-p", type=float, default=ACTION_GENERATION_TOP_P)
    parser.add_argument("--action-max-new-tokens", type=int, default=ACTION_GENERATION_MAX_NEW_TOKENS)
    parser.add_argument("--disable-training-candidate-choice", action="store_true", help="Disable candidate-choice prompting for the local GRPO training loop only.")
    parser.add_argument("--max-continuation-steps", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--save-model-dir", default="")
    parser.add_argument("--sanity-check", action="store_true", help="Validate dependencies/GPU reporting without running training.")
    parser.add_argument("--format-check", action="store_true", help="Validate action JSON parsing, repair, and candidate-choice scaffolding.")
    parser.add_argument("--candidate-choice", action="store_true", default=None, help="Ask the model to choose among generated valid candidate actions during training.")
    parser.add_argument("--sample-model-actions", action="store_true", help="During --format-check, sample the selected model if CUDA/dependencies are available.")
    parser.add_argument("--warmstart-data", action="store_true", help="Generate ImprovedPolicy SFT action-format data before GRPO.")
    parser.add_argument("--warmstart-output", default="", help="Optional path for --warmstart-data JSONL output.")
    parser.add_argument("--real-train", action="store_true", help="Explicit marker for the actual online local training path.")
    return parser


def extended_dependency_status() -> dict[str, bool]:
    status = dependency_status()
    for module_name in ("torch", "peft"):
        status[module_name] = importlib.util.find_spec(module_name) is not None
    return status


def _is_hf_job_environment() -> bool:
    return any(bool(os.environ.get(name)) for name in HF_JOB_ENV_VARS)


def safe_parse_action(
    text: str,
    *,
    observation: dict[str, Any] | None = None,
    candidate_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return parse_llm_action(text, observation=observation, candidate_actions=candidate_actions).action


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


def _aggregate_metric(payload: dict[str, Any], metric: str) -> float:
    aggregate = payload.get("aggregate", {})
    if not aggregate:
        return 0.0
    first_value = next(iter(aggregate.values()))
    return float(first_value.get(metric, 0.0))


def _training_rate(training_log: list[dict[str, Any]], key: str) -> float:
    values = [
        bool(sample.get(key))
        for episode in training_log
        for sample in episode.get("sample_parse_results", [])
    ]
    return _rate(values)


def _error_code_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        error_code = candidate.get("parse", {}).get("error_code") or "valid"
        counts[error_code] = counts.get(error_code, 0) + 1
    return counts


def _candidate_failure_examples(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for candidate in candidates:
        parse_info = candidate.get("parse", {})
        if parse_info.get("valid_action_shape"):
            continue
        examples.append(
            {
                "error_code": parse_info.get("error_code"),
                "notes": parse_info.get("notes", []),
                "raw_text": candidate.get("raw_text"),
                "prompt_hash": candidate.get("prompt_hash"),
                "prompt_preview": candidate.get("prompt_preview"),
            }
        )
        if len(examples) >= FAILED_EXAMPLE_LIMIT:
            break
    return examples


def _candidate_flag_rate(candidates: list[dict[str, Any]], key: str) -> float:
    return _rate(candidate.get("parse", {}).get(key) for candidate in candidates)


def _parser_error_code_counts(training_log: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for episode in training_log:
        for error_code, value in episode.get("parser_error_code_counts", {}).items():
            counts[error_code] = counts.get(error_code, 0) + int(value)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _failed_parse_examples(training_log: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for episode in training_log:
        for example in episode.get("failed_parse_examples", []):
            examples.append({"episode": episode.get("episode"), **example})
            if len(examples) >= FAILED_EXAMPLE_LIMIT:
                return examples
    return examples


def _length_stats(training_log: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [int(episode.get(key, 0) or 0) for episode in training_log if episode.get(key) is not None]
    if not values:
        return {"min": 0, "max": 0, "mean": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values)),
    }


def _prompt_length_stats(training_log: list[dict[str, Any]]) -> dict[str, int]:
    return _length_stats(training_log, "prompt_char_length")


def _completion_length_stats(training_log: list[dict[str, Any]]) -> dict[str, int]:
    return _length_stats(training_log, "completion_token_length")


def _training_warnings(reward_history: list[float], training_log: list[dict[str, Any]]) -> list[str]:
    warnings = []
    if reward_history and len(set(reward_history)) <= 1:
        warnings.append("No learning signal: all rollout rewards identical.")
    if any(item.get("done_reason") == "invalid_action_cap_reached" for item in training_log):
        warnings.append("At least one training episode hit invalid_action_cap_reached.")
    if any(item.get("no_learning_signal_warning") for item in training_log):
        warnings.append("At least one GRPO group had identical candidate returns; use --candidate-choice or warm-start data.")
    return warnings


def _format_check_examples(candidate_actions: list[dict[str, Any]]) -> list[tuple[str, str]]:
    first = candidate_actions[0] if candidate_actions else {"action": "prioritize_order", "params": {"order_id": "O1"}}
    action_name = first["action"]
    return [
        ("plain_json", json.dumps(first)),
        ("markdown_fence", f"```json\n{json.dumps(first)}\n```"),
        ("single_quotes", str(first)),
        ("action_name_alias", json.dumps({"action_name": action_name, "params": first["params"]})),
        ("nested_action", json.dumps({"action": first})),
        ("candidate_choice", json.dumps({"choice": 1})),
        ("extra_text", f"I would do this now: {json.dumps(first)} because it is safe."),
        ("invalid_free_text", "send the nearest drone immediately"),
    ]


def _rate(values) -> float:
    items = list(values)
    return round(sum(1 for item in items if item) / len(items), 4) if items else 0.0


def _parse_csv_list(raw: str, *, fallback: list[str]) -> list[str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or fallback


def write_blocked_training_artifacts(args: argparse.Namespace, reason: str) -> dict[str, Path]:
    output_dir = Path(args.output_dir)
    trainer = LocalGRPOTrainer(args)
    payload = {
        "mode": "local-grpo",
        "status": "blocked",
        "training_executed": False,
        "note": (
            "A real candidate-choice GRPO command was requested, but the run stopped before "
            "model loading or optimizer updates. This is an honest blocked-run artifact, not "
            "a training success claim."
        ),
        "reason": reason,
        "selected_model": args.model,
        "candidate_choice": args.candidate_choice,
        "curriculum": trainer.tasks,
        "eval_tasks": trainer.eval_tasks,
        "dependency_status": trainer.dependencies,
        "device": trainer._device_summary(),
        "hyperparameters": {
            "seed": args.seed,
            "episodes": args.episodes,
            "group_size": args.group_size,
            "learning_rate": args.learning_rate,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_tokens": args.max_prompt_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_continuation_steps": args.max_continuation_steps,
            "max_actions": args.max_actions,
        },
        "reward_improved": None,
        "mean_reward_delta": None,
        "warnings": [
            "No optimizer updates were run.",
            "No eval_before/eval_after reward comparison exists for this blocked run.",
            "Run the same command on a CUDA GPU machine to produce real training evidence.",
        ],
    }
    eval_placeholder = {
        "status": "not_run",
        "training_executed": False,
        "reason": reason,
        "note": "Evaluation was not run because candidate-choice GRPO did not start.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "training_metrics": write_json(output_dir / "training_metrics.json", payload),
        "eval_before": write_json(output_dir / "eval_before.json", eval_placeholder),
        "eval_after": write_json(
            output_dir / "eval_after.json",
            {**eval_placeholder, "mean_reward_delta": None},
        ),
    }


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

        if args.format_check:
            destination = trainer.write_format_check()
            print(f"Wrote {destination}")
            print(destination.read_text())
            return 0

        if args.warmstart_data:
            from generate_sft_action_data import generate_examples, write_jsonl

            destination = Path(args.warmstart_output) if args.warmstart_output else Path(args.output_dir) / "sft_action_data.jsonl"
            rows = generate_examples(trainer.tasks, max_actions=args.max_actions)
            write_jsonl(destination, rows)
            print(json.dumps({"output": str(destination), "examples": len(rows), "tasks": trainer.tasks}, indent=2))
            return 0

        result_paths = trainer.run()
        print(f"Wrote {result_paths['training_metrics']}")
        print(f"Wrote {result_paths['eval_before']}")
        print(f"Wrote {result_paths['eval_after']}")
        return 0
    except RuntimeError as exc:
        if args.real_train:
            result_paths = write_blocked_training_artifacts(args, str(exc))
            print(f"Wrote blocked run artifact {result_paths['training_metrics']}", file=sys.stderr)
            print(f"Wrote blocked run artifact {result_paths['eval_before']}", file=sys.stderr)
            print(f"Wrote blocked run artifact {result_paths['eval_after']}", file=sys.stderr)
        print(f"Local GRPO training is not ready: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
