from __future__ import annotations

import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

from scripts.train_grpo_local import LocalGRPOTrainer, SAFE_REPETITION_PENALTY, SAFE_TOP_K

ROOT = Path(__file__).resolve().parents[1]


class FakeTensor:
    def __init__(self, shape=(1, 3), *, finite: bool = True) -> None:
        self.shape = shape
        self.finite = finite

    def to(self, device):
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def numel(self):
        if len(self.shape) == 1:
            return self.shape[0]
        return self.shape[-1]

    def __getitem__(self, key):
        if isinstance(key, int) and len(self.shape) > 1:
            return FakeTensor((self.shape[1],), finite=self.finite)
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], slice):
            start = key[1].start or 0
            stop = key[1].stop if key[1].stop is not None else self.shape[-1]
            return FakeTensor((max(stop - start, 0),), finite=self.finite)
        return self


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTokenizer:
    def __init__(self, *, shape=(1, 3)) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 2
        self._shape = shape

    def __call__(self, prompt, **kwargs):
        return {
            "input_ids": FakeTensor(self._shape),
            "attention_mask": FakeTensor(self._shape),
        }

    def decode(self, ids, skip_special_tokens=True):
        return '{"action":"prioritize_order","params":{"order_id":"O1"}}'


class FakeTorch:
    float32 = "float32"
    long = "long"

    @staticmethod
    def no_grad():
        return FakeNoGrad()

    @staticmethod
    def tensor(values, dtype=None):
        return FakeTensor((1, len(values)))

    @staticmethod
    def isfinite(value):
        return SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: value.finite))


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(use_cache=False)
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        return FakeTensor((1, 5))

    def parameters(self):
        return []


def _run(command: list[str], *, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT / 'scripts'}:."
    if extra_env:
        env.update(extra_env)
    return subprocess.run(command, cwd=ROOT, env=env, check=False, capture_output=True, text=True)


def _build_args(**overrides) -> Namespace:
    base = {
        "output_dir": "artifacts/training/test-hf",
        "tasks": "easy,medium,demo",
        "eval_tasks": "easy,medium,demo,hard",
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "seed": 7,
        "episodes": 1,
        "group_size": 2,
        "learning_rate": 1e-5,
        "max_new_tokens": 32,
        "max_prompt_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "action_temperature": 0.35,
        "action_top_p": 0.9,
        "action_max_new_tokens": 48,
        "max_continuation_steps": 2,
        "max_actions": 32,
        "candidate_choice": None,
        "disable_training_candidate_choice": False,
        "save_model_dir": None,
        "sanity_check": False,
        "format_check": False,
    }
    base.update(overrides)
    return Namespace(**base)


def _build_generation_trainer(*, hf_safe: bool) -> LocalGRPOTrainer:
    trainer = LocalGRPOTrainer(_build_args())
    trainer.hf_safe_generation = hf_safe
    trainer.use_safe_sampling = hf_safe
    trainer.remove_invalid_values_supported = True
    trainer.renormalize_logits_supported = True
    trainer.device = SimpleNamespace(type="cuda")
    trainer.sampling_device = trainer.device
    trainer.dtype_name = "float16"
    trainer.sampling_dtype_name = "float16"
    trainer.detected_transformers_version = "test"
    trainer.device_capability_major = 7
    trainer.device_name = "NVIDIA T4"
    trainer.tokenizer = FakeTokenizer()
    trainer.sampling_tokenizer = trainer.tokenizer
    trainer._torch = FakeTorch()
    trainer.model = FakeModel()
    trainer.sampling_model = trainer.model
    return trainer


def test_sanity_check_writes_honest_metadata(tmp_path: Path) -> None:
    output_dir = tmp_path / "local_sanity"
    result = _run([sys.executable, "scripts/train_grpo_local.py", "--sanity-check", "--output-dir", str(output_dir)])

    assert result.returncode == 0
    payload = json.loads((output_dir / "sanity_check.json").read_text())
    assert payload["mode"] == "local-grpo-sanity"
    assert payload["training_executed"] is False
    assert payload["selected_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert payload["curriculum"] == ["easy", "medium", "demo"]
    assert payload["candidate_choice_default"] is True
    assert "gpu_available" in payload["device"]


def test_training_fails_honestly_without_gpu(tmp_path: Path) -> None:
    output_dir = tmp_path / "local_train"
    result = _run([sys.executable, "scripts/train_grpo_local.py", "--output-dir", str(output_dir), "--episodes", "1"])

    assert result.returncode == 2
    assert "Local GRPO training is not ready:" in result.stderr
    assert not (output_dir / "training_metrics.json").exists()


def test_format_check_writes_valid_action_rates(tmp_path: Path) -> None:
    output_dir = tmp_path / "format_check"
    result = _run(
        [
            sys.executable,
            "scripts/train_grpo_local.py",
            "--format-check",
            "--tasks",
            "easy,demo",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert result.returncode == 0
    payload = json.loads((output_dir / "format_check.json").read_text())
    assert payload["training_executed"] is False
    assert payload["valid_json_rate"] >= 0.75
    assert payload["valid_action_rate"] >= 0.75


def test_hf_job_detection_uses_env_toggle(monkeypatch) -> None:
    monkeypatch.delenv("HF_JOB_ID", raising=False)
    monkeypatch.delenv("HF_SPACE_ID", raising=False)
    monkeypatch.delenv("SPACE_ID", raising=False)
    trainer = LocalGRPOTrainer(_build_args())
    assert trainer.hf_safe_generation is False
    assert trainer.use_safe_sampling is False

    monkeypatch.setenv("HF_JOB_ID", "job-123")
    hf_trainer = LocalGRPOTrainer(_build_args())
    assert hf_trainer.hf_safe_generation is True
    assert hf_trainer.use_safe_sampling is True


def test_hf_safe_generation_adds_sampling_guards() -> None:
    trainer = _build_generation_trainer(hf_safe=True)

    trainer._generate_completion("prompt", do_sample=True)

    assert trainer.latest_generation_hf_safe is True
    assert trainer.latest_generation_used_sampling_model is True
    assert trainer.latest_generation_model_kind == "sampling"
    assert trainer.latest_generation_top_k == SAFE_TOP_K
    assert trainer.latest_generation_repetition_penalty == SAFE_REPETITION_PENALTY
    assert trainer.latest_generation_removed_invalid_values is True
    assert trainer.latest_generation_renormalize_logits is True
    assert trainer.latest_generation_kwargs["top_k"] == SAFE_TOP_K
    assert trainer.latest_generation_kwargs["repetition_penalty"] == SAFE_REPETITION_PENALTY
    assert trainer.latest_generation_kwargs["remove_invalid_values"] is True
    assert trainer.latest_generation_kwargs["renormalize_logits"] is True
    assert trainer.latest_generation_kwargs["temperature"] == 0.35
    assert trainer.latest_generation_kwargs["top_p"] == 0.9
    assert trainer.latest_generation_kwargs["max_new_tokens"] == 32
    assert trainer.latest_generation_completion_length == 2
    assert trainer.last_generation_stop_reason == "max_new_tokens"
    assert trainer.last_prompt_hash is not None
    assert trainer.last_prompt_preview == "prompt"
    assert trainer.training_candidate_choice is True
    assert trainer.training_generation_temperature == 0.35
    assert trainer.training_generation_top_p == 0.9
    assert trainer.training_generation_max_new_tokens == 32


def test_disable_training_candidate_choice_turns_off_training_default() -> None:
    trainer = LocalGRPOTrainer(_build_args(disable_training_candidate_choice=True))
    assert trainer.training_candidate_choice is False


def test_non_hf_training_still_uses_tighter_sampling_settings() -> None:
    trainer = _build_generation_trainer(hf_safe=False)

    trainer._generate_completion("prompt", do_sample=True)

    assert trainer.latest_generation_kwargs["temperature"] == 0.35
    assert trainer.latest_generation_kwargs["top_p"] == 0.9
    assert trainer.latest_generation_kwargs["top_k"] == SAFE_TOP_K
    assert trainer.latest_generation_kwargs["repetition_penalty"] == SAFE_REPETITION_PENALTY
    assert trainer.latest_generation_used_sampling_model is False
    assert trainer.last_generation_stop_reason == "max_new_tokens"


class _FakeShape:
    def __init__(self, value: int) -> None:
        self._value = value

    def __getitem__(self, index: int) -> int:
        return self._value


class _FakeIds:
    def __init__(self, length: int) -> None:
        self.shape = _FakeShape(length)

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self



def test_failed_parse_metadata_is_persisted() -> None:
    trainer = LocalGRPOTrainer(_build_args())
    trainer.last_parser_error_counts = {"no_json_object": 2}
    trainer.last_failed_examples = [
        {
            "error_code": "no_json_object",
            "notes": ["empty_output"],
            "raw_text": "not json",
            "prompt_hash": "abc123",
            "prompt_preview": "PROMPT",
        }
    ]
    trainer.last_candidate_choice_rate = 0.5
    trainer.last_repair_rate = 0.0
    trainer.last_invalid_action_cap_hits = 1
    trainer.last_prompt_preview = "PROMPT"
    trainer.last_prompt_hash = "abc123"
    trainer.last_prompt_char_length = 99
    trainer.last_candidate_count = 3
    trainer.last_prompt_token_length = 77
    trainer.last_completion_token_length = 13
    candidate = {
        "parse": {
            "valid_json": False,
            "valid_action_shape": False,
            "repaired": False,
            "used_candidate_choice": False,
            "error_code": "no_json_object",
            "notes": ["empty_output"],
        },
        "raw_text": "not json",
        "prompt_hash": "abc123",
        "prompt_preview": "PROMPT",
        "prompt_char_length": 99,
        "prompt_token_length": 77,
        "completion_token_length": 13,
        "candidate_count": 3,
        "generation": {"temperature": 0.35, "top_p": 0.9, "max_new_tokens": 32, "stop_reason": "max_new_tokens", "do_sample": True},
        "invalid_action": True,
        "estimated_total_return": -90.0,
    }
    trainer.reward_history = []
    trainer.loss_history = []
    trainer.args.group_size = 1

    trainer._run_training_episode = lambda *args, **kwargs: None
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
    ]

    assert current_samples[0]["raw_text"] == "not json"
    assert current_samples[0]["prompt_hash"] == "abc123"
    assert current_samples[0]["prompt_preview"] == "PROMPT"
    assert current_samples[0]["prompt_token_length"] == 77
    assert current_samples[0]["completion_token_length"] == 13
    assert trainer.last_failed_examples[0]["raw_text"] == "not json"
    assert trainer.last_parser_error_counts["no_json_object"] == 2
    assert trainer.last_candidate_choice_rate == 0.5
    assert trainer.last_invalid_action_cap_hits == 1


def test_training_metrics_include_parser_debug_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_JOB_ID", "job-debug-fields")
    output_dir = tmp_path / "hf_debug"
    trainer = LocalGRPOTrainer(_build_args(output_dir=str(output_dir)))
    trainer.dependencies = {"torch": True, "transformers": True}
    trainer._validate_training_environment = lambda: None
    trainer._seed_everything = lambda: None
    trainer._load_model = lambda: None
    trainer._device_summary = lambda: {"gpu_available": True, "dtype": "float16"}
    trainer._evaluate_model = lambda policy_id: {
        "aggregate": {"overall": {"mean_total_reward": 0.0, "completed_deliveries": 0, "urgent_successes": 0, "safety_violations": 1}},
        "policy_results": {policy_id: {"completed_deliveries": 0, "urgent_successes": 0, "safety_violations": 1}},
        "tasks": [],
        "ranking": [],
    }
    trainer._run_training_episode = lambda task_id, episode_number: {
        "episode": episode_number,
        "task_id": task_id,
        "episode_return": -90.0,
        "mean_loss": 0.0,
        "update_steps": 1,
        "invalid_action_count": 4,
        "done_reason": "invalid_action_cap_reached",
        "terminated_by": "invalid_action_cap",
        "final_cumulative_reward": {},
        "candidate_choice": True,
        "parser_error_code_counts": {"no_json_object": 2},
        "candidate_choice_rate": 0.0,
        "repair_rate": 0.0,
        "failed_parse_examples": [{"error_code": "no_json_object", "raw_text": "plain text", "prompt_hash": "hash1", "prompt_preview": "PROMPT"}],
        "prompt_preview": "PROMPT",
        "prompt_hash": "hash1",
        "prompt_char_length": 120,
        "candidate_count": 3,
        "prompt_token_length": 64,
        "completion_token_length": 19,
        "invalid_action_cap_hits": 1,
        "no_learning_signal_warning": True,
        "sample_parse_results": [
            {
                "valid_json": False,
                "valid_action_shape": False,
                "repaired": False,
                "used_candidate_choice": False,
                "error_code": "no_json_object",
                "notes": ["json_parse_failed"],
                "raw_text": "plain text",
                "prompt_hash": "hash1",
                "prompt_preview": "PROMPT",
                "prompt_char_length": 120,
                "prompt_token_length": 64,
                "completion_token_length": 19,
                "candidate_count": 3,
                "generation": {"temperature": 0.35, "top_p": 0.9, "max_new_tokens": 32, "stop_reason": "max_new_tokens", "do_sample": True},
                "invalid_action": True,
                "estimated_total_return": -90.0,
            }
        ],
        "valid_json_rate": 0.0,
        "valid_action_rate": 0.0,
    }

    payload = json.loads(trainer.run()["training_metrics"].read_text())

    assert payload["parser_error_code_counts"]["no_json_object"] == 2
    assert payload["candidate_choice_rate"] == 0.0
    assert payload["repair_rate"] == 0.0
    assert payload["invalid_action_cap_hits"] == 1
    assert payload["failed_parse_examples"][0]["raw_text"] == "plain text"
    assert payload["prompt_length_stats"]["max"] == 120
    assert payload["completion_length_stats"]["max"] == 19
    monkeypatch.delenv("HF_JOB_ID", raising=False)


def test_non_hf_generation_keeps_sampling_guards_off() -> None:
    trainer = _build_generation_trainer(hf_safe=False)

    trainer._generate_completion("prompt", do_sample=True)

    assert trainer.latest_generation_hf_safe is False
    assert trainer.latest_generation_used_sampling_model is False
    assert trainer.latest_generation_model_kind == "training"
    assert trainer.latest_generation_top_k == SAFE_TOP_K
    assert trainer.latest_generation_repetition_penalty == SAFE_REPETITION_PENALTY
    assert trainer.latest_generation_kwargs["top_k"] == SAFE_TOP_K
    assert trainer.latest_generation_kwargs["repetition_penalty"] == SAFE_REPETITION_PENALTY
    assert trainer.last_generation_stop_reason == "max_new_tokens"
    assert trainer.latest_generation_completion_length == 2
    assert trainer.last_prompt_preview == "prompt"
    assert trainer.last_prompt_hash is not None
    assert trainer.training_candidate_choice is True
    assert trainer.training_generation_temperature == 0.35
    assert trainer.training_generation_top_p == 0.9
    assert trainer.training_generation_max_new_tokens == 32


def test_sanity_check_records_candidate_choice_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_JOB_ID", "job-789")
    trainer = LocalGRPOTrainer(_build_args(output_dir=str(tmp_path)))
    payload = json.loads(trainer.write_sanity_check().read_text())
    assert payload["candidate_choice_default"] is True
    monkeypatch.delenv("HF_JOB_ID", raising=False)


def test_fake_ids_shape_for_debug_helpers() -> None:
    ids = _FakeIds(17)
    assert ids.shape[0] == 17
    assert ids.detach().cpu() is ids
    assert ids.to("cuda") is ids


def test_failed_parse_examples_are_capped() -> None:
    trainer = LocalGRPOTrainer(_build_args())
    trainer.last_failed_examples = [
        {"raw_text": "one"},
        {"raw_text": "two"},
        {"raw_text": "three"},
    ]
    assert len(trainer.last_failed_examples) == 3


def test_has_nonfinite_gradients_detects_nan() -> None:
    trainer = LocalGRPOTrainer(_build_args())
    trainer._torch = FakeTorch()
    trainer.model = SimpleNamespace(
        parameters=lambda: [
            SimpleNamespace(grad=SimpleNamespace(finite=True)),
            SimpleNamespace(grad=SimpleNamespace(finite=False)),
        ]
    )

    assert trainer._has_nonfinite_gradients() is True


def test_training_metrics_include_hf_safety_fields(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_JOB_ID", "job-metrics")
    output_dir = tmp_path / "hf_train"
    trainer = LocalGRPOTrainer(_build_args(output_dir=str(output_dir)))
    trainer.dependencies = {"torch": True, "transformers": True}
    trainer.dtype_name = "float16"
    trainer.sampling_dtype_name = "float16"
    trainer.generation_safety_flags = {
        "hf_safe_generation": True,
        "use_safe_sampling": True,
        "remove_invalid_values": True,
        "renormalize_logits": True,
    }
    trainer.skipped_nonfinite_updates = 2
    trainer._validate_training_environment = lambda: None
    trainer._seed_everything = lambda: None
    trainer._load_model = lambda: None
    trainer._device_summary = lambda: {"gpu_available": True, "dtype": "float16"}
    trainer._run_training_episode = lambda task_id, episode_number: {
        "valid_json": True,
        "valid_action_shape": True,
        "invalid_action_count": 0,
    }
    trainer._evaluate_model = lambda policy_id: {
        "aggregate": {
            "overall": {
                "mean_total_reward": 1.0,
                "completed_deliveries": 1,
                "urgent_successes": 1,
                "safety_violations": 0,
            }
        },
        "policy_results": {
            policy_id: {
                "completed_deliveries": 1,
                "urgent_successes": 1,
                "safety_violations": 0,
            }
        },
        "tasks": [],
        "ranking": [],
    }

    payload = json.loads(trainer.run()["training_metrics"].read_text())

    assert payload["training_executed"] is True
    assert payload["generation_safety"]["hf_safe_generation"] is True
    assert payload["sampling_model"] == {"enabled": True, "dtype": "float16"}
    assert payload["skipped_nonfinite_updates"] == 2
    assert payload["output_dir"] == str(output_dir)

    monkeypatch.delenv("HF_JOB_ID", raising=False)


def test_sanity_check_records_hf_generation_metadata(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_JOB_ID", "job-789")
    trainer = LocalGRPOTrainer(_build_args(output_dir=str(tmp_path)))
    trainer.generation_safety_flags = {
        "hf_safe_generation": True,
        "use_safe_sampling": True,
        "remove_invalid_values": True,
        "renormalize_logits": False,
    }
    trainer.sampling_dtype_name = "float16"

    payload = json.loads(trainer.write_sanity_check().read_text())

    assert payload["training_executed"] is False
    assert payload["generation_safety"]["hf_safe_generation"] is True
    assert payload["generation_safety"]["use_safe_sampling"] is True
    assert payload["sampling_model"] == {"enabled": True, "dtype": "float16"}

    monkeypatch.delenv("HF_JOB_ID", raising=False)
