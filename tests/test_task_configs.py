from __future__ import annotations

from pathlib import Path

import yaml

from urbanair.models import FleetProfilesConfig, RewardWeightsConfig, TaskConfig

ROOT = Path(__file__).resolve().parents[1]


def test_task_configs_load_into_models() -> None:
    task_dir = ROOT / "configs" / "tasks"
    task_files = sorted(task_dir.glob("*.yaml"))

    assert [path.name for path in task_files] == ["demo.yaml", "easy.yaml", "hard.yaml", "medium.yaml"]

    for path in task_files:
        payload = yaml.safe_load(path.read_text())
        config = TaskConfig.model_validate(payload)
        assert config.task_id == path.stem
        assert config.horizon > 0
        assert config.initial_orders >= config.urgent_orders
        assert all(isinstance(event.params, dict) for event in config.scripted_events)


def test_reward_and_fleet_configs_load() -> None:
    reward_payload = yaml.safe_load((ROOT / "configs" / "reward_weights.yaml").read_text())
    fleet_payload = yaml.safe_load((ROOT / "configs" / "fleet_profiles.yaml").read_text())

    reward_config = RewardWeightsConfig.model_validate(reward_payload)
    fleet_config = FleetProfilesConfig.model_validate(fleet_payload)

    assert "delivery_success" in reward_config.positive
    assert "invalid_action" in reward_config.negative
    assert len(fleet_config.profiles) == 4
    assert len(fleet_config.deployment_profiles) == 6
