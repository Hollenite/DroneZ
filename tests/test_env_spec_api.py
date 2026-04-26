from __future__ import annotations

from fastapi.testclient import TestClient

from urbanair.server.app import app


client = TestClient(app)


def test_env_spec_endpoint_describes_environment() -> None:
    response = client.get("/env-spec")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "DroneZ Fleet Delivery Environment"
    assert "reset" in payload["openenv_loop"]["sequence"]
    assert payload["links"]["agent_playground"] == "/agent/index.html"
    assert payload["training_status"]["reward_improvement"].startswith("not proven")


def test_action_space_endpoint_returns_allowed_actions() -> None:
    response = client.get("/action-space")

    assert response.status_code == 200
    payload = response.json()
    names = {item["action"] for item in payload["actions"]}
    assert "assign_delivery" in names
    assert "attempt_delivery" in names
    assert "candidate-choice" in payload["candidate_choice_note"]


def test_observation_space_endpoint_returns_schema_fields() -> None:
    response = client.get("/observation-space")

    assert response.status_code == 200
    fields = response.json()["fields"]
    assert "fleet" in fields
    assert "orders" in fields
    assert "city" in fields
    assert "charging" in fields


def test_valid_actions_and_step_loop() -> None:
    reset_response = client.post("/reset", json={"task_id": "easy"})
    assert reset_response.status_code == 200
    session_id = reset_response.json()["session_id"]

    valid_response = client.post("/valid-actions", json={"session_id": session_id})
    assert valid_response.status_code == 200
    actions = valid_response.json()["actions"]
    assert actions
    assert "action" in actions[0]["action"]
    assert "reason" in actions[0]

    step_response = client.post("/step", json={"action": actions[0]["action"]})
    assert step_response.status_code == 200
    payload = step_response.json()
    assert "reward" in payload
    assert "done" in payload
    assert "info" in payload
    assert "observation" in payload
