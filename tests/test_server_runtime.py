from __future__ import annotations

from fastapi.testclient import TestClient

from urbanair.server.app import app


client = TestClient(app)


def test_tasks_endpoint_lists_expected_tasks() -> None:
    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_task_id"] == "easy"
    assert payload["tasks"] == ["demo", "easy", "hard", "medium"]


def test_session_create_and_step_flow() -> None:
    create_response = client.post("/sessions", json={"task_id": "easy"})

    assert create_response.status_code == 200
    created = create_response.json()
    session_id = created["session_id"]
    observation = created["observation"]
    assert observation["task_id"] == "easy"

    drone = next(item for item in observation["fleet"] if item["drone_type"] != "relay" and item["status"] == "idle")
    order = next(item for item in observation["orders"] if item["assigned_drone_id"] is None)
    step_response = client.post(
        f"/sessions/{session_id}/step",
        json={
            "action": {
                "action": "assign_delivery",
                "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]},
            }
        },
    )

    assert step_response.status_code == 200
    stepped = step_response.json()
    assert stepped["session_id"] == session_id
    assert stepped["observation"]["step"] == 1
    assert stepped["info"]["invalid_action"] is False


def test_unknown_session_returns_not_found() -> None:
    response = client.post("/sessions/missing/step", json={"action": {"action": "prioritize_order", "params": {"order_id": "O1"}}})

    assert response.status_code == 404
