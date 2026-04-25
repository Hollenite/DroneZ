from __future__ import annotations

from fastapi.testclient import TestClient

from urbanair.server.app import app


client = TestClient(app)


def test_root_endpoint_opens_demo() -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/demo/index.html"


def test_api_endpoint_describes_runtime() -> None:
    response = client.get("/api")
    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "DroneZ OpenEnv Runtime"
    assert payload["health"] == "/health"
    assert payload["docs"] == "/docs"
    assert payload["demo"] == "/demo/index.html"


def test_tasks_endpoint_lists_expected_tasks() -> None:
    response = client.get("/tasks")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_task_id"] == "easy"
    assert payload["tasks"] == ["demo", "easy", "hard", "medium"]


def test_default_reset_state_and_step_endpoints() -> None:
    reset_response = client.post("/reset", json={"task_id": "demo"})
    assert reset_response.status_code == 200
    reset_payload = reset_response.json()
    assert reset_payload["observation"]["task_id"] == "demo"

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["state"]["task_id"] == "demo"

    observation = reset_payload["observation"]
    drone = next(item for item in observation["fleet"] if item["drone_type"] != "relay" and item["status"] == "idle")
    order = next(item for item in observation["orders"] if item["assigned_drone_id"] is None)
    step_response = client.post(
        "/step",
        json={
            "action": {
                "action": "assign_delivery",
                "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]},
            }
        },
    )
    assert step_response.status_code == 200
    step_payload = step_response.json()
    assert step_payload["session_id"] == reset_payload["session_id"]
    assert step_payload["observation"]["step"] == 1
    assert step_payload["info"]["invalid_action"] is False


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

    state_response = client.get(f"/sessions/{session_id}/state")
    assert state_response.status_code == 200
    state_payload = state_response.json()
    assert state_payload["state"]["task_id"] == "easy"
    assert state_payload["state"]["episode_action_count"] >= 1


def test_unknown_session_returns_not_found() -> None:
    response = client.post("/sessions/missing/step", json={"action": {"action": "prioritize_order", "params": {"order_id": "O1"}}})

    assert response.status_code == 404
