from __future__ import annotations

from inference import format_step_trace, resolve_policy, run_inference_episode


def test_resolve_policy_supports_expected_ids() -> None:
    assert resolve_policy("random").policy_id == "random"
    assert resolve_policy("naive").policy_id == "naive"
    assert resolve_policy("heuristic").policy_id == "heuristic"
    assert resolve_policy("improved").policy_id == "improved"


def test_run_inference_episode_returns_demo_trace_and_summary() -> None:
    result = run_inference_episode("demo", resolve_policy("heuristic"), max_steps=8)

    assert result["summary"].task_id == "demo"
    assert result["summary"].policy_id == "heuristic"
    assert len(result["trace"]) == 8
    assert any(step["triggered_scripted_events"] for step in result["trace"])


def test_format_step_trace_includes_action_and_reward() -> None:
    lines = format_step_trace(
        [
            {
                "step": 1,
                "action_index": 1,
                "action": {"action": "prioritize_order", "params": {"order_id": "O1"}},
                "reward": 3.5,
                "done": False,
                "invalid_action": False,
                "pending_orders": 4,
                "triggered_scripted_events": ["urgent_insertion"],
            }
        ]
    )

    assert "STEP 1 (action 1) | action={'action': 'prioritize_order', 'params': {'order_id': 'O1'}} | reward=3.50 | pending_orders=4 | invalid=False | done=False" == lines[0]
    assert "scripted_events=['urgent_insertion']" in lines[1]
