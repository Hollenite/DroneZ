from __future__ import annotations

from urbanair.sim.engine import SimulatorEngine


def test_failed_delivery_produces_recovery_action() -> None:
    engine = SimulatorEngine.from_repo_configs()
    state = engine.reset("demo")

    drone = next(drone for drone in state.fleet if drone.drone_type.value == "fast_light")
    order = next(order for order in state.orders if order.priority.value == "urgent")

    order.recipient_availability = order.recipient_availability.UNAVAILABLE

    engine.step(
        state,
        {
            "action_type": "assign_delivery",
            "drone_id": drone.drone_id,
            "order_id": order.order_id,
        },
    )
    drone.eta = 0

    result = engine.step(
        state,
        {
            "action_type": "attempt_delivery",
            "drone_id": drone.drone_id,
            "mode": "handoff",
        },
    )

    assert order.retry_count >= 1
    assert order.order_id in result.info["recovery_actions"]
    assert result.info["recovery_actions"][order.order_id] in {"retry", "reassign", "locker_fallback"}
    assert any("failed" in event.lower() or "fallback" in event.lower() for event in result.observation.recent_events)


def test_failure_clears_assignment_for_reassignment() -> None:
    engine = SimulatorEngine.from_repo_configs()
    state = engine.reset("demo")

    drone = next(drone for drone in state.fleet if drone.drone_type.value == "fast_light")
    order = next(order for order in state.orders if order.priority.value == "urgent")
    order.recipient_availability = order.recipient_availability.UNAVAILABLE

    engine.step(
        state,
        {
            "action_type": "assign_delivery",
            "drone_id": drone.drone_id,
            "order_id": order.order_id,
        },
    )
    drone.eta = 0
    engine.step(
        state,
        {
            "action_type": "attempt_delivery",
            "drone_id": drone.drone_id,
            "mode": "handoff",
        },
    )

    assert order.assigned_drone_id is None
    assert drone.assigned_order_id is None
    assert order.order_id in state.pending_recovery_orders
