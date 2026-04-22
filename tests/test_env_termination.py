from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment
from urbanair.enums import DroneStatus


def test_done_when_horizon_reached() -> None:
    env = DroneZEnvironment()
    env.reset("easy")
    env.state.tick = env.state.task_config.horizon

    _, reward, done, info = env.step({"action": "unknown_action", "params": {}})

    assert reward == -5.0
    assert done is True
    assert info["done_reason"] == "horizon_reached"


def test_done_when_all_orders_resolved() -> None:
    env = DroneZEnvironment()
    env.reset("easy")
    env.state.resolved_order_ids = {order.order_id for order in env.state.orders}

    _, reward, done, info = env.step({"action": "unknown_action", "params": {}})

    assert reward == -5.0
    assert done is True
    assert info["done_reason"] == "all_orders_resolved"


def test_done_when_no_viable_drones_remain() -> None:
    env = DroneZEnvironment()
    env.reset("easy")
    for drone in env.state.fleet:
        drone.status = DroneStatus.OFFLINE
        drone.battery = 0

    _, reward, done, info = env.step({"action": "unknown_action", "params": {}})

    assert reward == -5.0
    assert done is True
    assert info["done_reason"] == "no_viable_drones"
