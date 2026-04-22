from __future__ import annotations

from urbanair.enums import DroneStatus
from urbanair.sim.engine import SimulatorEngine


def test_step_advances_tick_and_changes_state() -> None:
    engine = SimulatorEngine.from_repo_configs()
    state = engine.reset("easy")

    idle_drone = next(drone for drone in state.fleet if drone.status == DroneStatus.IDLE and drone.drone_type.value != "relay")
    order = next(order for order in state.orders if order.assigned_drone_id is None)

    before_deadline = order.deadline
    before_battery = idle_drone.battery

    result = engine.step(
        state,
        {
            "action_type": "assign_delivery",
            "drone_id": idle_drone.drone_id,
            "order_id": order.order_id,
        },
    )

    assert state.tick == 1
    assert order.deadline == before_deadline - 1
    assert idle_drone.battery < before_battery
    assert result.observation.time_step == 1
    assert isinstance(result.observation.recent_events, list)
