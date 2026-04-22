from __future__ import annotations

from urbanair.env.environment import DroneZEnvironment


def test_valid_assign_delivery_advances_tick() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    drone = next(item for item in observation["fleet"] if item["drone_type"] != "relay" and item["status"] == "idle")
    order = next(item for item in observation["orders"] if item["assigned_drone_id"] is None)

    next_observation, reward, done, info = env.step(
        {
            "action": "assign_delivery",
            "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]},
        }
    )

    assert next_observation["step"] == 1
    assert reward != 0.0
    assert done is False
    assert info["invalid_action"] is False
    assert next_observation["last_action"]["action"] == "assign_delivery"


def test_invalid_action_returns_penalty_without_advancing_tick() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    next_observation, reward, done, info = env.step(
        {
            "action": "assign_delivery",
            "params": {"drone_id": "missing", "order_id": "O1"},
        }
    )

    assert observation["step"] == 0
    assert next_observation["step"] == 0
    assert reward == -5.0
    assert done is False
    assert info["invalid_action"] is True
    assert info["error_code"] == "unknown_drone"


def test_reserve_charger_tracks_station_reservation() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    drone = next(item for item in observation["fleet"] if item["drone_type"] != "relay" and item["status"] == "idle")
    station = observation["charging"][0]

    next_observation, _, _, info = env.step(
        {"action": "reserve_charger", "params": {"drone_id": drone["drone_id"], "station_id": station["station_id"]}}
    )

    reserved = next(item for item in next_observation["fleet"] if item["drone_id"] == drone["drone_id"])
    charging_station = next(item for item in next_observation["charging"] if item["station_id"] == station["station_id"])
    assert reserved["reserved_station_id"] == station["station_id"]
    assert reserved["hold_reason"] == "charger_reserved"
    assert drone["drone_id"] in charging_station["reserved_drone_ids"]
    assert info["invalid_action"] is False


def test_delay_order_marks_order_for_recovery() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")

    order = next(item for item in observation["orders"] if item["status"] == "queued")
    next_observation, _, _, info = env.step({"action": "delay_order", "params": {"order_id": order["order_id"]}})

    delayed = next(item for item in next_observation["orders"] if item["order_id"] == order["order_id"])
    assert delayed["status"] == "deferred"
    assert order["order_id"] in info["pending_recovery_orders"]


def test_hold_and_resume_fleet_toggle_zone_state() -> None:
    env = DroneZEnvironment()
    observation, _ = env.reset("easy")
    zone_id = next(sector["zone_id"] for sector in observation["city"]["sectors"] if sector["zone_id"] != "hub")

    held_observation, _, _, held_info = env.step({"action": "hold_fleet", "params": {"zone_id": zone_id}})
    held_sector = next(sector for sector in held_observation["city"]["sectors"] if sector["zone_id"] == zone_id)
    assert held_sector["operations_paused"] is True
    assert held_info["zone_holds"][zone_id] == "manual_hold"

    resumed_observation, _, _, resumed_info = env.step({"action": "resume_operations", "params": {"zone_id": zone_id}})
    resumed_sector = next(sector for sector in resumed_observation["city"]["sectors"] if sector["zone_id"] == zone_id)
    assert resumed_sector["operations_paused"] is False
    assert zone_id not in resumed_info["zone_holds"]


def test_reroute_updates_visible_corridor_and_path() -> None:
    env = DroneZEnvironment()
    env.reset("easy")
    drone = next(item for item in env.state.fleet if item.drone_type.value != "relay" and item.status.value == "idle")
    order = next(item for item in env.state.orders if item.assigned_drone_id is None)
    sector = next(item for item in env.state.sectors if item.zone_id == order.zone_id)
    sector.is_no_fly = True
    order.assigned_drone_id = drone.drone_id
    order.status = "assigned"
    drone.assigned_order_id = order.order_id
    drone.status = drone.status.ASSIGNED
    drone.target_zone = order.zone_id
    drone.eta = 2
    drone.active_corridor = "direct"
    drone.flight_path = [drone.current_zone, "direct", order.zone_id]

    next_observation, reward, _, info = env.step({"action": "reroute", "params": {"drone_id": drone.drone_id, "corridor": "safe"}})

    routed = next(item for item in next_observation["fleet"] if item["drone_id"] == drone.drone_id)
    assert routed["active_corridor"] == "safe"
    assert routed["flight_path"][1] == "safe"
    assert info["invalid_action"] is False
    assert info["reward_breakdown"]["negative"].get("unnecessary_reroute") is None
    assert info["reward_breakdown"]["positive"]["regulatory_compliance"] > 0
    assert reward > 0


def test_reroute_is_rejected_for_held_zone() -> None:
    env = DroneZEnvironment()
    env.reset("easy")
    drone = next(item for item in env.state.fleet if item.drone_type.value != "relay" and item.status.value == "idle")
    order = next(item for item in env.state.orders if item.assigned_drone_id is None)
    sector = next(item for item in env.state.sectors if item.zone_id == order.zone_id)
    sector.operations_paused = True
    order.assigned_drone_id = drone.drone_id
    order.status = "assigned"
    drone.assigned_order_id = order.order_id
    drone.status = drone.status.ASSIGNED
    drone.target_zone = order.zone_id
    drone.eta = 2
    drone.active_corridor = "direct"
    drone.flight_path = [drone.current_zone, "direct", order.zone_id]

    next_observation, reward, _, info = env.step({"action": "reroute", "params": {"drone_id": drone.drone_id, "corridor": "safe"}})

    routed = next(item for item in next_observation["fleet"] if item["drone_id"] == drone.drone_id)
    assert info["invalid_action"] is True
    assert info["error_code"] == "zone_paused"
    assert reward == -5.0
    assert routed["active_corridor"] == "direct"
    assert routed["flight_path"][1] == "direct"


def test_swap_assignments_cross_links_orders_and_drones() -> None:
    env = DroneZEnvironment()
    env.reset("easy")
    drones = [item for item in env.state.fleet if item.drone_type.value != "relay"][:2]
    orders = [item for item in env.state.orders if item.assigned_drone_id is None][:2]

    for drone, order in zip(drones, orders):
        drone.status = drone.status.ASSIGNED
        drone.assigned_order_id = order.order_id
        drone.target_zone = order.zone_id
        drone.eta = 2
        drone.active_corridor = "direct"
        drone.flight_path = [drone.current_zone, "direct", order.zone_id]
        order.assigned_drone_id = drone.drone_id
        order.status = "assigned"

    swapped_observation, _, _, info = env.step({"action": "swap_assignments", "params": {"drone_a": drones[0].drone_id, "drone_b": drones[1].drone_id}})

    swapped_a = next(item for item in swapped_observation["fleet"] if item["drone_id"] == drones[0].drone_id)
    swapped_b = next(item for item in swapped_observation["fleet"] if item["drone_id"] == drones[1].drone_id)
    order_a = next(item for item in swapped_observation["orders"] if item["order_id"] == orders[0].order_id)
    order_b = next(item for item in swapped_observation["orders"] if item["order_id"] == orders[1].order_id)

    assert swapped_a["assigned_order_id"] == orders[1].order_id
    assert swapped_b["assigned_order_id"] == orders[0].order_id
    assert order_a["assigned_drone_id"] == drones[1].drone_id
    assert order_b["assigned_drone_id"] == drones[0].drone_id
    assert info["invalid_action"] is False
