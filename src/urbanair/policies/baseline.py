from __future__ import annotations

from typing import Any

from .base import Policy


class NaivePolicy(Policy):
    policy_id = "naive"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        fleet = observation["fleet"]
        orders = observation["orders"]
        charging = observation["charging"]

        recovery_order = _select_recovery_order(orders)
        if recovery_order is not None:
            return {"action": "fallback_to_locker", "params": {"order_id": recovery_order["order_id"], "locker_id": f"L-{recovery_order['zone_id']}"}}

        low_battery_drone = next((item for item in fleet if item["drone_type"] != "relay" and item["status"] in {"idle", "holding"} and item["battery"] <= 25), None)
        if low_battery_drone is not None:
            station = min(charging, key=lambda item: (item["queue_size"], item["occupied_slots"], item["station_id"]))
            if not low_battery_drone.get("reserved_station_id"):
                return {"action": "reserve_charger", "params": {"drone_id": low_battery_drone["drone_id"], "station_id": station["station_id"]}}
            return {"action": "return_to_charge", "params": {"drone_id": low_battery_drone["drone_id"], "station_id": station["station_id"]}}

        attempt_drone = next((item for item in fleet if item["assigned_order_id"] and item.get("eta") == 0), None)
        if attempt_drone is not None:
            return {"action": "attempt_delivery", "params": {"drone_id": attempt_drone["drone_id"], "mode": "handoff"}}

        drone = next((item for item in fleet if item["drone_type"] != "relay" and item["status"] in {"idle", "holding"} and item.get("hold_reason") != "zone_hold"), None)
        order = next((item for item in orders if item["assigned_drone_id"] is None and item["status"] not in {"delivered", "canceled"}), None)
        if drone is not None and order is not None:
            return {"action": "assign_delivery", "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]}}

        focus_order = _select_focus_order(orders)
        return {"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}}


class HeuristicPolicy(Policy):
    policy_id = "heuristic"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        fleet = observation["fleet"]
        orders = observation["orders"]
        charging = observation["charging"]
        sectors = {sector["zone_id"]: sector for sector in observation["city"]["sectors"]}
        held_zones = set(observation["city"].get("held_zones", []))

        if held_zones:
            return {"action": "resume_operations", "params": {"zone_id": sorted(held_zones)[0]}}

        unsafe_sector = next((sector for sector in sectors.values() if sector["is_no_fly"] and sector["zone_id"] != "hub"), None)
        if unsafe_sector is not None:
            active_zone_drone = next((drone for drone in fleet if drone["current_zone"] == unsafe_sector["zone_id"] and drone["status"] in {"idle", "holding"}), None)
            if active_zone_drone is not None:
                return {"action": "hold_fleet", "params": {"zone_id": unsafe_sector["zone_id"]}}

        locker_candidate = next((order for order in orders if order["status"] in {"deferred", "delivery_attempted"} and order["recipient_availability"] == "unavailable"), None)
        if locker_candidate is not None:
            return {"action": "fallback_to_locker", "params": {"order_id": locker_candidate["order_id"], "locker_id": f"L-{locker_candidate['zone_id']}"}}

        attempt_drone = next((item for item in fleet if item["assigned_order_id"] and item.get("eta") == 0), None)
        if attempt_drone is not None:
            order = next((item for item in orders if item["order_id"] == attempt_drone["assigned_order_id"]), None)
            mode = "locker" if order and order["recipient_availability"] == "unavailable" else "handoff"
            return {"action": "attempt_delivery", "params": {"drone_id": attempt_drone["drone_id"], "mode": mode}}

        stale_order = next((order for order in orders if order["status"] in {"deferred", "delivery_attempted"}), None)
        if stale_order is not None and stale_order["priority"] in {"urgent", "medical"}:
            return {"action": "delay_order", "params": {"order_id": stale_order["order_id"]}}

        charge_candidate = next(
            (
                item
                for item in fleet
                if item["drone_type"] != "relay"
                and item["status"] in {"idle", "holding"}
                and item.get("hold_reason") != "zone_hold"
                and (item["battery"] <= 30 or item["health_risk"] in {"high", "critical"})
            ),
            None,
        )
        if charge_candidate is not None:
            station = min(charging, key=lambda item: (item["queue_size"], item["occupied_slots"], item["station_id"]))
            if charge_candidate.get("reserved_station_id") != station["station_id"]:
                return {"action": "reserve_charger", "params": {"drone_id": charge_candidate["drone_id"], "station_id": station["station_id"]}}
            return {"action": "return_to_charge", "params": {"drone_id": charge_candidate["drone_id"], "station_id": station["station_id"]}}

        best_pair: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for drone in fleet:
            if drone["drone_type"] == "relay" or drone["status"] not in {"idle", "holding"} or drone.get("hold_reason") == "zone_hold":
                continue
            for order in orders:
                if order["assigned_drone_id"] is not None or order["status"] in {"delivered", "canceled"}:
                    continue
                if order["package_weight"] > drone["payload_capacity"]:
                    continue
                score = _score_assignment(drone, order, sectors.get(order["zone_id"]))
                if best_pair is None or score > best_pair[0]:
                    best_pair = (score, drone, order)

        if best_pair is not None:
            _, drone, order = best_pair
            return {"action": "assign_delivery", "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]}}

        focus_order = _select_focus_order(orders)
        return {"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}}


def _select_focus_order(orders: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [order for order in orders if order["status"] not in {"delivered", "canceled"}]
    return min(candidates, key=lambda item: (_priority_rank(item["priority"]), item["deadline"], item["order_id"]))


def _score_assignment(drone: dict[str, Any], order: dict[str, Any], sector: dict[str, Any] | None) -> float:
    priority_bonus = {"medical": 30.0, "urgent": 20.0, "normal": 0.0}[order["priority"]]
    deadline_bonus = max(0.0, 12.0 - float(order["deadline"]))
    battery_bonus = drone["battery"] / 10.0
    weather_penalty = 0.0
    congestion_penalty = 0.0
    hold_penalty = 4.0 if drone.get("hold_reason") else 0.0
    if sector is not None:
        weather_penalty = {"clear": 0.0, "moderate_wind": 2.0, "heavy_rain": 4.0, "storm": 6.0}[sector["weather"]]
        congestion_penalty = float(sector["congestion_score"]) * 5.0
        if sector.get("operations_paused"):
            congestion_penalty += 25.0
    return priority_bonus + deadline_bonus + battery_bonus - weather_penalty - congestion_penalty - hold_penalty


def _select_recovery_order(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            order
            for order in orders
            if order["status"] in {"deferred", "delivery_attempted"} and order["recipient_availability"] == "unavailable"
        ),
        None,
    )


def _priority_rank(priority: str) -> int:
    return {"medical": 0, "urgent": 1, "normal": 2}[priority]
