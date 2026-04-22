from __future__ import annotations

from typing import Any

from .base import Policy


class NaivePolicy(Policy):
    policy_id = "naive"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        fleet = observation["fleet"]
        orders = observation["orders"]
        charging = observation["charging"]

        drone = next((item for item in fleet if item["drone_type"] != "relay" and item["status"] in {"idle", "holding"}), None)
        order = next((item for item in orders if item["assigned_drone_id"] is None and item["status"] not in {"delivered", "canceled"}), None)
        if drone is not None and order is not None:
            return {"action": "assign_delivery", "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]}}

        low_battery_drone = next((item for item in fleet if item["drone_type"] != "relay" and item["status"] in {"idle", "holding"} and item["battery"] <= 25), None)
        if low_battery_drone is not None:
            station = min(charging, key=lambda item: (item["queue_size"], item["occupied_slots"], item["station_id"]))
            return {"action": "return_to_charge", "params": {"drone_id": low_battery_drone["drone_id"], "station_id": station["station_id"]}}

        focus_order = _select_focus_order(orders)
        return {"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}}


class HeuristicPolicy(Policy):
    policy_id = "heuristic"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        fleet = observation["fleet"]
        orders = observation["orders"]
        charging = observation["charging"]
        sectors = {sector["zone_id"]: sector for sector in observation["city"]["sectors"]}

        charge_candidate = next(
            (
                item
                for item in fleet
                if item["drone_type"] != "relay"
                and item["status"] in {"idle", "holding"}
                and (item["battery"] <= 30 or item["health_risk"] in {"high", "critical"})
            ),
            None,
        )
        if charge_candidate is not None:
            station = min(charging, key=lambda item: (item["queue_size"], item["occupied_slots"], item["station_id"]))
            return {"action": "return_to_charge", "params": {"drone_id": charge_candidate["drone_id"], "station_id": station["station_id"]}}

        best_pair: tuple[float, dict[str, Any], dict[str, Any]] | None = None
        for drone in fleet:
            if drone["drone_type"] == "relay" or drone["status"] not in {"idle", "holding"}:
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
    if sector is not None:
        weather_penalty = {"clear": 0.0, "moderate_wind": 2.0, "heavy_rain": 4.0, "storm": 6.0}[sector["weather"]]
        congestion_penalty = float(sector["congestion_score"]) * 5.0
    return priority_bonus + deadline_bonus + battery_bonus - weather_penalty - congestion_penalty


def _priority_rank(priority: str) -> int:
    return {"medical": 0, "urgent": 1, "normal": 2}[priority]
