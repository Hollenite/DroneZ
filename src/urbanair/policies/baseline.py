from __future__ import annotations

from typing import Any

from .base import Policy


class RandomPolicy(Policy):
    policy_id = "random"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        candidates = _build_candidate_actions(observation, allow_reroute=True, safe_only=False)
        return candidates[_stable_index(observation, len(candidates))]


class NaivePolicy(Policy):
    policy_id = "naive"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        fleet = observation["fleet"]
        orders = observation["orders"]
        charging = observation["charging"]

        attempt_drone = _ready_delivery_drone(fleet)
        if attempt_drone is not None:
            return {"action": "attempt_delivery", "params": {"drone_id": attempt_drone["drone_id"], "mode": "handoff"}}

        recovery_order = _select_recovery_order(orders)
        if recovery_order is not None:
            return {"action": "fallback_to_locker", "params": {"order_id": recovery_order["order_id"], "locker_id": f"L-{recovery_order['zone_id']}"}}

        low_battery_drone = next(
            (
                item
                for item in fleet
                if item["drone_type"] != "relay"
                and item["status"] in {"idle", "holding"}
                and item["battery"] <= 25
                and item.get("hold_reason") != "zone_hold"
            ),
            None,
        )
        if low_battery_drone is not None:
            station = min(charging, key=lambda item: (item["queue_size"], item["occupied_slots"], item["station_id"]))
            if low_battery_drone.get("reserved_station_id") != station["station_id"]:
                return {"action": "reserve_charger", "params": {"drone_id": low_battery_drone["drone_id"], "station_id": station["station_id"]}}
            return {"action": "return_to_charge", "params": {"drone_id": low_battery_drone["drone_id"], "station_id": station["station_id"]}}

        drone = next(
            (
                item
                for item in fleet
                if item["drone_type"] != "relay"
                and item["status"] in {"idle", "holding"}
                and item.get("hold_reason") != "zone_hold"
            ),
            None,
        )
        order = next((item for item in orders if item["assigned_drone_id"] is None and item["status"] not in {"delivered", "canceled"}), None)
        if drone is not None and order is not None:
            return {"action": "assign_delivery", "params": {"drone_id": drone["drone_id"], "order_id": order["order_id"]}}

        focus_order = _select_focus_order(orders)
        return {"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}}


class HeuristicPolicy(Policy):
    policy_id = "heuristic"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return _choose_structured_action(observation, prefer_strict_safety=False)


class ImprovedPolicy(Policy):
    policy_id = "improved"

    def choose_action(self, observation: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return _choose_structured_action(observation, prefer_strict_safety=True)


def _choose_structured_action(observation: dict[str, Any], prefer_strict_safety: bool) -> dict[str, Any]:
    fleet = observation["fleet"]
    orders = observation["orders"]
    charging = observation["charging"]
    sectors = {sector["zone_id"]: sector for sector in observation["city"]["sectors"]}
    held_zones = set(observation["city"].get("held_zones", []))

    attempt_drone = _ready_delivery_drone(fleet)
    if attempt_drone is not None:
        order = next((item for item in orders if item["order_id"] == attempt_drone["assigned_order_id"]), None)
        mode = "locker" if order and order["recipient_availability"] == "unavailable" else "handoff"
        return {"action": "attempt_delivery", "params": {"drone_id": attempt_drone["drone_id"], "mode": mode}}

    locker_candidate = _select_recovery_order(orders)
    if locker_candidate is not None:
        return {"action": "fallback_to_locker", "params": {"order_id": locker_candidate["order_id"], "locker_id": f"L-{locker_candidate['zone_id']}"}}

    resumable_zone = _select_resumable_zone(held_zones, sectors)
    if resumable_zone is not None:
        return {"action": "resume_operations", "params": {"zone_id": resumable_zone}}

    reroute_candidate = _select_reroute_candidate(fleet, sectors)
    if reroute_candidate is not None:
        return {"action": "reroute", "params": {"drone_id": reroute_candidate["drone_id"], "corridor": "safe"}}

    hold_zone = _select_hold_zone(fleet, sectors, held_zones)
    if hold_zone is not None:
        return {"action": "hold_fleet", "params": {"zone_id": hold_zone}}

    charge_candidate = _select_charge_candidate(fleet, charging, orders)
    if charge_candidate is not None:
        return charge_candidate

    swap_candidate = _select_swap_candidate(fleet, orders, sectors)
    if swap_candidate is not None and not prefer_strict_safety:
        return {"action": "swap_assignments", "params": swap_candidate}

    best_assignment = _select_best_assignment(fleet, orders, sectors, prefer_strict_safety=prefer_strict_safety)
    if best_assignment is not None:
        return {"action": "assign_delivery", "params": best_assignment}

    focus_order = _select_focus_order(orders)
    if focus_order["priority"] in {"urgent", "medical"} and focus_order["deadline"] <= 2:
        return {"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}}

    stale_order = next(
        (
            order
            for order in orders
            if order["status"] in {"deferred", "delivery_attempted"}
            and order["priority"] == "normal"
            and order["recipient_availability"] != "unavailable"
        ),
        None,
    )
    if stale_order is not None and not prefer_strict_safety:
        return {"action": "delay_order", "params": {"order_id": stale_order["order_id"]}}

    return {"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}}


def _build_candidate_actions(observation: dict[str, Any], allow_reroute: bool, safe_only: bool) -> list[dict[str, Any]]:
    fleet = observation["fleet"]
    orders = observation["orders"]
    charging = observation["charging"]
    sectors = {sector["zone_id"]: sector for sector in observation["city"]["sectors"]}
    candidates: list[dict[str, Any]] = []

    attempt_drone = _ready_delivery_drone(fleet)
    if attempt_drone is not None:
        candidates.append({"action": "attempt_delivery", "params": {"drone_id": attempt_drone["drone_id"], "mode": "handoff"}})

    recovery_order = _select_recovery_order(orders)
    if recovery_order is not None:
        candidates.append({"action": "fallback_to_locker", "params": {"order_id": recovery_order["order_id"], "locker_id": f"L-{recovery_order['zone_id']}"}})

    if allow_reroute:
        reroute_candidate = _select_reroute_candidate(fleet, sectors)
        if reroute_candidate is not None:
            candidates.append({"action": "reroute", "params": {"drone_id": reroute_candidate["drone_id"], "corridor": "safe"}})

    charge_candidate = _select_charge_candidate(fleet, charging, orders)
    if charge_candidate is not None:
        candidates.append(charge_candidate)

    for assignment in _top_assignment_candidates(fleet, orders, sectors, prefer_strict_safety=safe_only)[:3]:
        candidates.append({"action": "assign_delivery", "params": assignment})

    focus_order = _select_focus_order(orders)
    candidates.append({"action": "prioritize_order", "params": {"order_id": focus_order["order_id"]}})
    return candidates


def _stable_index(observation: dict[str, Any], length: int) -> int:
    fingerprint = f"{observation['task_id']}:{observation['step']}:{len(observation['orders'])}:{len(observation['fleet'])}"
    return sum(ord(char) for char in fingerprint) % max(1, length)


def _ready_delivery_drone(fleet: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in fleet
            if item["drone_type"] != "relay"
            and item["assigned_order_id"]
            and item.get("eta") == 0
        ),
        None,
    )


def _select_recovery_order(orders: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            order
            for order in orders
            if order["status"] not in {"delivered", "canceled", "locker_fallback"}
            and order["recipient_availability"] == "unavailable"
        ),
        None,
    )


def _select_reroute_candidate(fleet: list[dict[str, Any]], sectors: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return next(
        (
            drone
            for drone in fleet
            if drone["drone_type"] != "relay"
            and drone.get("target_zone")
            and drone["status"] in {"assigned", "in_flight"}
            and drone.get("eta") is not None
            and drone["eta"] > 0
            and _is_hazardous_sector(sectors.get(drone["target_zone"]))
            and drone.get("active_corridor") != "safe"
        ),
        None,
    )


def _select_resumable_zone(held_zones: set[str], sectors: dict[str, dict[str, Any]]) -> str | None:
    for zone_id in sorted(held_zones):
        if not _is_hazardous_sector(sectors.get(zone_id)):
            return zone_id
    return None


def _select_hold_zone(fleet: list[dict[str, Any]], sectors: dict[str, dict[str, Any]], held_zones: set[str]) -> str | None:
    for zone_id, sector in sorted(sectors.items()):
        if zone_id == "hub" or zone_id in held_zones or not _is_hazardous_sector(sector):
            continue
        active_zone_drone = next((drone for drone in fleet if drone["current_zone"] == zone_id and drone["status"] in {"idle", "holding"}), None)
        if active_zone_drone is not None:
            return zone_id
    return None


def _select_charge_candidate(
    fleet: list[dict[str, Any]],
    charging: list[dict[str, Any]],
    orders: list[dict[str, Any]],
) -> dict[str, Any] | None:
    pending_urgent = any(order["status"] not in {"delivered", "canceled"} and order["priority"] in {"urgent", "medical"} for order in orders)
    candidate = next(
        (
            item
            for item in fleet
            if item["drone_type"] != "relay"
            and item["status"] in {"idle", "holding"}
            and item.get("hold_reason") != "zone_hold"
            and (
                item["battery"] <= 30
                or item["health_risk"] in {"high", "critical"}
                or (not pending_urgent and item["battery"] <= 45 and item["status"] == "holding")
            )
        ),
        None,
    )
    if candidate is None:
        return None

    station = min(charging, key=lambda item: (item["queue_size"], item["occupied_slots"], item["station_id"]))
    if candidate.get("reserved_station_id") != station["station_id"]:
        return {"action": "reserve_charger", "params": {"drone_id": candidate["drone_id"], "station_id": station["station_id"]}}
    return {"action": "return_to_charge", "params": {"drone_id": candidate["drone_id"], "station_id": station["station_id"]}}


def _select_best_assignment(
    fleet: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    sectors: dict[str, dict[str, Any]],
    *,
    prefer_strict_safety: bool,
) -> dict[str, str] | None:
    candidates = _top_assignment_candidates(fleet, orders, sectors, prefer_strict_safety=prefer_strict_safety)
    return candidates[0] if candidates else None


def _top_assignment_candidates(
    fleet: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    sectors: dict[str, dict[str, Any]],
    *,
    prefer_strict_safety: bool,
) -> list[dict[str, str]]:
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for drone in fleet:
        if drone["drone_type"] == "relay" or drone["status"] not in {"idle", "holding"} or drone.get("hold_reason") == "zone_hold":
            continue
        for order in orders:
            if order["assigned_drone_id"] is not None or order["status"] in {"delivered", "canceled"}:
                continue
            if order["package_weight"] > drone["payload_capacity"]:
                continue
            sector = sectors.get(order["zone_id"])
            if sector is not None and (sector["is_no_fly"] or sector.get("operations_paused")):
                continue
            if prefer_strict_safety and sector is not None and sector["weather"] in {"heavy_rain", "storm"}:
                continue
            score = _score_assignment(drone, order, sector, prefer_strict_safety=prefer_strict_safety)
            scored.append((score, drone, order))
    scored.sort(key=lambda item: (-item[0], item[1]["drone_id"], item[2]["order_id"]))
    return [{"drone_id": drone["drone_id"], "order_id": order["order_id"]} for _, drone, order in scored]


def _select_focus_order(orders: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [order for order in orders if order["status"] not in {"delivered", "canceled"}]
    return min(candidates, key=lambda item: (_priority_rank(item["priority"]), item["deadline"], item["order_id"]))


def _score_assignment(
    drone: dict[str, Any],
    order: dict[str, Any],
    sector: dict[str, Any] | None,
    *,
    prefer_strict_safety: bool,
) -> float:
    priority_bonus = {"medical": 36.0, "urgent": 24.0, "normal": 0.0}[order["priority"]]
    deadline_bonus = max(0.0, 14.0 - float(order["deadline"]))
    battery_bonus = drone["battery"] / 8.0
    payload_bonus = 2.0 if drone["payload_capacity"] >= order["package_weight"] else -10.0
    hold_penalty = 5.0 if drone.get("hold_reason") else 0.0

    weather_penalty = 0.0
    congestion_penalty = 0.0
    safety_penalty = 0.0
    if sector is not None:
        weather_penalty = {"clear": 0.0, "moderate_wind": 2.0, "heavy_rain": 6.0, "storm": 12.0}[sector["weather"]]
        congestion_penalty = float(sector["congestion_score"]) * 5.0
        if sector.get("is_no_fly"):
            safety_penalty += 18.0 if prefer_strict_safety else 9.0
        if sector.get("operations_paused"):
            safety_penalty += 18.0 if prefer_strict_safety else 9.0
        if sector["weather"] in {"heavy_rain", "storm"}:
            safety_penalty += 8.0 if prefer_strict_safety else 4.0

    return priority_bonus + deadline_bonus + battery_bonus + payload_bonus - weather_penalty - congestion_penalty - safety_penalty - hold_penalty


def _select_swap_candidate(
    fleet: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    sectors: dict[str, dict[str, Any]],
) -> dict[str, str] | None:
    assigned = [drone for drone in fleet if drone.get("assigned_order_id") and drone["status"] == "assigned"]
    orders_by_id = {order["order_id"]: order for order in orders}
    for index, drone_a in enumerate(assigned):
        order_a = orders_by_id.get(drone_a["assigned_order_id"])
        if order_a is None:
            continue
        for drone_b in assigned[index + 1 :]:
            order_b = orders_by_id.get(drone_b["assigned_order_id"])
            if order_b is None:
                continue
            if order_a["package_weight"] > drone_b["payload_capacity"] or order_b["package_weight"] > drone_a["payload_capacity"]:
                continue
            current_score = _score_assignment(drone_a, order_a, sectors.get(order_a["zone_id"]), prefer_strict_safety=False) + _score_assignment(drone_b, order_b, sectors.get(order_b["zone_id"]), prefer_strict_safety=False)
            swapped_score = _score_assignment(drone_a, order_b, sectors.get(order_b["zone_id"]), prefer_strict_safety=False) + _score_assignment(drone_b, order_a, sectors.get(order_a["zone_id"]), prefer_strict_safety=False)
            if swapped_score > current_score + 10:
                return {"drone_a": drone_a["drone_id"], "drone_b": drone_b["drone_id"]}
    return None


def _is_hazardous_sector(sector: dict[str, Any] | None) -> bool:
    if sector is None:
        return False
    return (
        sector.get("is_no_fly", False)
        or sector.get("operations_paused", False)
        or sector.get("weather") in {"heavy_rain", "storm"}
        or float(sector.get("congestion_score", 0.0)) >= 0.7
    )


def _priority_rank(priority: str) -> int:
    return {"medical": 0, "urgent": 1, "normal": 2}[priority]
