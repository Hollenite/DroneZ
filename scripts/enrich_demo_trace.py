from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = ROOT / "artifacts" / "traces"

ZONE_LAYOUT: dict[str, dict[str, float | str]] = {
    "hub": {"x": 455, "y": 325, "w": 170, "h": 120, "label": "Central Hub"},
    "Z1": {"x": 115, "y": 105, "w": 185, "h": 125, "label": "Downtown"},
    "Z2": {"x": 450, "y": 90, "w": 190, "h": 130, "label": "Hospital District"},
    "Z3": {"x": 780, "y": 120, "w": 185, "h": 125, "label": "East Logistics"},
    "Z4": {"x": 120, "y": 500, "w": 185, "h": 125, "label": "Market Zone"},
    "Z5": {"x": 455, "y": 530, "w": 190, "h": 125, "label": "Campus"},
    "Z6": {"x": 785, "y": 500, "w": 185, "h": 125, "label": "Suburb"},
}

POLICIES = ("improved", "heuristic", "random", "naive")


def zone_center(zone_id: str) -> dict[str, float]:
    zone = ZONE_LAYOUT.get(zone_id) or ZONE_LAYOUT["hub"]
    return {
        "x": float(zone["x"]) + float(zone["w"]) / 2,
        "y": float(zone["y"]) + float(zone["h"]) / 2,
    }


def sector_risk(sector: dict[str, Any]) -> float:
    risk = float(sector.get("congestion_score") or 0) * 0.35
    weather = sector.get("weather") or "clear"
    if weather in {"storm", "heavy_rain", "heavy_wind"}:
        risk += 0.45
    elif weather not in {"clear", "calm", None}:
        risk += 0.22
    if sector.get("is_no_fly"):
        risk += 0.7
    if sector.get("operations_paused"):
        risk += 0.35
    return min(1.0, round(risk, 3))


def wind_for_sector(sector: dict[str, Any], tick: int) -> int:
    weather = sector.get("weather") or "clear"
    base = {
        "clear": 10,
        "calm": 8,
        "moderate_wind": 28,
        "rain": 24,
        "heavy_rain": 38,
        "storm": 54,
        "heavy_wind": 62,
    }.get(weather, 18)
    return int(base + (tick % 5) * 2 + float(sector.get("congestion_score") or 0) * 6)


def route_color(corridor: str | None, risk: float, blocked: bool) -> str:
    if corridor == "safe":
        return "green"
    if blocked:
        return "red"
    if risk >= 0.55:
        return "yellow"
    return "purple"


def make_route_points(start: dict[str, float], end: dict[str, float], *, index: int, safe: bool, blocked: bool) -> list[dict[str, float]]:
    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    distance = math.hypot(dx, dy) or 1
    normal_x = -dy / distance
    normal_y = dx / distance
    bend = 80 + index * 14
    if safe:
        bend += 55
    if blocked:
        bend += 25
    return [
        {"x": round(start["x"], 1), "y": round(start["y"], 1)},
        {"x": round(start["x"] + dx * 0.32 + normal_x * bend, 1), "y": round(start["y"] + dy * 0.32 + normal_y * bend, 1)},
        {"x": round(start["x"] + dx * 0.66 - normal_x * bend * 0.45, 1), "y": round(start["y"] + dy * 0.66 - normal_y * bend * 0.45, 1)},
        {"x": round(end["x"], 1), "y": round(end["y"], 1)},
    ]


def enrich_frame(frame: dict[str, Any], index: int, payload: dict[str, Any]) -> dict[str, Any]:
    observation = frame.get("observation") or frame.get("state", {}).get("observation") or payload.get("initial_observation") or {}
    info = frame.get("info") or frame.get("state", {}).get("info") or {}
    tick = int(observation.get("step") or frame.get("tick") or index)
    sectors = observation.get("city", {}).get("sectors", [])
    sectors_by_id = {sector.get("zone_id"): sector for sector in sectors}
    active_no_fly = set(observation.get("city", {}).get("active_no_fly_zones") or [])
    held_zones = set(observation.get("city", {}).get("held_zones") or [])

    zone_visuals = []
    for zone_id, layout in ZONE_LAYOUT.items():
        sector = sectors_by_id.get(zone_id, {"zone_id": zone_id, "weather": "clear", "congestion_score": 0.0})
        risk = sector_risk(sector)
        warning = None
        if sector.get("is_no_fly") or zone_id in active_no_fly:
            warning = "restricted airspace"
        elif sector.get("operations_paused") or zone_id in held_zones:
            warning = "operations paused"
        elif sector.get("weather") not in {None, "clear", "calm"}:
            warning = f"{sector.get('weather')} weather"
        elif risk >= 0.45:
            warning = "congestion risk"
        zone_visuals.append(
            {
                "zone_id": zone_id,
                "label": layout["label"],
                "x": layout["x"],
                "y": layout["y"],
                "w": layout["w"],
                "h": layout["h"],
                "weather": sector.get("weather") or "clear",
                "wind_speed_kph": wind_for_sector(sector, tick),
                "risk_score": risk,
                "is_no_fly": bool(sector.get("is_no_fly") or zone_id in active_no_fly),
                "operations_paused": bool(sector.get("operations_paused") or zone_id in held_zones),
                "congestion_score": float(sector.get("congestion_score") or 0),
                "obstacle_warning": warning,
            }
        )

    route_segments = []
    telemetry = []
    action = frame.get("action") or {}
    action_name = action.get("action") or "initial_state"
    action_params = action.get("params") or {}
    for drone_index, drone in enumerate(observation.get("fleet", [])):
        current_zone = drone.get("current_zone") or "hub"
        target_zone = drone.get("target_zone") or current_zone
        current = zone_center(current_zone)
        target = zone_center(target_zone)
        zone_risk = next((zone["risk_score"] for zone in zone_visuals if zone["zone_id"] == current_zone), 0.0)
        target_blocked = any(zone["zone_id"] == target_zone and zone["is_no_fly"] for zone in zone_visuals)
        corridor = drone.get("active_corridor")
        safe = corridor == "safe" or action_name == "reroute"
        color = route_color(corridor, float(zone_risk), target_blocked)
        route_points = make_route_points(current, target, index=drone_index, safe=safe, blocked=target_blocked)
        if target_zone and target_zone != current_zone:
            route_segments.append(
                {
                    "drone_id": drone.get("drone_id"),
                    "points": route_points,
                    "route_color": color,
                    "route_type": "safe reroute" if safe else "planned corridor",
                    "risk_score": zone_risk,
                    "label": "avoids restricted/weather zones" if safe else "active mission path",
                }
            )

        battery = float(drone.get("battery") or 0)
        eta = drone.get("eta")
        speed = 0 if drone.get("status") in {"idle", "charging"} else max(12, 68 - int(eta or 0) * 6)
        altitude = 0 if drone.get("status") == "charging" else 85 + drone_index * 16 + (tick % 4) * 4
        assigned = drone.get("assigned_order_id")
        telemetry.append(
            {
                "drone_id": drone.get("drone_id"),
                "zone": current_zone,
                "target_zone": target_zone,
                "x": round(current["x"] + (drone_index % 3) * 18 - 18, 1),
                "y": round(current["y"] + (drone_index % 2) * 18 - 9, 1),
                "altitude_m": altitude,
                "speed_kph": speed,
                "battery": battery,
                "wind_exposure": next((zone["wind_speed_kph"] for zone in zone_visuals if zone["zone_id"] == current_zone), 0),
                "payload_kg": 0 if not assigned else 1 + (drone_index % 3),
                "gps_lock": battery > 8,
                "imu_status": "stable" if drone.get("health_risk") not in {"critical", "high"} else "degraded",
                "camera_status": "online",
                "lidar_status": "online" if drone_index % 2 == 0 else "standby",
                "thermal_status": "online" if drone.get("drone_type") in {"long_range_sensitive", "relay"} else "standby",
                "sensor_fusion_confidence": round(max(0.55, 0.98 - float(zone_risk) * 0.35 - max(0, 25 - battery) * 0.006), 2),
                "health_state": drone.get("health_risk") or "low",
                "assigned_order": assigned,
                "eta": eta,
                "current_action": action_name if action_params.get("drone_id") == drone.get("drone_id") else "monitor",
                "route_corridor": corridor or "direct",
                "route_risk": "blocked" if target_blocked else ("caution" if zone_risk >= 0.5 else "nominal"),
            }
        )

    orders = observation.get("orders", [])
    alerts = []
    for zone in zone_visuals:
        if zone["obstacle_warning"]:
            alerts.append(f"{zone['zone_id']}: {zone['obstacle_warning']}")
    for event in (observation.get("recent_events") or [])[-5:]:
        alerts.append(str(event))

    pending_orders = [order for order in orders if order.get("status") not in {"delivered", "canceled"}]
    urgent_orders = [order for order in pending_orders if order.get("priority") in {"urgent", "medical"}]
    visualization = {
        "source": "simulated visualization metadata derived from DroneZ environment trace",
        "frame_index": index,
        "zone_layout": zone_visuals,
        "route_segments": route_segments,
        "drone_telemetry": telemetry,
        "environment": {
            "dominant_weather": max((zone["weather"] for zone in zone_visuals), key=lambda item: sum(zone["weather"] == item for zone in zone_visuals), default="clear"),
            "max_wind_kph": max((zone["wind_speed_kph"] for zone in zone_visuals), default=0),
            "active_alerts": alerts[:10],
            "storm_zones": [zone["zone_id"] for zone in zone_visuals if zone["weather"] in {"storm", "heavy_rain", "heavy_wind"}],
            "restricted_zones": [zone["zone_id"] for zone in zone_visuals if zone["is_no_fly"]],
        },
        "control_layers": {
            "low_level_drone": {
                "pid_stability": "nominal",
                "state_estimation": "Kalman/sensor fusion simulated",
                "gps_navigation": "locked",
                "safety_rules": "active",
            },
            "high_level_rl_ai": {
                "decision": action_name,
                "mission_role": "assignment/reroute/charging/recovery optimization",
                "reward_signal": info.get("cumulative_reward", {}).get("total", frame.get("cumulative_reward", 0)),
            },
            "control_tower": {
                "fleet_monitoring": "active",
                "global_route_planning": "active",
                "human_override": "standby",
                "organization_policy": "safety-first medical logistics",
            },
        },
        "tower": {
            "dispatch_queue": [order.get("order_id") for order in pending_orders[:8]],
            "urgent_queue": [order.get("order_id") for order in urgent_orders[:5]],
            "override_status": "standby",
            "rl_recommendation": f"Next mission action: {action_name}",
            "fleet_health": {
                "active": sum(1 for drone in observation.get("fleet", []) if drone.get("status") not in {"idle", "charging"}),
                "idle": sum(1 for drone in observation.get("fleet", []) if drone.get("status") == "idle"),
                "charging": sum(1 for drone in observation.get("fleet", []) if drone.get("status") == "charging"),
                "low_battery": sum(1 for drone in observation.get("fleet", []) if float(drone.get("battery") or 0) < 25),
            },
        },
    }
    enriched = dict(frame)
    enriched["visualization"] = visualization
    return enriched


def enrich_payload(payload: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(payload)
    frames = payload.get("frames") or []
    enriched["visualization_schema"] = {
        "version": "1.0",
        "source": "derived from DroneZ traces for UI replay only",
        "note": "This is not real aircraft telemetry. It is simulated visualization metadata for a mission-level RL environment.",
    }
    enriched["frames"] = [enrich_frame(frame, index, payload) for index, frame in enumerate(frames)]
    return enriched


def enrich_one(task: str, policy: str) -> Path:
    source = TRACE_DIR / f"{task}_{policy}_trace.json"
    if not source.exists():
        raise FileNotFoundError(f"Missing trace: {source}")
    payload = json.loads(source.read_text())
    destination = TRACE_DIR / f"{task}_{policy}_enriched.json"
    destination.write_text(json.dumps(enrich_payload(payload), indent=2))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Derive UI visualization metadata from DroneZ replay traces.")
    parser.add_argument("--task", default="demo")
    parser.add_argument("--policy", default="all", help="Policy id or 'all'.")
    args = parser.parse_args()

    policies = POLICIES if args.policy == "all" else (args.policy,)
    written = [enrich_one(args.task, policy) for policy in policies]
    print(json.dumps({"written": [str(path) for path in written]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
