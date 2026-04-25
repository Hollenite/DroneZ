from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..policies.baseline import _build_candidate_actions


ACTION_SCHEMAS: dict[str, tuple[str, ...]] = {
    "assign_delivery": ("drone_id", "order_id"),
    "reroute": ("drone_id", "corridor"),
    "return_to_charge": ("drone_id", "station_id"),
    "reserve_charger": ("drone_id", "station_id"),
    "delay_order": ("order_id",),
    "prioritize_order": ("order_id",),
    "swap_assignments": ("drone_a", "drone_b"),
    "attempt_delivery": ("drone_id", "mode"),
    "fallback_to_locker": ("order_id", "locker_id"),
    "hold_fleet": ("zone_id",),
    "resume_operations": ("zone_id",),
}


@dataclass(frozen=True)
class ActionParseResult:
    action: dict[str, Any]
    valid_json: bool
    repaired: bool = False
    used_candidate_choice: bool = False
    error_code: str | None = None
    notes: list[str] = field(default_factory=list)
    raw_object: Any | None = None

    @property
    def valid_action_shape(self) -> bool:
        return self.action.get("action") != "__invalid__"


def compact_observation_summary(observation: dict[str, Any]) -> str:
    """Create a short LLM prompt summary that avoids the giant human-readable trace block."""

    fleet_lines = []
    for drone in observation.get("fleet", []):
        fleet_lines.append(
            "|".join(
                [
                    str(drone.get("drone_id")),
                    f"type={drone.get('drone_type')}",
                    f"status={drone.get('status')}",
                    f"battery={drone.get('battery')}",
                    f"zone={drone.get('current_zone')}",
                    f"assigned={drone.get('assigned_order_id')}",
                    f"eta={drone.get('eta')}",
                    f"target={drone.get('target_zone')}",
                    f"corridor={drone.get('active_corridor')}",
                ]
            )
        )

    order_lines = []
    for order in observation.get("orders", []):
        if order.get("status") in {"delivered", "canceled"}:
            continue
        order_lines.append(
            "|".join(
                [
                    str(order.get("order_id")),
                    f"priority={order.get('priority')}",
                    f"status={order.get('status')}",
                    f"zone={order.get('zone_id')}",
                    f"deadline={order.get('deadline')}",
                    f"recipient={order.get('recipient_availability')}",
                    f"assigned={order.get('assigned_drone_id')}",
                    f"weight={order.get('package_weight')}",
                ]
            )
        )

    sector_lines = []
    for sector in observation.get("city", {}).get("sectors", []):
        hazard_bits = []
        if sector.get("is_no_fly"):
            hazard_bits.append("no_fly")
        if sector.get("operations_paused"):
            hazard_bits.append("paused")
        if sector.get("weather") not in {None, "clear"}:
            hazard_bits.append(f"weather={sector.get('weather')}")
        if float(sector.get("congestion_score", 0.0)) >= 0.7:
            hazard_bits.append(f"congestion={sector.get('congestion_score')}")
        if hazard_bits or sector.get("zone_id") == "hub":
            sector_lines.append(f"{sector.get('zone_id')}:{','.join(hazard_bits) or 'clear'}")

    charger_lines = []
    for station in observation.get("charging", []):
        charger_lines.append(
            f"{station.get('station_id')}|occupied={station.get('occupied_slots')}/{station.get('capacity')}|queue={station.get('queue_size')}"
        )

    return "\n".join(
        [
            f"TASK={observation.get('task_id')} STEP={observation.get('step')}/{observation.get('max_steps')} PENDING={len(order_lines)}",
            "FLEET:",
            *fleet_lines[:8],
            "ORDERS:",
            *(order_lines[:10] or ["none"]),
            "HAZARDS:",
            *(sector_lines[:8] or ["none"]),
            "CHARGERS:",
            *(charger_lines[:5] or ["none"]),
        ]
    )


def build_candidate_actions(observation: dict[str, Any], *, max_candidates: int = 8) -> list[dict[str, Any]]:
    candidates = _build_candidate_actions(observation, allow_reroute=True, safe_only=True)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = json.dumps(candidate, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= max_candidates:
            break
    return deduped


def build_action_prompt(
    observation: dict[str, Any],
    *,
    candidate_choice: bool = False,
    max_candidates: int = 8,
) -> str:
    schema_lines = [
        f"- {name}: params={list(params)}"
        for name, params in ACTION_SCHEMAS.items()
        if name in set(observation.get("action_reminder", ACTION_SCHEMAS))
    ]
    candidates = build_candidate_actions(observation, max_candidates=max_candidates)
    candidate_lines = [f"{index}. {json.dumps(action, sort_keys=True)}" for index, action in enumerate(candidates, start=1)]

    if candidate_choice:
        response_rule = (
            'Return exactly one JSON object and nothing else. Prefer a numbered candidate: {"choice": 1}. '
            'If you cannot use a choice, return exactly one full action JSON object.'
        )
    else:
        response_rule = (
            'Return exactly one JSON object and nothing else: '
            '{"action": "assign_delivery", "params": {"drone_id": "FA-1", "order_id": "O1"}}'
        )
    return "\n".join(
        [
            "You are DroneZ, an LLM mission-level drone fleet controller.",
            response_rule,
            "Do not explain. Do not use markdown. Do not invent IDs.",
            "",
            "COMPACT_OBSERVATION:",
            compact_observation_summary(observation),
            "",
            "ACTION_SCHEMA:",
            *schema_lines,
            "",
            "VALID_CANDIDATES:",
            *(candidate_lines or ['1. {"action": "prioritize_order", "params": {"order_id": "O1"}}']),
        ]
    )


def parse_llm_action(
    text: str,
    *,
    observation: dict[str, Any] | None = None,
    candidate_actions: list[dict[str, Any]] | None = None,
) -> ActionParseResult:
    candidate_actions = candidate_actions or (build_candidate_actions(observation) if observation else [])
    parsed, valid_json, notes = _parse_first_object(text)
    if not valid_json:
        return ActionParseResult(
            action=_invalid_action(),
            valid_json=False,
            error_code="no_json_object",
            notes=notes,
            raw_object=None,
        )

    action, repaired, choice_used, repair_notes = _repair_action(parsed, observation=observation, candidate_actions=candidate_actions)
    notes.extend(repair_notes)
    error_code = None if action.get("action") != "__invalid__" else "unrepairable_action"
    return ActionParseResult(
        action=action,
        valid_json=True,
        repaired=repaired,
        used_candidate_choice=choice_used,
        error_code=error_code,
        notes=notes,
        raw_object=parsed,
    )


def _parse_first_object(text: str) -> tuple[Any | None, bool, list[str]]:
    notes: list[str] = []
    candidate = str(text or "").strip()
    if not candidate:
        return None, False, ["empty_output"]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", candidate, flags=re.DOTALL | re.IGNORECASE)
    search_space = fenced + [candidate]
    if fenced:
        notes.append("stripped_markdown_fence")

    for item in search_space:
        for obj_text in _balanced_json_objects(item):
            for parser_name, parser in (("json", json.loads), ("literal_eval", ast.literal_eval)):
                try:
                    parsed = parser(obj_text)
                    if parser_name == "literal_eval":
                        notes.append("parsed_python_literal")
                    return parsed, True, notes
                except Exception:
                    continue
    return None, False, notes or ["json_parse_failed"]


def _balanced_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote_char = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _repair_action(
    parsed: Any,
    *,
    observation: dict[str, Any] | None,
    candidate_actions: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool, bool, list[str]]:
    notes: list[str] = []
    if not isinstance(parsed, dict):
        return _invalid_action(), False, False, ["parsed_object_not_dict"]

    if "choice" in parsed:
        choice = _coerce_int(parsed.get("choice"))
        if choice is not None and 1 <= choice <= len(candidate_actions):
            return candidate_actions[choice - 1], True, True, [f"candidate_choice_{choice}"]
        return _invalid_action(), False, True, ["invalid_candidate_choice"]

    raw_action = parsed.get("action")
    raw_params = parsed.get("params")
    if isinstance(raw_action, dict):
        nested = raw_action
        raw_action = nested.get("action") or nested.get("action_name") or nested.get("name")
        raw_params = nested.get("params") or nested.get("arguments") or raw_params
        notes.append("unwrapped_nested_action")

    action_name = raw_action or parsed.get("action_name") or parsed.get("name") or parsed.get("command") or parsed.get("tool")
    params = raw_params or parsed.get("arguments") or {}
    if not isinstance(params, dict):
        params = {}
        notes.append("reset_non_dict_params")

    if not action_name or action_name not in ACTION_SCHEMAS:
        matching_candidate = _candidate_matching_action(candidate_actions, str(action_name or ""))
        if matching_candidate:
            return matching_candidate, True, False, notes + ["repaired_unknown_action_from_candidate"]
        return _invalid_action(), False, False, notes + ["unknown_action"]

    repaired = False
    missing = [key for key in ACTION_SCHEMAS[action_name] if key not in params or params.get(key) in {None, ""}]
    if missing:
        filled = _fill_missing_params(action_name, params, missing, observation, candidate_actions)
        if filled is None:
            return _invalid_action(), False, False, notes + [f"missing_params:{','.join(missing)}"]
        params = filled
        repaired = True
        notes.append("filled_missing_params_from_observation")

    normalized = {"action": action_name, "params": {key: params[key] for key in ACTION_SCHEMAS[action_name]}}
    return normalized, repaired or bool(notes), False, notes


def _fill_missing_params(
    action_name: str,
    params: dict[str, Any],
    missing: list[str],
    observation: dict[str, Any] | None,
    candidate_actions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    matching = _candidate_matching_action(candidate_actions, action_name)
    if matching:
        merged = dict(matching["params"])
        merged.update({key: value for key, value in params.items() if value not in {None, ""}})
        return merged

    if observation is None:
        return None

    merged = dict(params)
    if action_name in {"prioritize_order", "delay_order"}:
        order = _first_open_order(observation)
        if order:
            merged.setdefault("order_id", order["order_id"])
    elif action_name == "attempt_delivery":
        drone = _ready_drone(observation)
        if drone:
            merged.setdefault("drone_id", drone["drone_id"])
            merged.setdefault("mode", "handoff")
    elif action_name == "fallback_to_locker":
        order = _first_unavailable_order(observation) or _first_open_order(observation)
        if order:
            merged.setdefault("order_id", order["order_id"])
            merged.setdefault("locker_id", f"L-{order['zone_id']}")
    elif action_name in {"hold_fleet", "resume_operations"}:
        zone_id = _first_zone(observation)
        if zone_id:
            merged.setdefault("zone_id", zone_id)
    elif action_name in {"reserve_charger", "return_to_charge"}:
        drone = _first_available_drone(observation)
        station = next(iter(observation.get("charging", [])), None)
        if drone and station:
            merged.setdefault("drone_id", drone["drone_id"])
            merged.setdefault("station_id", station["station_id"])
    elif action_name == "reroute":
        drone = _first_assigned_drone(observation)
        if drone:
            merged.setdefault("drone_id", drone["drone_id"])
            merged.setdefault("corridor", "safe")
    elif action_name == "assign_delivery":
        drone = _first_available_drone(observation)
        order = _first_open_order(observation)
        if drone and order:
            merged.setdefault("drone_id", drone["drone_id"])
            merged.setdefault("order_id", order["order_id"])

    if all(key in merged and merged[key] not in {None, ""} for key in ACTION_SCHEMAS[action_name]):
        return merged
    return None


def _candidate_matching_action(candidate_actions: list[dict[str, Any]], action_name: str) -> dict[str, Any] | None:
    return next((candidate for candidate in candidate_actions if candidate.get("action") == action_name), None)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except Exception:
        return None


def _invalid_action() -> dict[str, Any]:
    return {"action": "__invalid__", "params": {}}


def _first_open_order(observation: dict[str, Any]) -> dict[str, Any] | None:
    return next((order for order in observation.get("orders", []) if order.get("status") not in {"delivered", "canceled"}), None)


def _first_unavailable_order(observation: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            order
            for order in observation.get("orders", [])
            if order.get("status") not in {"delivered", "canceled"} and order.get("recipient_availability") == "unavailable"
        ),
        None,
    )


def _ready_drone(observation: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            drone
            for drone in observation.get("fleet", [])
            if drone.get("drone_type") != "relay" and drone.get("assigned_order_id") and drone.get("eta") == 0
        ),
        None,
    )


def _first_available_drone(observation: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            drone
            for drone in observation.get("fleet", [])
            if drone.get("drone_type") != "relay" and drone.get("status") in {"idle", "holding"}
        ),
        None,
    )


def _first_assigned_drone(observation: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            drone
            for drone in observation.get("fleet", [])
            if drone.get("drone_type") != "relay" and drone.get("assigned_order_id")
        ),
        None,
    )


def _first_zone(observation: dict[str, Any]) -> str | None:
    return next((sector.get("zone_id") for sector in observation.get("city", {}).get("sectors", []) if sector.get("zone_id")), None)
