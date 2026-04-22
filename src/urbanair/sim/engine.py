from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..enums import ActionType, DroneStatus, DropMode, RecipientAvailability
from ..models import (
    ChargingStationState,
    DroneState,
    EmergencyEvent,
    EnvironmentObservation,
    OrderState,
    PolicyNotice,
    RewardBreakdown,
    SectorState,
    StepResult,
    TaskConfig,
)
from ..utils.seeding import SeedBundle, create_seed_bundle, derive_seed
from .city import build_city
from .delivery_logic import resolve_delivery_attempts
from .disruptions import evolve_disruptions
from .scripted_events import apply_scripted_events
from .fleet import advance_fleet_tick, apply_relay_effect, assign_order, build_fleet, send_to_charge
from .hidden_dynamics import HiddenState, build_hidden_state, update_hidden_state
from .orders import build_orders, tick_order_deadlines, update_customer_availability

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
TASKS_DIR = CONFIG_DIR / "tasks"


def _merge_breakdowns(current: RewardBreakdown, delta: RewardBreakdown) -> RewardBreakdown:
    positive = {
        key: current.positive.get(key, 0.0) + delta.positive.get(key, 0.0)
        for key in set(current.positive) | set(delta.positive)
    }
    negative = {
        key: current.negative.get(key, 0.0) + delta.negative.get(key, 0.0)
        for key in set(current.negative) | set(delta.negative)
    }
    return RewardBreakdown.from_components(positive=positive, negative=negative)


def _build_observation(state: SimulatorState) -> EnvironmentObservation:
    return EnvironmentObservation(
        time_step=state.tick,
        horizon=state.task_config.horizon,
        task_id=state.task_config.task_id,
        fleet=[drone.model_copy(deep=True) for drone in state.fleet],
        orders=[order.model_copy(deep=True) for order in state.orders],
        sectors=[sector.model_copy(deep=True) for sector in state.sectors],
        charging_stations=[station.model_copy(deep=True) for station in state.charging_stations],
        policy_notices=[notice.model_copy(deep=True) for notice in state.policy_notices],
        emergency_events=[event.model_copy(deep=True) for event in state.emergency_events],
        recent_events=list(state.recent_events),
        warnings=[event for event in state.recent_events if "critical" in event.lower() or "failed" in event.lower()],
        allowed_actions=[action.value for action in ActionType],
        reward_summary=state.cumulative_reward,
    )


@dataclass
class SimulatorState:
    task_config: TaskConfig
    seed_bundle: SeedBundle
    tick: int = 0
    fleet: list[DroneState] = field(default_factory=list)
    orders: list[OrderState] = field(default_factory=list)
    sectors: list[SectorState] = field(default_factory=list)
    charging_stations: list[ChargingStationState] = field(default_factory=list)
    policy_notices: list[PolicyNotice] = field(default_factory=list)
    emergency_events: list[EmergencyEvent] = field(default_factory=list)
    hidden_state: HiddenState | None = None
    recent_events: list[str] = field(default_factory=list)
    resolved_order_ids: set[str] = field(default_factory=set)
    next_order_index: int = 1
    cumulative_reward: RewardBreakdown = field(default_factory=RewardBreakdown)
    reward_inputs: dict[str, float] = field(default_factory=dict)
    triggered_scripted_events: list[str] = field(default_factory=list)
    zone_hold_reasons: dict[str, str] = field(default_factory=dict)
    pending_recovery_orders: set[str] = field(default_factory=set)
    just_recovered_orders: set[str] = field(default_factory=set)
    just_reserved_chargers: int = 0
    just_delay_actions: int = 0
    just_hold_actions: int = 0
    just_resume_actions: int = 0
    just_fallback_actions: int = 0
    just_attempt_actions: int = 0
    just_idle_with_pending: int = 0
    just_abandoned_urgent: int = 0
    just_balanced_charge: int = 0
    just_restricted_zone_events: int = 0
    just_recovery_successes: int = 0
    just_low_reattempt: int = 0
    just_fleet_utilization: int = 0
    just_energy_efficient: int = 0
    just_policy_compliance: int = 0
    just_overloaded_assignments: int = 0
    just_congestion_penalty: int = 0
    just_unnecessary_delay: int = 0
    just_unnecessary_hold: int = 0
    just_reserved_charge_queue_miss: int = 0
    charging_queue_order: list[str] = field(default_factory=list)
    delivery_attempt_required: set[str] = field(default_factory=set)
    auto_attempt_enabled: bool = False


class SimulatorEngine:
    def __init__(self, task_configs: dict[str, TaskConfig], fleet_profiles, reward_weights) -> None:
        self.task_configs = task_configs
        self.fleet_profiles = fleet_profiles
        self.reward_weights = reward_weights

    @classmethod
    def from_repo_configs(cls) -> "SimulatorEngine":
        task_configs = {
            path.stem: TaskConfig.model_validate(yaml.safe_load(path.read_text()))
            for path in TASKS_DIR.glob("*.yaml")
        }
        fleet_profiles = yaml.safe_load((CONFIG_DIR / "fleet_profiles.yaml").read_text())
        reward_weights = yaml.safe_load((CONFIG_DIR / "reward_weights.yaml").read_text())
        from ..models import FleetProfilesConfig, RewardWeightsConfig

        return cls(
            task_configs=task_configs,
            fleet_profiles=FleetProfilesConfig.model_validate(fleet_profiles).profiles,
            reward_weights=RewardWeightsConfig.model_validate(reward_weights),
        )

    def reset(self, task_id: str) -> SimulatorState:
        task_config = self.task_configs[task_id]
        seed_bundle = create_seed_bundle(task_config.seed)
        city_rng = create_seed_bundle(derive_seed(seed_bundle.seed, 1)).rng
        fleet_rng = create_seed_bundle(derive_seed(seed_bundle.seed, 2)).rng
        order_rng = create_seed_bundle(derive_seed(seed_bundle.seed, 3)).rng
        hidden_rng = create_seed_bundle(derive_seed(seed_bundle.seed, 4)).rng

        sectors, charging_stations, policy_notices, emergency_events = build_city(task_config, city_rng)
        fleet = build_fleet(task_config, self.fleet_profiles, fleet_rng)
        orders = build_orders(task_config, sectors, order_rng)
        hidden_state = build_hidden_state(task_config, sectors, hidden_rng)
        update_hidden_state(hidden_state, sectors)
        apply_relay_effect(fleet, hidden_state.failure_prone_zones)

        return SimulatorState(
            task_config=task_config,
            seed_bundle=seed_bundle,
            tick=0,
            fleet=fleet,
            orders=orders,
            sectors=sectors,
            charging_stations=charging_stations,
            policy_notices=policy_notices,
            emergency_events=emergency_events,
            hidden_state=hidden_state,
            recent_events=[f"Task {task_id} initialized with seed {task_config.seed}."],
            resolved_order_ids=set(),
            next_order_index=len(orders) + 1,
        )

    def step(self, state: SimulatorState, action: dict[str, Any] | None = None) -> StepResult:
        events: list[str] = []
        state.just_recovered_orders.clear()
        state.just_reserved_chargers = 0
        state.just_delay_actions = 0
        state.just_hold_actions = 0
        state.just_resume_actions = 0
        state.just_fallback_actions = 0
        state.just_attempt_actions = 0
        state.just_idle_with_pending = 0
        state.just_abandoned_urgent = 0
        state.just_balanced_charge = 0
        state.just_restricted_zone_events = 0
        state.just_recovery_successes = 0
        state.just_low_reattempt = 0
        state.just_fleet_utilization = 0
        state.just_energy_efficient = 0
        state.just_policy_compliance = 0
        state.just_overloaded_assignments = 0
        state.just_congestion_penalty = 0
        state.just_unnecessary_delay = 0
        state.just_unnecessary_hold = 0
        state.just_reserved_charge_queue_miss = 0
        reward_inputs = {
            "deliveries_completed": 0.0,
            "urgent_successes": 0.0,
            "failed_attempts": 0.0,
            "deadline_misses": 0.0,
            "critical_battery": 0.0,
        }

        # 1-2. Validate/apply simplified operational decision
        if action:
            events.extend(self._apply_action(state, action))

        # 3. Advance time by one tick
        state.tick += 1

        # 4. Update drone positions / statuses
        events.extend(advance_fleet_tick(state.fleet, state.sectors))

        # 5. Resolve charging events
        self._resolve_charging_pressure(state)

        # 6. Resolve delivery attempts and outcomes
        delivery_events, delivered_ids, recovery_actions = resolve_delivery_attempts(
            state.fleet,
            state.orders,
            state.sectors,
            state.hidden_state.failure_prone_zones if state.hidden_state else set(),
            state.task_config.failed_drop_probability,
            create_seed_bundle(derive_seed(state.seed_bundle.seed, 20 + state.tick)).rng,
            auto_attempt_enabled=state.auto_attempt_enabled,
            delivery_attempt_required=state.delivery_attempt_required,
        )
        events.extend(delivery_events)
        state.resolved_order_ids.update(delivered_ids)
        state.pending_recovery_orders.update(recovery_actions.keys())
        reward_inputs["deliveries_completed"] += float(len(delivered_ids))
        reward_inputs["failed_attempts"] += float(len(recovery_actions))
        reward_inputs["urgent_successes"] += float(
            sum(1 for order in state.orders if order.order_id in delivered_ids and order.priority.value in {"urgent", "medical"})
        )
        state.just_recovery_successes = len(state.just_recovered_orders)
        state.just_low_reattempt = sum(
            1 for order in state.orders if order.order_id in state.just_recovered_orders and order.retry_count <= 1
        )

        # 7. Update customer availability windows
        order_events = update_customer_availability(
            state.orders,
            state.resolved_order_ids,
            state.hidden_state.no_show_zones if state.hidden_state else set(),
            {order.order_id: order.zone_id for order in state.orders},
            create_seed_bundle(derive_seed(state.seed_bundle.seed, 40 + state.tick)).rng,
        )
        events.extend(order_events)
        deadline_events = tick_order_deadlines(state.orders, state.resolved_order_ids)
        events.extend(deadline_events)
        reward_inputs["deadline_misses"] += float(len(deadline_events))

        # 8. Trigger or evolve disruptions
        scripted_events, scripted_next_order_index, triggered_scripted_events = apply_scripted_events(
            state.task_config.scripted_events,
            state.tick,
            state.sectors,
            state.policy_notices,
            state.orders,
            state.next_order_index,
        )
        disruption_events, next_order_index = evolve_disruptions(
            state.task_config,
            state.tick,
            state.sectors,
            state.charging_stations,
            state.policy_notices,
            state.emergency_events,
            state.orders,
            scripted_next_order_index,
            state.hidden_state.sector_weather_bias if state.hidden_state else {},
            create_seed_bundle(derive_seed(state.seed_bundle.seed, 60 + state.tick)).rng,
            triggered_scripted_events,
        )
        state.next_order_index = next_order_index
        state.triggered_scripted_events = triggered_scripted_events
        events.extend(scripted_events)
        events.extend(disruption_events)

        # 9. Update hidden dynamics
        if state.hidden_state is not None:
            update_hidden_state(state.hidden_state, state.sectors)
            apply_relay_effect(state.fleet, state.hidden_state.failure_prone_zones)

        # 10. Compute terminal conditions
        done = self._is_done(state)

        # 11. Produce reward breakdown
        pending_orders = [order for order in state.orders if order.order_id not in state.resolved_order_ids and order.status != "canceled"]
        state.just_fleet_utilization = sum(
            1 for drone in state.fleet if drone.drone_type.value != "relay" and drone.status in {DroneStatus.ASSIGNED, DroneStatus.IN_FLIGHT}
        )
        state.just_energy_efficient = sum(
            1 for drone in state.fleet if drone.status == DroneStatus.IN_FLIGHT and drone.battery > 25
        )
        state.just_idle_with_pending = sum(
            1 for drone in state.fleet if pending_orders and drone.drone_type.value != "relay" and drone.status == DroneStatus.IDLE
        )
        state.just_policy_compliance = int(
            all(not (drone.status == DroneStatus.IN_FLIGHT and any(
                sector.zone_id == drone.current_zone and (sector.is_no_fly or sector.operations_paused)
                for sector in state.sectors
            )) for drone in state.fleet)
        )
        state.just_congestion_penalty = sum(
            1
            for order in state.orders
            if order.assigned_drone_id is not None and any(
                sector.zone_id == order.zone_id and sector.congestion_score >= 0.4 for sector in state.sectors
            )
        )
        reward_inputs["critical_battery"] += float(sum(1 for drone in state.fleet if drone.battery <= 12))
        breakdown = self._build_reward_breakdown(state, reward_inputs)
        state.cumulative_reward = _merge_breakdowns(state.cumulative_reward, breakdown)
        state.reward_inputs = reward_inputs

        # 12. Build next observation inputs for the later environment layer
        state.recent_events = (state.recent_events + events)[-12:]

        return StepResult(
            observation=_build_observation(state),
            reward=breakdown.total,
            done=done,
            info={
                "resolved_order_ids": sorted(state.resolved_order_ids),
                "reward_inputs": reward_inputs,
                "recovery_actions": recovery_actions,
                "triggered_scripted_events": list(state.triggered_scripted_events),
                "pending_recovery_orders": sorted(state.pending_recovery_orders),
                "delivery_attempt_required": sorted(state.delivery_attempt_required),
                "zone_holds": dict(state.zone_hold_reasons),
            },
            reward_breakdown=breakdown,
        )

    def _apply_action(self, state: SimulatorState, action: dict[str, Any]) -> list[str]:
        action_type = ActionType(action["action_type"])
        events: list[str] = []

        if action_type == ActionType.ASSIGN_DELIVERY:
            order = next((item for item in state.orders if item.order_id == action["order_id"] and item.order_id not in state.resolved_order_ids), None)
            drone = next((item for item in state.fleet if item.drone_id == action["drone_id"]), None)
            if order and drone and drone.status in {DroneStatus.IDLE, DroneStatus.HOLDING}:
                drone.hold_reason = None
                order.assigned_drone_id = drone.drone_id
                order.status = "assigned"
                if order.order_id in state.pending_recovery_orders:
                    state.pending_recovery_orders.discard(order.order_id)
                    state.just_recovered_orders.add(order.order_id)
                state.just_overloaded_assignments += int(sum(1 for item in state.orders if item.assigned_drone_id == drone.drone_id) > 0)
                state.delivery_attempt_required.discard(order.order_id)
                events.extend(assign_order(drone, order.order_id, order.zone_id, state.sectors))

        elif action_type == ActionType.RETURN_TO_CHARGE:
            drone = next((item for item in state.fleet if item.drone_id == action["drone_id"]), None)
            station = next((item for item in state.charging_stations if item.station_id == action["station_id"]), None)
            if drone and station:
                if drone.assigned_order_id:
                    order = next((item for item in state.orders if item.order_id == drone.assigned_order_id), None)
                    if order is not None:
                        order.assigned_drone_id = None
                        order.status = "deferred"
                        state.pending_recovery_orders.add(order.order_id)
                        state.just_abandoned_urgent += int(order.priority.value in {"urgent", "medical"})
                    drone.assigned_order_id = None
                    drone.target_zone = None
                events.extend(send_to_charge(drone, station))
                if drone.hold_reason == "charging_queue" and drone.drone_id not in state.charging_queue_order:
                    state.charging_queue_order.append(drone.drone_id)

        elif action_type == ActionType.RESERVE_CHARGER:
            drone = next((item for item in state.fleet if item.drone_id == action["drone_id"]), None)
            station = next((item for item in state.charging_stations if item.station_id == action["station_id"]), None)
            if drone and station:
                state.just_reserved_chargers += 1
                events.extend(send_to_charge(drone, station, reserve_only=True))

        elif action_type == ActionType.PRIORITIZE_ORDER:
            order = next((item for item in state.orders if item.order_id == action["order_id"]), None)
            if order:
                order.deadline = max(1, order.deadline - 1)
                events.append(f"{order.order_id} was prioritized.")

        elif action_type == ActionType.DELAY_ORDER:
            order = next((item for item in state.orders if item.order_id == action["order_id"]), None)
            if order:
                state.just_delay_actions += 1
                state.just_unnecessary_delay += int(order.priority.value in {"urgent", "medical"})
                order.deadline += 1
                order.status = "deferred"
                if order.assigned_drone_id:
                    drone = next((item for item in state.fleet if item.drone_id == order.assigned_drone_id), None)
                    if drone is not None:
                        drone.assigned_order_id = None
                        drone.target_zone = None
                        if drone.status != DroneStatus.CHARGING:
                            drone.status = DroneStatus.HOLDING
                            drone.hold_reason = "order_delayed"
                    order.assigned_drone_id = None
                state.pending_recovery_orders.add(order.order_id)
                events.append(f"{order.order_id} was delayed for replanning.")

        elif action_type == ActionType.ATTEMPT_DELIVERY:
            drone = next((item for item in state.fleet if item.drone_id == action["drone_id"]), None)
            if drone and drone.assigned_order_id:
                state.just_attempt_actions += 1
                state.delivery_attempt_required.add(drone.assigned_order_id)
                if action.get("mode") == "locker" or action.get("mode") == "handoff":
                    order = next((item for item in state.orders if item.order_id == drone.assigned_order_id), None)
                    if order is not None:
                        order.drop_mode = DropMode.LOCKER if action["mode"] == "locker" else DropMode.HANDOFF
                events.append(f"{drone.drone_id} committed to an explicit delivery attempt.")

        elif action_type == ActionType.FALLBACK_TO_LOCKER:
            order = next((item for item in state.orders if item.order_id == action["order_id"]), None)
            if order:
                state.just_fallback_actions += 1
                order.drop_mode = DropMode.LOCKER
                order.recipient_availability = RecipientAvailability.AVAILABLE
                order.status = "locker_fallback"
                state.pending_recovery_orders.discard(order.order_id)
                state.just_recovered_orders.add(order.order_id)
                events.append(f"{order.order_id} moved to locker fallback at {action['locker_id']}.")

        elif action_type == ActionType.HOLD_FLEET:
            zone_id = action["zone_id"]
            state.zone_hold_reasons[zone_id] = "manual_hold"
            state.just_hold_actions += 1
            sector = next((item for item in state.sectors if item.zone_id == zone_id), None)
            if sector is not None:
                sector.operations_paused = True
            for drone in state.fleet:
                if drone.current_zone == zone_id and drone.status in {DroneStatus.IDLE, DroneStatus.HOLDING}:
                    drone.status = DroneStatus.HOLDING
                    drone.hold_reason = "zone_hold"
            events.append(f"Fleet operations paused in {zone_id}.")

        elif action_type == ActionType.RESUME_OPERATIONS:
            zone_id = action["zone_id"]
            state.zone_hold_reasons.pop(zone_id, None)
            state.just_resume_actions += 1
            sector = next((item for item in state.sectors if item.zone_id == zone_id), None)
            if sector is not None:
                sector.operations_paused = False
            for drone in state.fleet:
                if drone.current_zone == zone_id and drone.status == DroneStatus.HOLDING and drone.hold_reason == "zone_hold":
                    drone.status = DroneStatus.IDLE
                    drone.hold_reason = None
            events.append(f"Fleet operations resumed in {zone_id}.")

        return events

    def _resolve_charging_pressure(self, state: SimulatorState) -> None:
        for station in state.charging_stations:
            charging_drones = [drone for drone in state.fleet if drone.status == DroneStatus.CHARGING and drone.reserved_station_id == station.station_id]
            station.occupied_slots = min(station.capacity, len(charging_drones))

            queued_drones = [
                drone
                for drone in state.fleet
                if drone.status == DroneStatus.HOLDING and drone.hold_reason == "charging_queue" and drone.reserved_station_id == station.station_id
            ]
            queued_drones.sort(key=lambda drone: state.charging_queue_order.index(drone.drone_id) if drone.drone_id in state.charging_queue_order else len(state.charging_queue_order))

            while station.occupied_slots < station.capacity and queued_drones:
                next_drone = queued_drones.pop(0)
                next_drone.status = DroneStatus.CHARGING
                next_drone.hold_reason = None
                station.occupied_slots += 1
                if station.queue_size > 0:
                    station.queue_size -= 1

            station.reserved_drone_ids = [
                drone_id
                for drone_id in station.reserved_drone_ids
                if any(drone.drone_id == drone_id and drone.reserved_station_id == station.station_id for drone in state.fleet)
            ]
            if station.queue_size == 0:
                state.charging_queue_order = [
                    drone_id for drone_id in state.charging_queue_order if any(
                        drone.drone_id == drone_id and drone.status == DroneStatus.HOLDING and drone.hold_reason == "charging_queue"
                        for drone in state.fleet
                    )
                ]

            if station.occupied_slots <= max(1, station.capacity - 1):
                state.just_balanced_charge += 1

    def _is_done(self, state: SimulatorState) -> bool:
        unresolved_orders = [order for order in state.orders if order.order_id not in state.resolved_order_ids and order.status != "canceled"]
        no_viable_drones = not any(
            drone.status != DroneStatus.OFFLINE
            and drone.battery > 10
            and not (drone.status == DroneStatus.HOLDING and drone.hold_reason == "zone_hold")
            for drone in state.fleet
        )
        return not unresolved_orders or state.tick >= state.task_config.horizon or no_viable_drones

    def _build_reward_breakdown(self, state: SimulatorState, reward_inputs: dict[str, float]) -> RewardBreakdown:
        positive: dict[str, float] = {}
        negative: dict[str, float] = {}
        if reward_inputs["deliveries_completed"]:
            positive["on_time_delivery"] = reward_inputs["deliveries_completed"] * self.reward_weights.positive["on_time_delivery"]
        if reward_inputs["urgent_successes"]:
            positive["urgent_delivery_completion"] = reward_inputs["urgent_successes"] * self.reward_weights.positive["urgent_delivery_completion"]
        if reward_inputs["failed_attempts"]:
            negative["failed_delivery_attempt"] = reward_inputs["failed_attempts"] * self.reward_weights.negative["failed_delivery_attempt"]
        if reward_inputs["deadline_misses"]:
            negative["missed_delivery_window"] = reward_inputs["deadline_misses"] * self.reward_weights.negative["missed_delivery_window"]
        if reward_inputs["critical_battery"]:
            negative["battery_critical_state"] = reward_inputs["critical_battery"] * self.reward_weights.negative["battery_critical_state"]
        if state.just_balanced_charge:
            positive["balanced_charging_usage"] = state.just_balanced_charge * self.reward_weights.positive["balanced_charging_usage"]
        if state.just_recovery_successes:
            positive["disruption_recovery"] = state.just_recovery_successes * self.reward_weights.positive["disruption_recovery"]
        if state.just_low_reattempt:
            positive["low_reattempt_rate"] = state.just_low_reattempt * self.reward_weights.positive["low_reattempt_rate"]
        if state.just_fleet_utilization:
            positive["fleet_utilization"] = state.just_fleet_utilization * self.reward_weights.positive["fleet_utilization"]
        if state.just_energy_efficient:
            positive["energy_efficiency"] = state.just_energy_efficient * self.reward_weights.positive["energy_efficiency"]
        if state.just_policy_compliance:
            positive["regulatory_compliance"] = state.just_policy_compliance * self.reward_weights.positive["regulatory_compliance"]
        if state.just_idle_with_pending:
            negative["idle_with_pending_orders"] = state.just_idle_with_pending * self.reward_weights.negative["idle_with_pending_orders"]
        if state.just_abandoned_urgent:
            negative["abandoned_urgent_order"] = state.just_abandoned_urgent * self.reward_weights.negative["abandoned_urgent_order"]
        if state.just_congestion_penalty:
            negative["congestion_from_poor_allocation"] = state.just_congestion_penalty * self.reward_weights.negative["congestion_from_poor_allocation"]
        if state.just_overloaded_assignments:
            negative["overloaded_assignment"] = state.just_overloaded_assignments * self.reward_weights.negative["overloaded_assignment"]
        return RewardBreakdown.from_components(positive=positive, negative=negative)
