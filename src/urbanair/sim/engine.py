from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..enums import ActionType, DroneStatus
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
        )
        events.extend(delivery_events)
        state.resolved_order_ids.update(delivered_ids)
        reward_inputs["deliveries_completed"] += float(len(delivered_ids))
        reward_inputs["failed_attempts"] += float(len(recovery_actions))
        reward_inputs["urgent_successes"] += float(
            sum(1 for order in state.orders if order.order_id in delivered_ids and order.priority.value in {"urgent", "medical"})
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
        reward_inputs["critical_battery"] += float(sum(1 for drone in state.fleet if drone.battery <= 12))
        breakdown = self._build_reward_breakdown(reward_inputs)
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
                order.assigned_drone_id = drone.drone_id
                order.status = "assigned"
                events.extend(assign_order(drone, order.order_id, order.zone_id, state.sectors))

        elif action_type == ActionType.RETURN_TO_CHARGE:
            drone = next((item for item in state.fleet if item.drone_id == action["drone_id"]), None)
            station = next((item for item in state.charging_stations if item.station_id == action["station_id"]), None)
            if drone and station:
                events.extend(send_to_charge(drone, station))

        elif action_type == ActionType.PRIORITIZE_ORDER:
            order = next((item for item in state.orders if item.order_id == action["order_id"]), None)
            if order:
                order.deadline = max(1, order.deadline - 1)
                events.append(f"{order.order_id} was prioritized.")

        return events

    def _resolve_charging_pressure(self, state: SimulatorState) -> None:
        for station in state.charging_stations:
            if station.queue_size > 0 and station.occupied_slots < station.capacity:
                station.queue_size -= 1
                station.occupied_slots += 1

    def _is_done(self, state: SimulatorState) -> bool:
        unresolved_orders = [order for order in state.orders if order.order_id not in state.resolved_order_ids and order.status != "canceled"]
        no_viable_drones = not any(drone.status != DroneStatus.OFFLINE and drone.battery > 10 for drone in state.fleet)
        return not unresolved_orders or state.tick >= state.task_config.horizon or no_viable_drones

    def _build_reward_breakdown(self, reward_inputs: dict[str, float]) -> RewardBreakdown:
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
        return RewardBreakdown.from_components(positive=positive, negative=negative)
