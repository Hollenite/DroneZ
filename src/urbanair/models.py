from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    ActionType,
    Difficulty,
    DroneStatus,
    DroneType,
    DropMode,
    OrderPriority,
    RecipientAvailability,
    RiskLevel,
    WeatherSeverity,
)


class DroneState(BaseModel):
    drone_id: str
    drone_type: DroneType
    status: DroneStatus = DroneStatus.IDLE
    battery: int = Field(ge=0, le=100)
    payload_capacity: int = Field(ge=0)
    current_zone: str
    assigned_order_id: str | None = None
    eta: int | None = Field(default=None, ge=0)
    health_risk: RiskLevel = RiskLevel.LOW
    communication_strength: str = "strong"
    maintenance_health: str = "good"
    reserved_station_id: str | None = None
    hold_reason: str | None = None
    delivered_order_count: int = Field(default=0, ge=0)
    failed_order_count: int = Field(default=0, ge=0)
    total_flight_ticks: int = Field(default=0, ge=0)
    home_zone: str = "hub"
    target_zone: str | None = None
    flight_path: list[str] = Field(default_factory=list)


class OrderState(BaseModel):
    order_id: str
    priority: OrderPriority = OrderPriority.NORMAL
    deadline: int = Field(ge=0)
    drop_mode: DropMode = DropMode.DOORSTEP
    recipient_availability: RecipientAvailability = RecipientAvailability.AVAILABLE
    retry_count: int = Field(default=0, ge=0)
    late_penalty: str = "medium"
    assigned_drone_id: str | None = None
    zone_id: str
    status: str = "queued"
    package_weight: int = Field(default=1, ge=1)
    fallback_options: list[str] = Field(default_factory=lambda: ["locker", "retry", "human_escalation"])


class SectorState(BaseModel):
    zone_id: str
    weather: WeatherSeverity = WeatherSeverity.CLEAR
    congestion_score: float = Field(default=0.0, ge=0.0, le=1.0)
    is_no_fly: bool = False
    likely_failure: bool = False
    operations_paused: bool = False


class ChargingStationState(BaseModel):
    station_id: str
    zone_id: str
    capacity: int = Field(ge=1)
    occupied_slots: int = Field(default=0, ge=0)
    queue_size: int = Field(default=0, ge=0)
    reserved_drone_ids: list[str] = Field(default_factory=list)


class PolicyNotice(BaseModel):
    notice_id: str
    zone_id: str | None = None
    message: str
    severity: RiskLevel = RiskLevel.MEDIUM


class EmergencyEvent(BaseModel):
    event_id: str
    zone_id: str
    summary: str
    severity: RiskLevel = RiskLevel.HIGH
    active: bool = True


class RewardBreakdown(BaseModel):
    positive: dict[str, float] = Field(default_factory=dict)
    negative: dict[str, float] = Field(default_factory=dict)
    total: float = 0.0

    @classmethod
    def from_components(
        cls,
        positive: dict[str, float] | None = None,
        negative: dict[str, float] | None = None,
    ) -> "RewardBreakdown":
        positive = positive or {}
        negative = negative or {}
        total = sum(positive.values()) + sum(negative.values())
        return cls(positive=positive, negative=negative, total=total)


class FleetAction(BaseModel):
    action_type: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)
    rationale: str | None = None


class EnvironmentObservation(BaseModel):
    time_step: int = Field(ge=0)
    horizon: int = Field(ge=1)
    task_id: str
    fleet: list[DroneState] = Field(default_factory=list)
    orders: list[OrderState] = Field(default_factory=list)
    sectors: list[SectorState] = Field(default_factory=list)
    charging_stations: list[ChargingStationState] = Field(default_factory=list)
    policy_notices: list[PolicyNotice] = Field(default_factory=list)
    emergency_events: list[EmergencyEvent] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    reward_summary: RewardBreakdown = Field(default_factory=RewardBreakdown)


class StepResult(BaseModel):
    observation: EnvironmentObservation
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


class EpisodeSummary(BaseModel):
    task_id: str
    policy_id: str | None = None
    seed: int | None = None
    total_reward: float
    steps_completed: int = Field(ge=0)
    completed_deliveries: int = Field(default=0, ge=0)
    failed_deliveries: int = Field(default=0, ge=0)
    urgent_successes: int = Field(default=0, ge=0)
    invalid_action_count: int = Field(default=0, ge=0)
    deadline_miss_count: int = Field(default=0, ge=0)
    critical_battery_events: int = Field(default=0, ge=0)
    action_counts: dict[str, int] = Field(default_factory=dict)
    triggered_scripted_events: list[str] = Field(default_factory=list)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)


class DynamicEventsConfig(BaseModel):
    weather: bool = True
    no_fly: bool = True
    emergency: bool = False
    charging_congestion: bool = True


class ScriptedEvent(BaseModel):
    tick: int = Field(ge=0)
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    task_id: str
    difficulty: Difficulty
    description: str
    seed: int
    horizon: int = Field(ge=1)
    fleet_counts: dict[DroneType, int]
    initial_orders: int = Field(ge=1)
    urgent_orders: int = Field(ge=0)
    dynamic_events: DynamicEventsConfig = Field(default_factory=DynamicEventsConfig)
    failed_drop_probability: float = Field(ge=0.0, le=1.0)
    hidden_factor_intensity: float = Field(ge=0.0, le=1.0)
    deterministic_demo: bool = False
    scripted_events: list[ScriptedEvent] = Field(default_factory=list)


class FleetProfile(BaseModel):
    drone_type: DroneType
    payload_capacity: int = Field(ge=0)
    battery_capacity: int = Field(ge=1)
    cruise_speed: int = Field(ge=1)
    weather_tolerance: str
    can_deliver: bool
    communication_role: str


class FleetProfilesConfig(BaseModel):
    profiles: dict[DroneType, FleetProfile]


class RewardWeightsConfig(BaseModel):
    positive: dict[str, float]
    negative: dict[str, float]
