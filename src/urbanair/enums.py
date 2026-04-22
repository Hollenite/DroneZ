from enum import Enum


class DroneType(str, Enum):
    FAST_LIGHT = "fast_light"
    HEAVY_CARRIER = "heavy_carrier"
    LONG_RANGE_SENSITIVE = "long_range_sensitive"
    RELAY = "relay"


class DroneStatus(str, Enum):
    IDLE = "idle"
    ASSIGNED = "assigned"
    IN_FLIGHT = "in_flight"
    CHARGING = "charging"
    HOLDING = "holding"
    OFFLINE = "offline"


class WeatherSeverity(str, Enum):
    CLEAR = "clear"
    MODERATE_WIND = "moderate_wind"
    HEAVY_RAIN = "heavy_rain"
    STORM = "storm"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OrderPriority(str, Enum):
    NORMAL = "normal"
    URGENT = "urgent"
    MEDICAL = "medical"


class RecipientAvailability(str, Enum):
    AVAILABLE = "available"
    UNCERTAIN = "uncertain"
    UNAVAILABLE = "unavailable"


class DropMode(str, Enum):
    DOORSTEP = "doorstep"
    LOCKER = "locker"
    HANDOFF = "handoff"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    DEMO = "demo"


class ActionType(str, Enum):
    ASSIGN_DELIVERY = "assign_delivery"
    REROUTE = "reroute"
    RETURN_TO_CHARGE = "return_to_charge"
    RESERVE_CHARGER = "reserve_charger"
    DELAY_ORDER = "delay_order"
    PRIORITIZE_ORDER = "prioritize_order"
    SWAP_ASSIGNMENTS = "swap_assignments"
    ATTEMPT_DELIVERY = "attempt_delivery"
    FALLBACK_TO_LOCKER = "fallback_to_locker"
    HOLD_FLEET = "hold_fleet"
    RESUME_OPERATIONS = "resume_operations"
