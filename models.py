"""Pydantic v2 typed models for the SCRM OpenEnv environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field
from openenv.core.env_server import Action, Observation, State


# ── Enums ──────────────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """Available agent actions."""
    MONITOR = "monitor"
    ASSESS_RISK = "assess_risk"
    MITIGATE = "mitigate"
    DIVERSIFY = "diversify"
    NEGOTIATE = "negotiate"
    RECOVER = "recover"
    UPDATE_SOP = "update_sop"
    FLAG_FOR_EXEC = "flag_for_exec"


class SupplierStatus(str, Enum):
    """Supplier health status categories."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    INACTIVE = "inactive"


class RiskSeverity(float, Enum):
    """Risk event severity levels with numeric values."""
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0


# ── Risk Event Types ───────────────────────────────────────────────────────────

RISK_EVENT_TYPES: dict[str, str] = {
    "natural_disaster": "Earthquake, flood, typhoon",
    "geopolitical": "Trade war, sanctions, export controls",
    "vendor_failure": "Quality failure, bankruptcy, fraud",
    "infrastructure": "Power outage, port closure, customs delay",
    "pandemic": "Worker shortages, quarantine restrictions",
    "regulatory": "New compliance requirements, audits",
}

MITIGATION_PLAYBOOKS: dict[str, str] = {
    "activate_backup_supplier": "Switch to pre-vetted backup",
    "increase_inventory": "Rush order from current suppliers",
    "negotiate_terms": "Renegotiate delivery timelines",
    "diversify_region": "Add supplier from different region",
    "cancel_order": "Cut loss and prevent waste",
}


# ── Core Models ────────────────────────────────────────────────────────────────

class SCRMState(State):
    """Internal state tracking."""
    # episode_id and step_count are inherited from State
    target_inventory: dict[str, float] = Field(default_factory=dict)
    active_suppliers: list[str] = Field(default_factory=dict)
    global_performance: dict[str, float] = Field(default_factory=dict)


class RewardBreakdown(BaseModel):
    """Reward breakdown."""
    base_reward: float
    risk_reward: float
    cost_reward: float
    resilience_reward: float
    penalty: float
    total: float

    @classmethod
    def compute(
        cls,
        base_reward: float,
        risk_reward: float,
        cost_reward: float,
        resilience_reward: float,
        penalty: float,
    ) -> "RewardBreakdown":
        raw = base_reward + risk_reward + cost_reward + resilience_reward + penalty
        total = max(-1.0, min(1.0, raw))
        return cls(
            base_reward=round(base_reward, 4),
            risk_reward=round(risk_reward, 4),
            cost_reward=round(cost_reward, 4),
            resilience_reward=round(resilience_reward, 4),
            penalty=round(penalty, 4),
            total=round(total, 4),
        )


class SCRMObservation(Observation):
    """Supply chain state observation returned by reset() and step()."""
    # done and reward are inherited from Observation
    timestamp: int = Field(description="Current simulation step number")
    inventory_levels: dict[str, float] = Field(description="part_id → current quantity on hand")
    supplier_status: dict[str, dict[str, Any]] = Field(description="supplier_id → {status, health_score, reliability, region, capacity_used}")
    risk_events: list[dict[str, Any]] = Field(description="List of active risk event dicts")
    cost_metrics: dict[str, float] = Field(description="total_cost, cost_variance, baseline_cost")
    delivery_performance: dict[str, float] = Field(description="on_time_delivery (0-1), quality_score (0-1)")
    market_conditions: dict[str, float] = Field(description="demand_volatility, price_index")
    reward_breakdown: Optional[RewardBreakdown] = Field(default=None, description="Detailed reward breakdown for the step")

    model_config = {
        "json_schema_extra": {
            "description": "Supply chain state with inventory, supplier status, active risks, and performance metrics",
            "type": "observation",
        }
    }


class SCRMAction(Action):
    """Agent action submitted to step()."""
    action_type: ActionType = Field(description="Type of action to take")
    supplier_id: Optional[str] = Field(default=None, description="Target supplier ID (if applicable)")
    part_id: Optional[str] = Field(default=None, description="Affected part number (if applicable)")
    action_data: Optional[dict[str, Any]] = Field(default=None, description="Action-specific parameters")

    model_config = {
        "json_schema_extra": {
            "description": "Actions include monitoring, risk assessment, mitigation, diversification, and recovery steps",
            "type": "action",
        }
    }
