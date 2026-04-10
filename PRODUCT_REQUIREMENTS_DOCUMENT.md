# Product Requirements Document
## Supply Chain Risk Management (SCRM) Environment

---

## 1. Executive Summary

**Environment Name**: Supply Chain Risk Management (SCRM)

**Domain**: Enterprise Supply Chain Operations

**Problem Statement**: Build an RL environment that simulates enterprise supply chain risk management, where an agent must monitor supply chain events, assess risks, diversify suppliers, negotiate alternatives, and maintain operational continuity during disruptions.

**Value Proposition**: 
- Models a critical but underrepresented domain in RL environments
- Directly applicable to training business agents for enterprise operations
- Tests multi-objective optimization (cost vs. resilience) under uncertainty
- Progressive difficulty from routine monitoring to global crisis management

---

## 2. Environment Overview

### 2.1 What It Simulates

The environment models a mid-sized electronics manufacturer's supply chain operations, with:
- 15+ suppliers across multiple regions (Asia, Europe, North America, LATAM)
- 3 product lines with different margin profiles and lead times
- Realistic risk events: natural disasters, geopolitical issues, vendor quality failures
- Financial considerations: inventory costs, penalty costs, opportunity costs

### 2.2 Core Mechanics

```
┌─────────────────────────────────────────────────────────────────┐
│                        SUPPLY CHAIN ENVIRONMENT                   │
├─────────────────────────────────────────────────────────────────┤
│  State: Supplier Health, Risk Events, Inventory, Cost Baseline   │
│                                                                     │
│  Actions: Monitor, Assess, Mitigate, Diversify, Recover          │
│                                                                     │
│  Reward: Cost savings + Resilience gains - Risk penalties        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Agent Goals by Task

| Task | Objective | Success Metric |
|------|-----------|----------------|
| Easy | Routine monitoring & minor disruptions | Maintain 95%+ on-time delivery |
| Medium | Multi-event crisis management | Minimize cost impact during crisis |
| Hard | Global cascade failure recovery | Full recovery within SLA constraints |

---

## 3. Technical Specifications

### 3.1 Environment Interface

```python
class SupplyChainEnvironment(ObservationSpace):
    def __init__(self, config: Config):
        ...
    
    def reset(self) -> Observation:
        """Reset environment to initial state"""
        ...
    
    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """
        Advance environment by one timestep
        Returns: (observation, reward, done, info)
        """
        ...
    
    def state(self) -> dict:
        """Get current state snapshot"""
        ...
    
    def state_is_terminal(self) -> bool:
        """Check if environment has terminated"""
        ...
```

### 3.2 Typed Models (Pydantic)

```python
class Observation(BaseModel):
    """Supply chain state observation"""
    timestamp: datetime
    inventory_levels: dict[str, float]  # part_number -> quantity
    supplier_status: dict[str, SupplierStatus]  # supplier_id -> status
    risk_events: list[RiskEvent]
    cost_metrics: CostMetrics
    delivery_performance: DeliveryMetrics
    market_conditions: MarketConditions

class Action(BaseModel):
    """Agent action space"""
    action_type: ActionType  # MONITOR, ASSESS_RISK, MITIGATE, DIVERSIFY, RECOVER
    supplier_id: Optional[str]
    part_id: Optional[str]
    action_data: Optional[dict]  # action-specific parameters

class Reward(BaseModel):
    """Reward breakdown"""
    base_reward: float  # core operational reward
    risk_reward: float  # risk reduction signal
    cost_reward: float  # cost optimization signal
    resilience_reward: float  # resilience improvement signal
    penalty: float  # SLA breaches, etc.
    total: float  # final reward
```

### 3.3 Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `MONITOR` | Scan for new risks/events | `{}` |
| `ASSESS_RISK` | Evaluate risk severity for supplier/part | `{supplier_id, risk_type}` |
| `MITIGATE` | Execute mitigation playbook | `{supplier_id, playbook_id}` |
| `DIVERSIFY` | Add new supplier to portfolio | `{supplier_id, target_parts, capacity}` |
| `NEGOTIATE` | Negotiate terms with supplier | `{supplier_id, term_type}` |
| `RECOVER` | Execute business continuity plan | `{event_id, recovery_options}` |
| `UPDATE_SOP` | Update standard operating procedures | `{sop_id, changes}` |
| `FLAG_FOR_EXEC` | Escalate to executive team | `{reason}` |

### 3.4 Reward Function Design

```
Total Reward = base_reward + risk_reward + cost_reward + resilience_reward + penalty

Where:
├─ base_reward: +0.05 per period if on-time delivery > 90%
├─ risk_reward: +0.10 per successful risk mitigation
├─ cost_reward: +0.05 * (cost_reduction / baseline_cost)
├─ resilience_reward: +0.08 per new supplier added
├─ penalty: -0.20 per SLA breach, -0.10 per delay event
└─ timeout: -0.50 if 50 steps without meaningful action
```

**Reward Range**: -1.0 to +1.0

**Sparse Penalty Events**:
- Customer complaint: -0.25
- Quality failure: -0.30
- Critical shortage: -0.40

### 3.5 Task System

#### Task 1: Routine Operations (Easy)
- **Duration**: 20 steps
- **Goal**: Handle routine supplier performance issues
- **Success Criteria**: Maintain >90% on-time delivery
- **Expected Score**: 0.7-0.85

#### Task 2: Multi-Region Crisis (Medium)
- **Duration**: 35 steps
- **Goal**: Manage simultaneous disruptions in 2+ regions
- **Success Criteria**: Minimize total cost impact
- **Expected Score**: 0.5-0.7

#### Task 3: Global Cascade Failure (Hard)
- **Duration**: 50 steps
- **Goal**: Recover from world-class event (pandemic-style)
- **Success Criteria**: Full recovery within SLA constraints
- **Expected Score**: 0.3-0.5

---

## 4. State Management

### 4.1 State Components

```
STATE = {
    "inventory": {
        "part_id": {
            "quantity": float,
            "min_level": float,
            "reorder_point": float
        }
    },
    "suppliers": {
        "supplier_id": {
            "health_score": float,  # 0-100
            "reliability": float,    # 0-1
            "regions": [str],
            "capacity_used": float,
            "active_events": [str]
        }
    },
    "risk_events": [
        {
            "id": str,
            "type": str,
            "severity": float,  # 0-1
            "impact": str,
            "affected_suppliers": [str],
            "status": "active|mitigated|resolved"
        }
    ],
    "performance": {
        "on_time_delivery": float,  # 0-1
        "quality_score": float,     # 0-1
        "cost_variance": float
    },
    "actions_taken": [str],  # Action history for grader
    "steps_elapsed": int
}
```

### 4.2 Episode Boundaries

- **Normal termination**: Episode 50 steps or when terminal event occurs
- **Early termination**: Terminal event (bankruptcy, critical shortage)
- **Hard reset**: Clean state restoration via reset()
- **Soft reset**: Continue from current state

---

## 5. Grader Implementation

### 5.1 Task Graders

```python
class TaskGraders:
    def __init__(self, env: SupplyChainEnvironment):
        self.env = env
        self.task_configs = {
            "easy": self._get_easy_config(),
            "medium": self._get_medium_config(),
            "hard": self._get_hard_config()
        }
    
    def grade_task(self, task_id: str, trajectory: list[tuple[Obs, Action, Reward]]) -> float:
        """
        Grade a task completion
        Returns: score between 0.0 and 1.0
        """
        ...
    
    def _get_easy_config(self):
        return {
            "target_delivery_rate": 0.90,
            "max_allowed_failures": 3,
            "weight_cost": 0.2,
            "weight_resilience": 0.3
        }
```

### 5.2 Grader Metrics

| Metric | Weight | Calculation |
|--------|--------|-------------|
| Delivery Rate | 40% | (on-time shipments / total shipments) |
| Cost Variance | 25% | 1 - (cost_increase / baseline) |
| Risk Mitigation | 20% | Successful mitigations / attempted |
| Supplier Health | 10% | Avg health score improvement |
| SLA Compliance | 5% | SLA breaches count |

### 5.3 Deterministic Scoring

```python
def calculate_grade(metrics: dict, task_config: dict) -> float:
    """Calculate deterministic grade"""
    score = (
        metrics["delivery_rate"] * task_config["weight_delivery"] * 1.0 +
        metrics["cost_variance"] * task_config["weight_cost"] * 0.95 +
        metrics["risk_mitigation"] * task_config["weight_risk"] * 0.85 +
        metrics["supplier_health"] * task_config["weight_health"]
    )
    # Clamp to [0, 1]
    return float(max(0.0, min(1.0, score)))
```

---

## 6. Non-Functional Requirements

### 6.1 Performance

- Episode completion time: < 3 seconds
- State serialization: < 100ms
- Memory footprint: < 100MB

### 6.2 Reliability

- Reset produces identical state for same seed
- No state leaks between episodes
- Graceful degradation on malformed actions

### 6.3 Compatibility

- Python 3.10+
- Pydantic v2
- Standard library only (no external deps beyond HF API)

---

## 7. Acceptance Criteria

| Criterion | Target | Notes |
|-----------|--------|-------|
| Environment validates | 100% pass | `openenv validate` command |
| Docker builds | Clean build | No warnings, all layers |
| HF Space deploys | Working space | REST API responds |
| Baseline scores | Reproducible | Same scores on re-run |
| Reward shaping | Varying signal | Not binary |
| Grader scores | 0.0-1.0 range | All tasks scored |

---

## 8. Deliverables

1. `openenv.yaml` - Environment configuration
2. `inference.py` - Baseline inference script
3. `Dockerfile` - Container build instructions
4. `README.md` - Full documentation
5. Environment source code (models/, tasks/, env/, utils/)

---

## 9. Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Baseline scores | All tasks > 0.3 | o1-mini achieves passing |
| Reward variance | > 0.5 range | Not sparse binary |
| State reset consistency | 100% | Deterministic resets |
| Task difficulty | Clear progression | Easy→Medium→Hard |

---

## 10. Appendix

### 10.1 Risk Event Types

```python
RISK_EVENT_TYPES = {
    "natural_disaster": "Earthquake, flood, typhoon",
    "geopolitical": "Trade war, sanctions, export controls",
    "vendor_failure": "Quality failure, bankruptcy, fraud",
    "infrastructure": "Power outage, port closure, customs delay",
    "pandemic": "Worker shortages, quarantine restrictions",
    "regulatory": "New compliance requirements, audits"
}
```

### 10.2 Mitigation Playbooks

```python
MITIGATION_PLAYBOOKS = {
    "activate_backup_supplier": "Switch to pre-vetted backup",
    "increase_inventory": "Rush order from current suppliers",
    "negotiate_terms": "Renegotiate delivery timelines",
    "diversify_region": "Add supplier from different region",
    "cancel_order": "Cut loss and prevent waste"
}
```

---

*Document Version: 1.0*
*Last Updated: 2026-04-05*
