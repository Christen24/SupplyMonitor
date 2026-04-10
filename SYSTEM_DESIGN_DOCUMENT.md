# System Design Document
## Supply Chain Risk Management (SCRM) Environment

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SCRM Environment Layer                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │    Models    │  │  State       │  │   Tasks       │          │
│  │   (Pydantic) │  │ Manager      │  │  (Graders)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Actions    │  │  Reward      │  │  Event       │          │
│  │   Engine     │  │  Shaper      │  │   Generator  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Environment Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  SupplyChainEnvironment  ──  step() / reset() / state()         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Infrastructure Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ├── Docker Container                                           │
│  ├── Hugging Face Space API                                      │
│  └── OpenAI Inference API                                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 Core Components

#### 2.1.1 `models/support.py` - Typed Models

```python
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from typing import Optional

class ActionType(Enum):
    MONITOR = "monitor"
    ASSESS_RISK = "assess_risk"
    MITIGATE = "mitigate"
    DIVERSIFY = "diversify"
    NEGOTIATE = "negotiate"
    RECOVER = "recover"
    UPDATE_SOP = "update_sop"
    FLAG_FOR_EXEC = "flag_for_exec"

class SupplierStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    INACTIVE = "inactive"

class RiskSeverity(Enum):
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.8
    CRITICAL = 1.0

class Observation(BaseModel):
    """Supply chain state observation"""
    timestamp: datetime = Field(description="Current simulation time")
    inventory_levels: dict[str, float] = Field(description="part_id -> quantity")
    supplier_status: dict[str, tuple[str, float, float]] = Field(
        description="supplier_id -> (status, health_score, reliability)"
    )
    risk_events: list[dict] = Field(description="Active risk events")
    cost_metrics: dict[str, float] = Field(description="Cost tracking")
    delivery_performance: dict[str, float] = Field(description="OTD, quality, etc.")
    market_conditions: dict[str, float] = Field(description="Market signals")

    class Config:
        json_schema_extra = {
            "description": "Supply chain state with inventory, supplier status, "
                          "active risks, and performance metrics",
            "type": "observation"
        }

class Action(BaseModel):
    """Agent action space"""
    action_type: ActionType = Field(description="Type of action to take")
    supplier_id: Optional[str] = Field(description="Target supplier")
    part_id: Optional[str] = Field(description="Affected part number")
    action_data: Optional[dict] = Field(description="Action-specific parameters")

    class Config:
        json_schema_extra = {
            "description": "Actions include monitoring, risk assessment, "
                          "mitigation, diversification, and recovery steps",
            "type": "action"
        }

class Reward(BaseModel):
    """Reward breakdown"""
    base_reward: float = Field(description="Core operational reward")
    risk_reward: float = Field(description="Risk reduction signal")
    cost_reward: float = Field(description="Cost optimization signal")
    resilience_reward: float = Field(description="Resilience improvement signal")
    penalty: float = Field(description="SLA breaches, penalties")
    total: float = Field(description="Final reward, clamped to [-1, 1]")

    class Config:
        json_schema_extra = {
            "description": "Reward includes partial progress signals: "
                          "+0.05 for stable operations, +0.10 for mitigations, "
                          "-0.20 per SLA breach"
        }
```

#### 2.1.2 `env/state_manager.py` - State Management

```python
from typing import Any
from models.support import Observation

class StateManager:
    """Manages supply chain state across episodes"""
    
    def __init__(self, config: dict):
        self.inventory = {}  # part_id -> {quantity, min_level, reorder_point}
        self.suppliers = {}  # supplier_id -> {health, reliability, regions, ...}
        self.risk_events = []  # list of active risk events
        self.actions_taken = []  # Action history for graders
        self.steps_elapsed = 0
        self.config = config
    
    def reset(self) -> Observation:
        """Reset to clean initial state"""
        self.inventory = {}
        self.suppliers = {}
        self.risk_events = []
        self.actions_taken = []
        self.steps_elapsed = 0
        # Initialize with seeded random state for reproducibility
        self._initialize_initial_state(seed=self.config.get("seed"))
        return self._get_observation()
    
    def step(self, action: Action) -> tuple[dict, bool]:
        """Process action, update state, return (is_terminal, next_state)"""
        action_type = action.action_type
        self.actions_taken.append({
            "action_type": action_type.value,
            "supplier_id": action.supplier_id,
            "part_id": action.part_id,
            "timestamp": self.steps_elapsed
        })
        
        # Update state based on action
        if action_type == ActionType.MONITOR:
            self._on_monitor(action)
        elif action_type == ActionType.ASSESS_RISK:
            self._on_assess_risk(action)
        elif action_type == ActionType.MITIGATE:
            self._on_mitigate(action)
        # ... other actions
        
        self.steps_elapsed += 1
        return (self.steps_elapsed >= self.config.get("max_steps", 50), 
                self._get_observation())
    
    def _initialize_initial_state(self, seed: int):
        """Initialize suppliers, inventory with seeded randomness"""
        rng = random.Random(seed)
        # Create initial supplier network
        regions = ["asia", "europe", "north_america", "latam"]
        for i, region in enumerate(regions):
            num_suppliers = rng.randint(3, 6)
            for j in range(num_suppliers):
                supplier_id = f"sup_{i}_{j}"
                self.suppliers[supplier_id] = {
                    "health": rng.uniform(80, 100),
                    "reliability": rng.uniform(0.85, 1.0),
                    "regions": [region],
                    "capacity_used": rng.uniform(0.3, 0.7),
                    "active_events": []
                }
    
    def _get_observation(self) -> Observation:
        """Convert internal state to Observation model"""
        return Observation(
            timestamp=self.steps_elapsed * self.config.get("step_interval", 1),
            inventory_levels={k: v["quantity"] for k, v in self.inventory.items()},
            supplier_status={
                k: (v["status"], v["health"], v["reliability"])
                for k, v in self.suppliers.items()
            },
            risk_events=self.risk_events,
            cost_metrics={
                "total_cost": self._calculate_total_cost(),
                "cost_variance": self._calculate_cost_variance()
            },
            delivery_performance={
                "on_time_delivery": self._calculate_otd(),
                "quality_score": self._calculate_quality(),
            },
            market_conditions=self._calculate_market_conditions()
        )
```

#### 2.1.3 `tasks/scenario_generator.py` - Scenario Tasks

```python
from typing import Literal

class ScenarioGenerator:
    """Generates task scenarios with graders"""
    
    def __init__(self, env: Any):
        self.env = env
    
    def create_easy_task(self) -> dict:
        """
        Task 1: Routine Operations (Easy)
        Duration: 20 steps
        Goal: Handle routine supplier issues
        """
        return {
            "id": "easy",
            "name": "Routine Operations",
            "difficulty": "easy",
            "duration_steps": 20,
            "objective": "Maintain supplier operations during minor disruptions",
            "success_criteria": {
                "min_delivery_rate": 0.90,
                "max_failures": 3,
                "penalty_thresholds": []
            },
            "initial_state": {
                "risk_level": "low",
                "events_count": 0
            }
        }
    
    def create_medium_task(self) -> dict:
        """
        Task 2: Multi-Region Crisis (Medium)
        Duration: 35 steps
        Goal: Manage simultaneous disruptions in 2+ regions
        """
        return {
            "id": "medium",
            "name": "Multi-Region Crisis",
            "difficulty": "medium",
            "duration_steps": 35,
            "objective": "Handle concurrent supplier disruptions across regions",
            "success_criteria": {
                "min_delivery_rate": 0.80,
                "max_cost_impact": 0.15,
                "required_actions": ["diversify", "mitigate", "recover"]
            },
            "initial_state": {
                "risk_level": "medium",
                "events_count": 2,
                "affected_regions": ["asia", "europe"]
            }
        }
    
    def create_hard_task(self) -> dict:
        """
        Task 3: Global Cascade Failure (Hard)
        Duration: 50 steps
        Goal: Recover from world-class event
        """
        return {
            "id": "hard",
            "name": "Global Cascade Failure",
            "difficulty": "hard",
            "duration_steps": 50,
            "objective": "Manage global supply chain crisis with cascading impacts",
            "success_criteria": {
                "min_delivery_rate": 0.70,
                "max_cost_impact": 0.25,
                "recovery_required": True,
                "escalations_required": 2
            },
            "initial_state": {
                "risk_level": "high",
                "events_count": 4,
                "affected_regions": ["asia", "europe", "north_america", "latam"],
                "cascading": True
            }
        }
```

---

## 3. Data Flow Diagrams

### 3.1 Episode Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     EPISODE LIFECYCLE                        │
└─────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │   reset()   │  ← New episode with clean state
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │   step()    │  ← Agent takes action
    │ (action)    │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  state()    │  ← Observe current state
    │   state()   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  get_reward()│ ← Compute reward from delta
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ is_terminal?│  ← Check termination conditions
    └──────┬──────┘
           │
      ┌────┴─────┐
      │          │
    YES         NO
      │          │
      ▼          ▼
   ┌────┐     ┌────────┐
   │END │     │ CONTINUE│
   └────┘     └────┬───┘
                   │
                   ▼
              ┌────────┐
              │ reward │
              └────────┘
```

### 3.2 Reward Computation Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    REWARD COMPUTATION                         │
└─────────────────────────────────────────────────────────────┘

   State(t-1) ─────┐
                   │
                   ▼
         ┌──────────────────┐
         │  Action Taken    │
         │  + State Change  │
         └──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  Delta Detection │
         │  - Inventory     │
         │  - Supplier      │
         │  - Performance   │
         └──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  Reward          │
         │  Components:     │
         │  - base_reward:  │
         │    stable ops    │
         │  - risk_reward:  │
         │    mitigation    │
         │  - cost_reward:  │
         │    variance      │
         │  - resilience:   │
         │    diversity     │
         │  - penalty:      │
         │    failures      │
         └──────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │  Sum & Clamp     │
         │  to [-1, 1]      │
         └──────────────────┘
```

---

## 4. Key Algorithms

### 4.1 Reward Shaping Algorithm

```python
def compute_reward(state_prev: dict, action: Action, state_curr: dict) -> float:
    """
    Compute reward with partial progress signals
    
    Returns: reward in range [-1.0, 1.0]
    """
    reward = Reward(
        base_reward=self._compute_base_reward(state_prev, state_curr),
        risk_reward=self._compute_risk_reward(action, state_curr),
        cost_reward=self._compute_cost_reward(state_curr),
        resilience_reward=self._compute_resilience_reward(state_curr),
        penalty=self._compute_penalty(state_curr)
    )
    return reward.total
    
    # Component implementations:
    # _compute_base_reward: +0.05 if OTD > 90%, else 0
    # _compute_risk_reward: +0.10 per successful mitigation
    # _compute_cost_reward: +0.05 * (cost_reduction / baseline)
    # _compute_resilience: +0.08 per new supplier added
    # _compute_penalty: -0.20 per SLA breach, -0.10 per delay
```

### 4.2 Risk Event Propagation

```python
def propagate_risk(event: RiskEvent, suppliers: dict) -> None:
    """
    Propagate risk event to affected suppliers
    
    Cascade effects:
    - Supplier health decreases by severity
    - Reliability may drop
    - May trigger chain reactions
    """
    severity = event.severity
    
    for supplier_id in event.affected_suppliers:
        if supplier_id in suppliers:
            suppliers[supplier_id]["health"] -= severity * 0.3
            suppliers[supplier_id]["reliability"] -= severity * 0.05
            
            # Chain reaction: if health < threshold, supplier degrades
            if suppliers[supplier_id]["health"] < 50:
                # Trigger quality degradation
                suppliers[supplier_id]["quality_degraded"] = True
```

### 4.3 Task Grading Algorithm

```python
def grade_task(trajectory: list[tuple[Obs, Action, Reward]], task_config: dict) -> float:
    """
    Compute deterministic task grade
    
    Returns: score in [0.0, 1.0]
    """
    metrics = {
        "delivery_rate": self._compute_delivery_rate(trajectory),
        "cost_variance": self._compute_cost_variance(trajectory),
        "risk_mitigation": self._compute_risk_mitigation(trajectory),
        "supplier_health": self._compute_health_improvement(trajectory),
        "sla_compliance": 1.0 - self._count_sla_breaches(trajectory) * 0.1
    }
    
    # Weighted sum
    grade = (
        metrics["delivery_rate"] * task_config["weight_delivery"] * 1.0 +
        metrics["cost_variance"] * task_config["weight_cost"] * 0.95 +
        metrics["risk_mitigation"] * task_config["weight_risk"] * 0.85 +
        metrics["supplier_health"] * task_config["weight_health"]
    )
    
    # Clamp and return
    return float(max(0.0, min(1.0, grade)))
```

---

## 5. File Structure

```
/
├── openenv.yaml                    # Environment config
├── inference.py                    # Baseline inference script
├── Dockerfile                      # Container build
├── README.md                       # Documentation
├── PRODUCT_REQUIREMENTS_DOCUMENT.md
├── SYSTEM_DESIGN_DOCUMENT.md
├── models/
│   ├── __init__.py
│   └── support.py                  # Pydantic models
├── env/
│   ├── __init__.py
│   └── environment.py              # Main Environment class
├── tasks/
│   ├── __init__.py
│   └── scenarios.py                # Scenario generator
├── utils/
│   ├── __init__.py
│   ├── helpers.py                  # Utility functions
│   └── risk_engine.py              # Risk propagation logic
└── configs/
    └── tasks.yaml                   # Task configurations
```

---

## 6. API Contract

### 6.1 Environment REST API

```python
# Expected HTTP endpoints when deployed to HF Space

# Reset environment
POST /reset
Response: {
    "observation": {...},
    "timestamp": "iso_timestamp"
}

# Take step
POST /step
Request: {
    "action": {...}  # Action JSON
}
Response: {
    "observation": {...},
    "reward": float,
    "done": bool,
    "info": {...}
}

# Get state
GET /state
Response: {
    "state": {...}
}
```

### 6.2 Action Schema

```json
{
    "action_type": "monitor",
    "supplier_id": null,
    "part_id": null,
    "action_data": {}
}
```

### 6.3 Observation Schema

```json
{
    "timestamp": "2026-04-05T12:00:00Z",
    "inventory_levels": {
        "part_001": 1500,
        "part_002": 800
    },
    "supplier_status": {
        "sup_asia_1": ["healthy", 85.0, 0.92]
    },
    "risk_events": [],
    "cost_metrics": {
        "total_cost": 125000.0,
        "cost_variance": 0.02
    },
    "delivery_performance": {
        "on_time_delivery": 0.95,
        "quality_score": 0.98
    },
    "market_conditions": {
        "demand_volatility": 0.15,
        "price_index": 1.02
    }
}
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```python
# models/support.py
def test_observation_creation():
    obs = Observation(
        timestamp=datetime.now(),
        inventory_levels={"part_001": 1000}
    )
    assert obs.inventory_levels == {"part_001": 1000}

# env/environment.py
def test_reset_clean_state():
    env = SupplyChainEnvironment(config)
    obs1 = env.reset()
    obs2 = env.reset()
    assert obs1.model_dump() == obs2.model_dump()  # Identical resets

# env/environment.py
def test_step_returns_tuple():
    env = SupplyChainEnvironment(config)
    env.reset()
    action = Action(action_type=ActionType.MONITOR)
    result = env.step(action)
    assert len(result) == 4
    assert isinstance(result[0], Observation)
    assert isinstance(result[1], float)
    assert isinstance(result[2], bool)
```

### 7.2 Integration Tests

```python
# Test full episode
def test_episode_completion():
    env = SupplyChainEnvironment(config)
    obs = env.reset()
    
    # Run until done
    total_reward = 0
    for _ in range(50):
        action = Action(action_type=ActionType.MONITOR)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            break
    
    assert 0 < total_reward < 1.0  # Reward within bounds
```

---

## 8. Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Episode duration | < 3s | Wall-clock time |
| Reset latency | < 50ms | From call to state |
| Step latency | < 100ms | From action to response |
| Memory usage | < 100MB | RSS size |
| State serialization | < 100ms | dump() time |

---

## 9. Security & Reliability

### 9.1 Input Validation

```python
# All actions validated through Pydantic models
# Prevents malformed actions from causing errors
action = Action.model_validate(action_dict)
```

### 9.2 State Isolation

```python
# Each episode gets fresh state
def reset(self):
    # All state dicts replaced with new ones
    self.inventory = {}
    self.suppliers = {}
    self.risk_events = []
```

### 9.3 Error Handling

```python
def step(self, action: Action):
    try:
        # Process action
        ...
        return obs, reward, done, info
    except Exception as e:
        # Graceful degradation
        return self._handle_error(e)
```

---

*Document Version: 1.0*
*Last Updated: 2026-04-05*
