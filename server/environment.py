"""Supply Chain Risk Management - OpenEnv core compliance format.



Follows the OpenEnv spec:

    reset()   SCRMObservation

    step()    SCRMObservation

    state()   SCRMState

"""



from __future__ import annotations



import copy

import random

import uuid

from typing import Any



from openenv.core.env_server import Environment



from models import (

    SCRMAction,

    ActionType,

    SCRMObservation,

    RewardBreakdown,

    SCRMState,

)

from tasks.scenarios import ScenarioGenerator

from utils.helpers import (

    clamp,

    clamp_open_unit_interval,

    generate_inventory,

    generate_supplier_network,
    supplier_status_label,
    safe_score,
)

from utils.risk_engine import RiskEngine





class SupplyChainEnvironment(Environment):

    """OpenEnv-compliant supply chain risk management environment."""

    

    SUPPORTS_CONCURRENT_SESSIONS = True



    def __init__(self):

        self._task_id = "medium"  # Default task

        self._seed = 42

        self._max_steps = 35



        # Internal simulation variables

        self._suppliers: dict[str, dict[str, Any]] = {}

        self._inventory: dict[str, dict[str, float]] = {}

        self._risk_engine: RiskEngine | None = None

        self._done: bool = False



        # Performance trackers

        self._on_time_delivery: float = 0.95

        self._quality_score: float = 0.98

        self._baseline_cost: float = 0.0

        self._total_cost: float = 0.0

        self._cost_variance: float = 0.0

        self._sla_breaches: int = 0

        self._mitigations_successful: int = 0

        self._suppliers_added: int = 0

        self._idle_steps: int = 0

        

        self._state = SCRMState(step_count=0)



    #  OpenEnv API 



    def reset(self, seed=None, episode_id=None, **kwargs) -> SCRMObservation:

        """Reset environment to a deterministic initial state."""

        self._seed = seed if seed is not None else 42

        # Use config if passed, else fallback to hardcoded

        task_config = kwargs.get("task_config", None)

        if task_config:

            self._task_id = task_config.get("id", "medium")

            self._max_steps = task_config.get("max_steps", 35)



        rng = random.Random(self._seed)



        self._suppliers = generate_supplier_network(rng)

        self._inventory = generate_inventory(rng)

        self._risk_engine = RiskEngine(self._task_id, self._seed)

        self._done = False



        # Performance

        self._on_time_delivery = 0.95

        self._quality_score = 0.98

        self._baseline_cost = self._calc_total_cost()

        self._total_cost = self._baseline_cost

        self._cost_variance = 0.0

        self._sla_breaches = 0

        self._mitigations_successful = 0

        self._suppliers_added = 0

        self._idle_steps = 0

        

        self._state = SCRMState(

            episode_id=episode_id or str(uuid.uuid4()),

            step_count=0,

            active_suppliers=list(self._suppliers.keys()),

            global_performance={

                "otd": self._on_time_delivery,

                "quality": self._quality_score

            }

        )



        return self._build_observation(reward=None, reward_breakdown=None, done=False)



    def step(self, action: SCRMAction, timeout_s=None, **kwargs) -> SCRMObservation:

        if self._done:
            return self._build_observation(reward=safe_score(0.5), reward_breakdown=None, done=True)



        prev_otd = self._on_time_delivery

        prev_cost = self._total_cost

        prev_health_avg = self._avg_supplier_health()



        info: dict[str, Any] = {"sla_breach": False, "action_valid": True}

        try:

            info = self._dispatch_action(action, info)

        except Exception as exc:

            info["action_valid"] = False

            info["error"] = str(exc)



        self._state.step_count += 1

        assert self._risk_engine is not None



        new_events = self._risk_engine.generate_events(self._state.step_count, self._suppliers)

        RiskEngine.propagate_risks(self._risk_engine.active_events, self._suppliers)

        self._risk_engine.cascade_check(self._suppliers)

        self._risk_engine.tick_natural_recovery(self._suppliers)



        for sup in self._suppliers.values():

            sup["status"] = supplier_status_label(sup["health"])



        self._update_performance()



        self._total_cost = self._calc_total_cost()

        if self._baseline_cost > 0:

            self._cost_variance = abs(self._total_cost - self._baseline_cost) / self._baseline_cost



        if self._on_time_delivery < 0.85:

            self._sla_breaches += 1

            info["sla_breach"] = True



        reward_obj = self._compute_reward(prev_otd, prev_cost, prev_health_avg, action, info)

        reward_total = reward_obj.total



        done = self._state.step_count >= self._max_steps

        all_below = all(v["quantity"] < v["min_level"] for v in self._inventory.values())

        if all_below:

            done = True

            reward_total = max(-1.0, reward_total - 0.40)



        self._done = done

        self._state.active_suppliers = list(self._suppliers.keys())

        self._state.global_performance = {"otd": self._on_time_delivery, "quality": self._quality_score}



        # Normalize reward from [-1, 1] to (0, 1) exclusive for hackathon validator
        normalized_reward = (reward_total + 1.0) / 2.0  # maps [-1,1] -> [0,1]
        safe_reward = safe_score(normalized_reward)

        return self._build_observation(

            reward=safe_reward, 

            reward_breakdown=reward_obj, 

            done=done

        )



    @property

    def state(self) -> SCRMState:

        return self._state



    #  Action dispatch 

    def _dispatch_action(self, action: SCRMAction, info: dict) -> dict:

        handlers = {

            ActionType.MONITOR: self._handle_monitor,

            ActionType.ASSESS_RISK: self._handle_assess_risk,

            ActionType.MITIGATE: self._handle_mitigate,

            ActionType.DIVERSIFY: self._handle_diversify,

            ActionType.NEGOTIATE: self._handle_negotiate,

            ActionType.RECOVER: self._handle_recover,

            ActionType.UPDATE_SOP: self._handle_update_sop,

            ActionType.FLAG_FOR_EXEC: self._handle_flag_for_exec,

        }

        handler = handlers.get(action.action_type, self._handle_monitor)

        return handler(action, info)



    def _handle_monitor(self, action: SCRMAction, info: dict) -> dict:

        info["action_result"] = "monitoring_complete"

        self._idle_steps += 1

        return info



    def _handle_assess_risk(self, action: SCRMAction, info: dict) -> dict:

        sid = action.supplier_id

        if sid and sid in self._suppliers:

            info["action_result"] = "assessment_complete"

        self._idle_steps = 0

        return info



    def _handle_mitigate(self, action: SCRMAction, info: dict) -> dict:

        if not self._risk_engine: return info

        event_id = (action.action_data or {}).get("event_id")

        if not event_id and self._risk_engine.active_events:

            event_id = self._risk_engine.active_events[0]["id"]



        if event_id and self._risk_engine.mitigate_event(event_id):

            self._mitigations_successful += 1

            info["action_result"] = "mitigation_successful"

            for evt in self._risk_engine.active_events + self._risk_engine.resolved_events:

                if evt["id"] == event_id:

                    for sid in evt.get("affected_suppliers", []):

                        if sid in self._suppliers:

                            self._suppliers[sid]["health"] = min(100, self._suppliers[sid]["health"] + 4)

        self._idle_steps = 0

        return info



    def _handle_diversify(self, action: SCRMAction, info: dict) -> dict:

        rng = random.Random(self._seed + self._state.step_count)

        region = (action.action_data or {}).get("region", rng.choice(["asia", "europe", "north_america", "latam"]))

        new_id = f"sup_{region}_{len(self._suppliers)}"

        self._suppliers[new_id] = {

            "health": round(rng.uniform(75, 95), 2),

            "reliability": round(rng.uniform(0.85, 0.95), 4),

            "region": region,

            "capacity_used": 0.1,

            "active_events": [],

            "status": "healthy",

            "quality_degraded": False,

        }

        self._suppliers_added += 1

        info["action_result"] = "supplier_added"

        self._idle_steps = 0

        return info



    def _handle_negotiate(self, action: SCRMAction, info: dict) -> dict:

        sid = action.supplier_id

        if sid and sid in self._suppliers:

            self._cost_variance = max(0, self._cost_variance - 0.02)

            self._suppliers[sid]["reliability"] = min(1.0, self._suppliers[sid]["reliability"] + 0.03)

            info["action_result"] = "negotiation_successful"

        self._idle_steps = 0

        return info



    def _handle_recover(self, action: SCRMAction, info: dict) -> dict:

        if not self._risk_engine: return info

        event_id = (action.action_data or {}).get("event_id")

        if not event_id and self._risk_engine.active_events:

            event_id = self._risk_engine.active_events[0]["id"]

        

        if event_id and self._risk_engine.resolve_event(event_id):

            info["action_result"] = "recovery_successful"

            self._mitigations_successful += 1

            self._on_time_delivery = min(1.0, self._on_time_delivery + 0.03)

        self._idle_steps = 0

        return info



    def _handle_update_sop(self, action: SCRMAction, info: dict) -> dict:

        self._quality_score = min(1.0, self._quality_score + 0.02)

        info["action_result"] = "sop_updated"

        self._idle_steps = 0

        return info



    def _handle_flag_for_exec(self, action: SCRMAction, info: dict) -> dict:

        self._on_time_delivery = min(1.0, self._on_time_delivery + 0.02)

        self._quality_score = min(1.0, self._quality_score + 0.01)

        for sup in self._suppliers.values():

            sup["health"] = min(100, sup["health"] + 3)

        info["action_result"] = "escalated"

        self._idle_steps = 0

        return info



    #  Reward & Internal helpers 



    def _compute_reward(self, prev_otd: float, prev_cost: float, prev_health_avg: float, action: SCRMAction, info: dict) -> RewardBreakdown:

        base = 0.05 if self._on_time_delivery > 0.90 else 0.0

        risk = 0.10 if info.get("action_result") in ("mitigation_successful", "recovery_successful") else 0.0

        cost_r = 0.0

        if self._baseline_cost > 0:

            reduction = max(0, prev_cost - self._total_cost)

            cost_r = 0.05 * (reduction / self._baseline_cost)

        cost_r = min(cost_r, 0.10)

        resilience = 0.08 if info.get("action_result") == "supplier_added" else 0.0

        penalty = 0.0

        if info.get("sla_breach"): penalty -= 0.20

        if self._on_time_delivery < prev_otd - 0.05: penalty -= 0.10

        if self._quality_score < 0.90: penalty -= 0.15

        if self._idle_steps >= 5: penalty -= 0.10

        return RewardBreakdown.compute(base, risk, cost_r, resilience, penalty)



    def _build_observation(self, reward: float | None, reward_breakdown: RewardBreakdown | None, done: bool = False) -> SCRMObservation:

        assert self._risk_engine is not None

        return SCRMObservation(

            done=done,

            reward=reward,

            reward_breakdown=reward_breakdown,

            timestamp=self._state.step_count,

            inventory_levels={pid: v["quantity"] for pid, v in self._inventory.items()},

            supplier_status={

                sid: {

                    "status": s["status"],

                    "health_score": round(s["health"], 2),

                    "reliability": round(s["reliability"], 4),

                    "region": s["region"],

                    "capacity_used": s["capacity_used"],

                }

                for sid, s in self._suppliers.items()

            },

            risk_events=self._risk_engine.get_active_events_serializable(),

            cost_metrics={

                "total_cost": round(self._total_cost, 2),

                "cost_variance": round(self._cost_variance, 4),

                "baseline_cost": round(self._baseline_cost, 2),

            },

            delivery_performance={

                "on_time_delivery": round(self._on_time_delivery, 4),

                "quality_score": round(self._quality_score, 4),

            },

            market_conditions={

                "demand_volatility": round(0.10 + self._cost_variance * 0.5, 4),

                "price_index": round(1.0 + self._cost_variance * 0.3, 4),

            },

        )



    def _calc_total_cost(self) -> float:

        inv_cost = sum(v["quantity"] * v["base_cost"] * 0.001 for v in self._inventory.values())

        sup_cost = sum(s["capacity_used"] * 1000 for s in self._suppliers.values())

        degraded_cost = sum((100 - s["health"]) * 50 for s in self._suppliers.values() if s.get("quality_degraded"))

        return round(inv_cost + sup_cost + degraded_cost, 2)



    def _avg_supplier_health(self) -> float:

        if not self._suppliers: return 0.0

        return sum(s["health"] for s in self._suppliers.values()) / len(self._suppliers)



    def _update_performance(self) -> None:

        avg_h = self._avg_supplier_health()

        target_otd = avg_h / 100.0

        self._on_time_delivery = clamp(self._on_time_delivery * 0.70 + target_otd * 0.30, 0.0, 1.0)

        degraded_count = sum(1 for s in self._suppliers.values() if s.get("quality_degraded"))

        if degraded_count > 0:

            self._quality_score = max(0.5, self._quality_score - degraded_count * 0.005)

        rng = random.Random(self._seed + self._state.step_count + 9999)

        for pid, inv in self._inventory.items():

            consumed = rng.uniform(15, 50)

            inv["quantity"] = max(0, inv["quantity"] - consumed)

