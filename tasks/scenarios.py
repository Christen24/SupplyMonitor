"""Scenario generator and deterministic task graders."""



from __future__ import annotations



from typing import Any



from utils.helpers import clamp, clamp_open_unit_interval, weighted_score, safe_score





#  Scenario definitions 





TASK_CONFIGS: dict[str, dict[str, Any]] = {

    "easy": {

        "id": "easy",

        "name": "Routine Operations",

        "difficulty": "easy",

        "duration_steps": 20,

        "seed": 42,

        "objective": "Maintain supplier operations during minor disruptions",

        "success_criteria": {

            "min_delivery_rate": 0.90,

            "max_failures": 3,

        },

        "initial_risk_level": "low",

        "grader_weights": {

            "delivery_rate": 0.40,

            "cost_variance": 0.20,

            "risk_mitigation": 0.20,

            "supplier_health": 0.15,

            "sla_compliance": 0.05,

        },

    },

    "medium": {

        "id": "medium",

        "name": "Multi-Region Crisis",

        "difficulty": "medium",

        "duration_steps": 35,

        "seed": 42,

        "objective": "Handle concurrent supplier disruptions across regions",

        "success_criteria": {

            "min_delivery_rate": 0.80,

            "max_cost_impact": 0.15,

            "required_actions": ["diversify", "mitigate", "recover"],

        },

        "initial_risk_level": "medium",

        "grader_weights": {

            "delivery_rate": 0.30,

            "cost_variance": 0.30,

            "risk_mitigation": 0.25,

            "supplier_health": 0.10,

            "sla_compliance": 0.05,

        },

    },

    "hard": {

        "id": "hard",

        "name": "Global Cascade Failure",

        "difficulty": "hard",

        "duration_steps": 50,

        "seed": 42,

        "objective": "Manage global supply chain crisis with cascading impacts",

        "success_criteria": {

            "min_delivery_rate": 0.70,

            "max_cost_impact": 0.25,

            "recovery_required": True,

            "escalations_required": 2,

        },

        "initial_risk_level": "high",

        "grader_weights": {

            "delivery_rate": 0.25,

            "cost_variance": 0.25,

            "risk_mitigation": 0.25,

            "supplier_health": 0.15,

            "sla_compliance": 0.10,

        },

    },

}





class ScenarioGenerator:

    """Creates task configuration dicts for easy / medium / hard tasks."""



    @staticmethod

    def create_task(difficulty: str) -> dict[str, Any]:

        """Return a copy of the scenario config for the given *difficulty*."""

        if difficulty not in TASK_CONFIGS:

            raise ValueError(f"Unknown difficulty: {difficulty!r}. Use easy/medium/hard.")

        return dict(TASK_CONFIGS[difficulty])



    @staticmethod

    def list_tasks() -> list[dict[str, Any]]:

        return [dict(v) for v in TASK_CONFIGS.values()]





#  Grader 





class TaskGrader:

    """Deterministic grader that scores a completed episode trajectory.



    The grader computes five metric components, applies difficulty-based

    scaling, and deducts for unresolved events and SLA breaches to

    produce a final score  [0.0, 1.0].

    """



    # Difficulty scaling caps -- raw weighted score is multiplied by this.

    # This ensures harder tasks naturally produce lower scores.

    _DIFFICULTY_CAPS: dict[str, float] = {

        "easy": 0.92,

        "medium": 0.72,

        "hard": 0.68,

    }



    def grade(

        self,

        task_id: str,

        trajectory: list[dict[str, Any]],

        final_state: dict[str, Any],

    ) -> float:

        """Grade a task completion.



        Parameters

        ----------

        task_id : str

            One of ``"easy"``, ``"medium"``, ``"hard"``.

        trajectory : list[dict]

            List of ``{"observation": ..., "action": ..., "reward": ...}`` dicts

            for every step in the episode.

        final_state : dict

            The environment ``state()`` at episode end.



        Returns

        -------

        float

            Score in [0.0, 1.0].

        """

        config = TASK_CONFIGS.get(task_id)

        if config is None:

            raise ValueError(f"Unknown task_id: {task_id!r}")



        metrics = self._compute_metrics(trajectory, final_state, config)

        raw_score = weighted_score(metrics, config["grader_weights"])



        # Apply difficulty cap

        cap = self._DIFFICULTY_CAPS.get(task_id, 0.95)

        scaled = raw_score * cap



        # Deductions for unresolved events at episode end

        unresolved = len(final_state.get("risk_events", []))

        scaled -= unresolved * 0.02



        # Deduction for SLA breaches

        sla_breaches = final_state.get("sla_breaches", 0)

        scaled -= sla_breaches * 0.015



        # Deduction for low final OTD

        final_otd = final_state.get("performance", {}).get("on_time_delivery", 1.0)

        if final_otd < 0.85:

            scaled -= (0.85 - final_otd) * 0.25



        # Strict Hackathon Validator Compliance: score must be in range (0, 1) exclusive.
        # safe_score tightly bounds to [0.05, 0.95].
        return safe_score(scaled)



    #  Metric computation 



    def _compute_metrics(

        self,

        trajectory: list[dict[str, Any]],

        final_state: dict[str, Any],

        config: dict[str, Any],

    ) -> dict[str, float]:

        return {

            "delivery_rate": self._delivery_rate(trajectory, final_state),

            "cost_variance": self._cost_variance(trajectory, final_state),

            "risk_mitigation": self._risk_mitigation(trajectory),

            "supplier_health": self._supplier_health(trajectory, final_state),

            "sla_compliance": self._sla_compliance(trajectory),

        }



    @staticmethod

    def _delivery_rate(

        trajectory: list[dict[str, Any]],

        final_state: dict[str, Any],

    ) -> float:

        """Average on-time delivery across the episode."""

        if not trajectory:

            return 0.1

        rates = []

        for step in trajectory:

            obs = step.get("observation", {})

            perf = obs.get("delivery_performance", {})

            rates.append(perf.get("on_time_delivery", 0.0))

        return clamp(sum(rates) / len(rates), 0.1, 0.9) if rates else 0.1



    @staticmethod

    def _cost_variance(

        trajectory: list[dict[str, Any]],

        final_state: dict[str, Any],

    ) -> float:

        """1 - (avg cost_variance  amplifier), so higher variance hits harder."""

        if not trajectory:

            return 0.1

        variances = []

        for step in trajectory:

            obs = step.get("observation", {})

            cm = obs.get("cost_metrics", {})

            variances.append(cm.get("cost_variance", 0.0))

        avg_var = sum(variances) / len(variances) if variances else 0.1

        # Amplify cost variance impact (3) so it drags score down meaningfully

        return clamp(1.0 - avg_var * 3.0, 0.1, 0.9)



    @staticmethod

    def _risk_mitigation(trajectory: list[dict[str, Any]]) -> float:

        """Ratio of successful mitigations to total risk events encountered."""

        successful = 0

        risk_events_seen = set()

        for step in trajectory:

            info = step.get("info", {})

            if info.get("action_result") in ("mitigation_successful", "recovery_successful"):

                successful += 1

            obs = step.get("observation", {})

            for evt in obs.get("risk_events", []):

                risk_events_seen.add(evt.get("id", ""))



        total_risks = max(len(risk_events_seen), 1)

        return clamp(successful / total_risks, 0.1, 0.9)



    @staticmethod

    def _supplier_health(

        trajectory: list[dict[str, Any]],

        final_state: dict[str, Any],

    ) -> float:

        """Average supplier health normalised to [0, 1]."""

        suppliers = final_state.get("suppliers", {})

        if not suppliers:

            return 0.1

        healths = [s.get("health", 0) for s in suppliers.values()]

        return clamp(sum(healths) / (len(healths) * 100), 0.1, 0.9)



    @staticmethod

    def _sla_compliance(trajectory: list[dict[str, Any]]) -> float:

        """1 - (breach_ratio  2) for amplified penalty."""

        breaches = 0

        for step in trajectory:

            info = step.get("info", {})

            if info.get("sla_breach"):

                breaches += 1

        total = max(len(trajectory), 1)

        return clamp(1.0 - (breaches / total) * 2.0, 0.1, 0.9)
