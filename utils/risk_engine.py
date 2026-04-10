"""Risk event engine – generation, propagation, and resolution."""

from __future__ import annotations

import random
from typing import Any

from models import RISK_EVENT_TYPES


# Pre-defined event schedules per difficulty for deterministic generation.
# Each entry is (step, event_type, severity, affected_region).
_EASY_EVENTS: list[tuple[int, str, float, str]] = [
    (5, "vendor_failure", 0.3, "asia"),
    (12, "infrastructure", 0.2, "europe"),
]

_MEDIUM_EVENTS: list[tuple[int, str, float, str]] = [
    (2, "natural_disaster", 0.6, "asia"),
    (3, "geopolitical", 0.5, "europe"),
    (10, "vendor_failure", 0.4, "north_america"),
    (18, "infrastructure", 0.5, "asia"),
    (25, "regulatory", 0.4, "europe"),
]

_HARD_EVENTS: list[tuple[int, str, float, str]] = [
    (1, "pandemic", 0.9, "asia"),
    (2, "geopolitical", 0.7, "europe"),
    (4, "natural_disaster", 0.8, "north_america"),
    (5, "infrastructure", 0.6, "latam"),
    (10, "vendor_failure", 0.7, "asia"),
    (15, "pandemic", 0.8, "europe"),
    (20, "regulatory", 0.6, "north_america"),
    (25, "natural_disaster", 0.7, "latam"),
    (30, "geopolitical", 0.8, "asia"),
    (35, "infrastructure", 0.5, "europe"),
    (40, "vendor_failure", 0.6, "north_america"),
]

EVENT_SCHEDULES: dict[str, list[tuple[int, str, float, str]]] = {
    "easy": _EASY_EVENTS,
    "medium": _MEDIUM_EVENTS,
    "hard": _HARD_EVENTS,
}


class RiskEngine:
    """Manages risk event lifecycle: generation, propagation, cascading, resolution."""

    def __init__(self, difficulty: str = "easy", seed: int = 42):
        self._difficulty = difficulty
        self._seed = seed
        self._rng = random.Random(seed)
        self._event_counter = 0
        self._schedule = list(EVENT_SCHEDULES.get(difficulty, _EASY_EVENTS))
        self.active_events: list[dict[str, Any]] = []
        self.resolved_events: list[dict[str, Any]] = []

    # ── Event Generation ───────────────────────────────────────────────────

    def generate_events(
        self,
        step: int,
        suppliers: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Generate any scheduled events for the current *step*.

        Returns list of newly created events.
        """
        new_events: list[dict[str, Any]] = []
        remaining: list[tuple[int, str, float, str]] = []

        for entry in self._schedule:
            trigger_step, event_type, severity, region = entry
            if step == trigger_step:
                event = self._create_event(event_type, severity, region, suppliers)
                new_events.append(event)
                self.active_events.append(event)
            else:
                remaining.append(entry)

        self._schedule = remaining
        return new_events

    def _create_event(
        self,
        event_type: str,
        severity: float,
        region: str,
        suppliers: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Produce a single risk event dict."""
        self._event_counter += 1
        affected = [
            sid for sid, s in suppliers.items() if s.get("region") == region
        ]
        return {
            "id": f"evt_{self._event_counter:04d}",
            "type": event_type,
            "severity": severity,
            "region": region,
            "description": RISK_EVENT_TYPES.get(event_type, event_type),
            "affected_suppliers": affected,
            "status": "active",
            "step_created": 0,
        }

    # ── Propagation ────────────────────────────────────────────────────────

    @staticmethod
    def propagate_risks(
        events: list[dict[str, Any]],
        suppliers: dict[str, dict[str, Any]],
    ) -> None:
        """Apply active risk effects to supplier health & reliability each tick."""
        for event in events:
            if event["status"] != "active":
                continue
            sev = event["severity"]
            for sid in event["affected_suppliers"]:
                sup = suppliers.get(sid)
                if sup is None:
                    continue
                sup["health"] = max(0, sup["health"] - sev * 12.0)
                sup["reliability"] = max(0, sup["reliability"] - sev * 0.04)
                if sup["health"] < 60:
                    sup["quality_degraded"] = True

    # ── Cascade Detection ──────────────────────────────────────────────────

    def cascade_check(
        self,
        suppliers: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """If enough suppliers hit critical, spawn a new cascade event.

        Returns list of any new cascade events generated.
        """
        critical_regions: dict[str, int] = {}
        for sup in suppliers.values():
            if sup["health"] < 30:
                r = sup.get("region", "unknown")
                critical_regions[r] = critical_regions.get(r, 0) + 1

        cascades: list[dict[str, Any]] = []
        for region, count in critical_regions.items():
            if count >= 3:
                # Check we haven't already spawned a cascade for this region
                already = any(
                    e["type"] == "cascade" and e["region"] == region
                    for e in self.active_events
                )
                if not already:
                    evt = self._create_event("cascade", 0.7, region, suppliers)
                    evt["description"] = f"Cascade failure in {region}"
                    cascades.append(evt)
                    self.active_events.append(evt)
        return cascades

    # ── Resolution ─────────────────────────────────────────────────────────

    def resolve_event(self, event_id: str) -> bool:
        """Mark an event as resolved. Returns True if found and resolved."""
        for event in self.active_events:
            if event["id"] == event_id:
                event["status"] = "resolved"
                self.resolved_events.append(event)
                self.active_events = [
                    e for e in self.active_events if e["id"] != event_id
                ]
                return True
        return False

    def mitigate_event(self, event_id: str) -> bool:
        """Mark an event as mitigated (reduced severity). Returns True if found."""
        for event in self.active_events:
            if event["id"] == event_id:
                event["status"] = "mitigated"
                event["severity"] = max(0, event["severity"] - 0.3)
                return True
        return False

    def get_active_events_serializable(self) -> list[dict[str, Any]]:
        """Return active events as plain dicts."""
        return [dict(e) for e in self.active_events]

    # ── Natural decay ──────────────────────────────────────────────────────

    def tick_natural_recovery(
        self,
        suppliers: dict[str, dict[str, Any]],
    ) -> None:
        """Slight natural recovery for suppliers each step."""
        for sup in suppliers.values():
            if sup["health"] < 100:
                sup["health"] = min(100, sup["health"] + 0.15)
            if sup["reliability"] < 1.0:
                sup["reliability"] = min(1.0, sup["reliability"] + 0.0005)
