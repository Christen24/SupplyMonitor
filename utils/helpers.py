"""Helper utilities for the SCRM environment."""

from __future__ import annotations

import random
from typing import Any


def clamp(value: float, lo: float = 0.01, hi: float = 0.99) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


def clamp_open_unit_interval(value: float, eps: float = 0.1) -> float:
    """Clamp *value* to the open interval (0, 1) with a configurable margin.

    The hackathon validator requires scores strictly in (0, 1),
    meaning exactly 0.0 and 1.0 are rejected.
    """
    safe_eps = max(1e-6, min(0.499999, float(eps)))
    return max(safe_eps, min(1.0 - safe_eps, float(value)))


def safe_score(value: float) -> float:
    """Unified strictly constrained bounds mapping to [0.05, 0.95]."""
    return round(clamp_open_unit_interval(value, eps=0.05), 4)


def weighted_score(metrics: dict[str, float], weights: dict[str, float]) -> float:
    """Compute weighted sum of *metrics* using *weights* (matching keys)."""
    total = 0.0
    for key, weight in weights.items():
        total += metrics.get(key, 0.1) * weight
    return clamp(total, 0.1, 0.9)


# ── Seeded generators ──────────────────────────────────────────────────────────

REGIONS = ["asia", "europe", "north_america", "latam"]

PART_CATALOG: list[dict[str, Any]] = [
    {"id": "part_001", "name": "Semiconductor Chip A", "min_level": 200, "reorder": 500, "base_cost": 12.50},
    {"id": "part_002", "name": "PCB Board B", "min_level": 300, "reorder": 600, "base_cost": 8.00},
    {"id": "part_003", "name": "Display Panel C", "min_level": 150, "reorder": 400, "base_cost": 25.00},
    {"id": "part_004", "name": "Battery Module D", "min_level": 250, "reorder": 550, "base_cost": 18.00},
    {"id": "part_005", "name": "Connector Assembly E", "min_level": 400, "reorder": 800, "base_cost": 3.50},
    {"id": "part_006", "name": "Casing Frame F", "min_level": 350, "reorder": 700, "base_cost": 6.00},
]


def generate_supplier_network(
    rng: random.Random,
    regions: list[str] | None = None,
    suppliers_per_region: tuple[int, int] = (3, 6),
) -> dict[str, dict[str, Any]]:
    """Create a seeded supplier network.

    Returns ``{supplier_id: {health, reliability, region, capacity_used, ...}}``.
    """
    regions = regions or REGIONS
    suppliers: dict[str, dict[str, Any]] = {}
    for i, region in enumerate(regions):
        n = rng.randint(*suppliers_per_region)
        for j in range(n):
            sid = f"sup_{region}_{j}"
            suppliers[sid] = {
                "health": round(rng.uniform(80, 100), 2),
                "reliability": round(rng.uniform(0.85, 1.0), 4),
                "region": region,
                "capacity_used": round(rng.uniform(0.3, 0.7), 2),
                "active_events": [],
                "status": "healthy",
                "quality_degraded": False,
            }
    return suppliers


def generate_inventory(
    rng: random.Random,
    parts: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, float]]:
    """Create seeded starting inventory.

    Returns ``{part_id: {quantity, min_level, reorder_point, base_cost}}``.
    """
    parts = parts or PART_CATALOG
    inv: dict[str, dict[str, float]] = {}
    for p in parts:
        qty = round(rng.uniform(p["reorder"] * 0.8, p["reorder"] * 1.5), 0)
        inv[p["id"]] = {
            "quantity": qty,
            "min_level": p["min_level"],
            "reorder_point": p["reorder"],
            "base_cost": p["base_cost"],
        }
    return inv


def supplier_status_label(health: float) -> str:
    """Return a status string based on health score."""
    if health >= 75:
        return "healthy"
    elif health >= 50:
        return "warning"
    elif health >= 25:
        return "critical"
    return "inactive"
