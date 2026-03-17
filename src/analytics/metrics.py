from __future__ import annotations

from collections import Counter
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * pct))
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def top_share(values: Sequence[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted((float(v) for v in values), reverse=True)
    take = max(1, int(len(ordered) * share))
    total = sum(ordered)
    if total <= 0:
        return 0.0
    return round(sum(ordered[:take]) / total, 6)


def gini(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(max(0.0, float(v)) for v in values)
    n = len(ordered)
    total = sum(ordered)
    if total <= 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(ordered))
    return round((2.0 * weighted) / (n * total) - (n + 1) / n, 6)


def lineage_energy(agents: Iterable[Dict]) -> Dict[str, float]:
    totals: Counter[str] = Counter()
    for agent in agents:
        totals[str(agent.get("lineage_id"))] += float(agent.get("energy", 0.0))
    return dict(totals)


def role_energy(agents: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[float]] = {}
    for agent in agents:
        role = str(agent.get("role", "unknown"))
        buckets.setdefault(role, []).append(float(agent.get("energy", 0.0)))
    return {
        role: {
            "total_energy": round(sum(vals), 4),
            "median_energy": round(median(vals), 4) if vals else 0.0,
            "mean_energy": round(mean(vals), 4) if vals else 0.0,
        }
        for role, vals in buckets.items()
    }


def rolling_slope(points: Sequence[float]) -> float:
    if len(points) < 2:
        return 0.0
    return round((float(points[-1]) - float(points[0])) / max(1, len(points) - 1), 6)


def detect_phase(
    population_window: Sequence[float],
    births_window: Sequence[float],
    deaths_window: Sequence[float],
    median_energy_window: Sequence[float],
    diversity_window: Sequence[float],
    collaboration_window: Sequence[float],
) -> Dict[str, float | str]:
    pop_slope = rolling_slope(population_window)
    balance = mean(births_window) - mean(deaths_window) if births_window and deaths_window else 0.0
    med_energy_slope = rolling_slope(median_energy_window)
    diversity_slope = rolling_slope(diversity_window)
    collaboration_slope = rolling_slope(collaboration_window)

    phase = "stabilization"
    if pop_slope > 0.2 and balance > 0:
        phase = "expansion"
    elif pop_slope < -0.2 and balance < 0 and med_energy_slope <= 0:
        phase = "collapse"
    elif pop_slope < -0.1 and med_energy_slope < -0.1:
        phase = "overshoot"

    return {
        "phase": phase,
        "rolling_population_slope": pop_slope,
        "rolling_birth_minus_death": round(balance, 6),
        "rolling_median_energy_slope": med_energy_slope,
        "rolling_diversity_slope": diversity_slope,
        "rolling_collaboration_slope": collaboration_slope,
    }
