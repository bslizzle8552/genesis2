from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List


@dataclass
class HealthySwarmWeights:
    population_stability: float = 0.24
    diversity: float = 0.2
    dominance_control: float = 0.2
    intergenerational_health: float = 0.16
    throughput: float = 0.14
    efficiency: float = 0.06


@dataclass
class HealthySwarmGates:
    min_start_population: float = 20.0
    max_start_population: float = 30.0
    max_target_reach_generation: int = 80
    min_target_band_fraction: float = 0.55
    max_population_volatility: float = 20.0
    min_late_avg_population: float = 60.0
    max_top_lineage_share: float = 0.60
    max_top3_lineage_share: float = 0.85
    min_late_lineage_count: float = 3.0
    min_late_births: float = 5.0
    min_solve_rate: float = 0.20


@dataclass
class HealthySwarmScoringConfig:
    target_population: float = 100.0
    target_population_tolerance: float = 20.0
    target_band_min: float = 90.0
    target_band_max: float = 110.0
    target_reach_generation: int = 80
    late_window_fraction: float = 0.25
    collapse_population_threshold: int = 25
    weights: HealthySwarmWeights = field(default_factory=HealthySwarmWeights)
    gates: HealthySwarmGates = field(default_factory=HealthySwarmGates)


class HealthySwarmScorer:
    def __init__(self, config: HealthySwarmScoringConfig | None = None) -> None:
        self.config = config or HealthySwarmScoringConfig()

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _late(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not timeline:
            return []
        late_count = max(1, int(len(timeline) * self.config.late_window_fraction))
        return timeline[-late_count:]

    def _score_population(self, timeline: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
        if not timeline:
            return 0.0, {"reason": "no_timeline"}
        late = self._late(timeline)
        pops = [float(row.get("population", 0)) for row in timeline]
        first_population = pops[0] if pops else 0.0
        late_pops = [float(row.get("population", 0)) for row in late]
        late_avg = mean(late_pops)
        late_std = pstdev(late_pops) if len(late_pops) > 1 else 0.0
        band_min = min(self.config.target_band_min, self.config.target_band_max)
        band_max = max(self.config.target_band_min, self.config.target_band_max)
        target = self.config.target_population
        tol = max(1e-9, self.config.target_population_tolerance)
        closeness = self._clamp01(1.0 - abs(late_avg - target) / tol)
        volatility = self._clamp01(1.0 - (late_std / max(1.0, target * 0.35)))
        band_fraction = sum(1 for p in late_pops if band_min <= p <= band_max) / max(1, len(late_pops))
        start_score = self._clamp01(1.0 - (abs(first_population - 25.0) / 10.0))
        reach_generation = next((idx for idx, p in enumerate(pops) if band_min <= p <= band_max), len(pops))
        reach_score = self._clamp01(1.0 - (max(0, reach_generation - self.config.target_reach_generation) / max(1.0, self.config.target_reach_generation)))
        collapse_count = sum(1 for p in pops if p < self.config.collapse_population_threshold)
        collapse_penalty = self._clamp01(1.0 - (collapse_count / max(1, len(pops))))
        score = (0.25 * closeness) + (0.2 * volatility) + (0.2 * collapse_penalty) + (0.2 * band_fraction) + (0.1 * start_score) + (0.05 * reach_score)
        return self._clamp01(score), {
            "start_population": round(first_population, 4),
            "target_band_min": band_min,
            "target_band_max": band_max,
            "target_reach_generation": self.config.target_reach_generation,
            "generation_reached_target_band": int(reach_generation),
            "late_target_band_fraction": round(band_fraction, 4),
            "late_avg_population": round(late_avg, 4),
            "late_population_std": round(late_std, 4),
            "collapse_generations": collapse_count,
            "target_population": target,
        }

    def _score_diversity(self, timeline: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
        if not timeline:
            return 0.0, {"reason": "no_timeline"}
        late = self._late(timeline)
        lineage_counts = [float(row.get("lineage_count") or row.get("dominance_metrics", {}).get("lineage_count", 0)) for row in timeline]
        late_lineages = lineage_counts[-len(late):]
        late_avg = mean(late_lineages)
        persistence = sum(1 for n in late_lineages if n >= 2) / max(1, len(late_lineages))
        monoculture_penalty = sum(1 for n in late_lineages if n <= 1) / max(1, len(late_lineages))
        lineage_norm = self._clamp01(late_avg / 12.0)
        score = (0.5 * lineage_norm) + (0.35 * persistence) + (0.15 * (1.0 - monoculture_penalty))
        return self._clamp01(score), {
            "late_avg_lineages": round(late_avg, 4),
            "late_diversity_persistence": round(persistence, 4),
            "late_monoculture_fraction": round(monoculture_penalty, 4),
        }

    def _score_dominance(self, timeline: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
        if not timeline:
            return 0.0, {"reason": "no_timeline"}
        late = self._late(timeline)
        top1 = [float(row.get("dominance_metrics", {}).get("top_lineage_energy_share", row.get("top_lineage_energy_share", 0.0))) for row in late]
        top3 = [float(row.get("dominance_metrics", {}).get("top_3_lineage_energy_share", row.get("top_3_lineage_energy_share", 0.0))) for row in late]
        gini = [float(row.get("energy_distribution", {}).get("gini", 1.0)) for row in late]
        repro_conc = [float(row.get("reproduction_concentration_metrics", {}).get("birth_share_top_10pct_agents", 1.0)) for row in late]
        top1_penalty = mean(top1)
        top3_penalty = mean(top3)
        gini_penalty = mean(gini)
        repro_penalty = mean(repro_conc)
        score = 1.0 - ((0.35 * top1_penalty) + (0.25 * top3_penalty) + (0.2 * gini_penalty) + (0.2 * repro_penalty))
        return self._clamp01(score), {
            "late_top_lineage_share": round(top1_penalty, 4),
            "late_top3_lineage_share": round(top3_penalty, 4),
            "late_energy_gini": round(gini_penalty, 4),
            "late_reproduction_concentration": round(repro_penalty, 4),
        }

    def _score_intergenerational(self, timeline: List[Dict[str, Any]], agents: List[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
        late = self._late(timeline)
        late_births = sum(float(row.get("births", 0)) for row in late)
        born_gens = [int(a.get("generation_born", 0)) for a in agents]
        non_founder = sum(1 for g in born_gens if g > 0)
        active_non_founder_share = non_founder / max(1, len(agents))
        max_depth = max(born_gens) if born_gens else 0
        late_birth_score = self._clamp01(late_births / 20.0)
        depth_score = self._clamp01(max_depth / max(1, len(timeline) * 0.5)) if timeline else 0.0
        score = (0.45 * late_birth_score) + (0.35 * active_non_founder_share) + (0.2 * depth_score)
        return self._clamp01(score), {
            "late_births": round(late_births, 4),
            "active_non_founder_share": round(active_non_founder_share, 4),
            "max_descendant_depth": max_depth,
        }

    def _score_throughput(self, timeline: List[Dict[str, Any]], totals: Dict[str, Any], tasks_per_generation: float) -> tuple[float, Dict[str, Any]]:
        solved = float(totals.get("solved", 0))
        total_tasks = max(1.0, float(len(timeline)) * float(tasks_per_generation))
        solve_rate = solved / total_tasks
        return self._clamp01(solve_rate), {
            "solved": solved,
            "total_tasks": total_tasks,
            "solve_rate": round(solve_rate, 6),
        }

    def _score_efficiency(self, timeline: List[Dict[str, Any]], totals: Dict[str, Any]) -> tuple[float, Dict[str, Any]]:
        if not timeline:
            return 0.0, {"reason": "no_timeline"}
        late = self._late(timeline)
        late_avg_pop = mean(float(row.get("population", 0)) for row in late)
        solved = float(totals.get("solved", 0))
        solve_per_agent = solved / max(1.0, late_avg_pop)
        score = self._clamp01(solve_per_agent / 2.0)
        return score, {
            "late_avg_population": round(late_avg_pop, 4),
            "solve_per_agent": round(solve_per_agent, 6),
        }

    def score_run(self, run_result: Dict[str, Any]) -> Dict[str, Any]:
        timeline = list(run_result.get("timeline", []))
        totals = dict(run_result.get("totals", {}))
        agents = list(run_result.get("agents", []))
        tasks_per_generation = float(run_result.get("config", {}).get("tasks_per_generation", 1))

        pop_score, pop_metrics = self._score_population(timeline)
        div_score, div_metrics = self._score_diversity(timeline)
        dom_score, dom_metrics = self._score_dominance(timeline)
        inter_score, inter_metrics = self._score_intergenerational(timeline, agents)
        throughput_score, throughput_metrics = self._score_throughput(timeline, totals, tasks_per_generation)
        eff_score, eff_metrics = self._score_efficiency(timeline, totals)

        weights = self.config.weights
        weighted = {
            "population_stability": pop_score * weights.population_stability,
            "diversity": div_score * weights.diversity,
            "dominance_control": dom_score * weights.dominance_control,
            "intergenerational_health": inter_score * weights.intergenerational_health,
            "throughput": throughput_score * weights.throughput,
            "efficiency": eff_score * weights.efficiency,
        }
        composite = self._clamp01(sum(weighted.values()))

        gates = self.config.gates
        gate_results = {
            "start_population_ok": gates.min_start_population <= pop_metrics.get("start_population", 0.0) <= gates.max_start_population,
            "target_reach_generation_ok": pop_metrics.get("generation_reached_target_band", len(timeline) + 1) <= gates.max_target_reach_generation,
            "target_band_stability_ok": pop_metrics.get("late_target_band_fraction", 0.0) >= gates.min_target_band_fraction,
            "population_volatility_ok": pop_metrics.get("late_population_std", 999.0) <= gates.max_population_volatility,
            "late_population_ok": pop_metrics.get("late_avg_population", 0.0) >= gates.min_late_avg_population,
            "top_lineage_share_ok": dom_metrics.get("late_top_lineage_share", 1.0) <= gates.max_top_lineage_share,
            "top3_lineage_share_ok": dom_metrics.get("late_top3_lineage_share", 1.0) <= gates.max_top3_lineage_share,
            "late_lineage_count_ok": div_metrics.get("late_avg_lineages", 0.0) >= gates.min_late_lineage_count,
            "late_births_ok": inter_metrics.get("late_births", 0.0) >= gates.min_late_births,
            "solve_rate_ok": throughput_metrics.get("solve_rate", 0.0) >= gates.min_solve_rate,
        }

        return {
            "composite_score": round(composite * 100.0, 4),
            "component_scores": {
                "population_stability": round(pop_score * 100.0, 4),
                "diversity": round(div_score * 100.0, 4),
                "dominance_control": round(dom_score * 100.0, 4),
                "intergenerational_health": round(inter_score * 100.0, 4),
                "throughput": round(throughput_score * 100.0, 4),
                "efficiency": round(eff_score * 100.0, 4),
            },
            "weighted_contributions": {k: round(v * 100.0, 4) for k, v in weighted.items()},
            "component_metrics": {
                "population_stability": pop_metrics,
                "diversity": div_metrics,
                "dominance_control": dom_metrics,
                "intergenerational_health": inter_metrics,
                "throughput": throughput_metrics,
                "efficiency": eff_metrics,
            },
            "gates": gate_results,
            "pass": all(gate_results.values()),
        }


def scorer_from_dict(overrides: Dict[str, Any] | None) -> HealthySwarmScorer:
    overrides = overrides or {}
    weights_data = dict(overrides.get("weights", {}))
    gates_data = dict(overrides.get("gates", {}))
    config_data = {k: v for k, v in overrides.items() if k not in {"weights", "gates"}}
    weights = HealthySwarmWeights(**{**HealthySwarmWeights().__dict__, **weights_data})
    gates = HealthySwarmGates(**{**HealthySwarmGates().__dict__, **gates_data})
    cfg = HealthySwarmScoringConfig(**{**HealthySwarmScoringConfig().__dict__, **config_data, "weights": weights, "gates": gates})
    return HealthySwarmScorer(cfg)
