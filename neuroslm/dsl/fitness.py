"""Fitness metrics for circuit evaluation."""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import math


@dataclass
class FitnessVector:
    structural_complexity: float
    modularity_score: float
    recurrence_ratio: float
    neuromodulation_coverage: float
    information_integration_proxy: float
    biological_plausibility: float

    def to_dict(self) -> Dict:
        return {
            "structural_complexity": self.structural_complexity,
            "modularity_score": self.modularity_score,
            "recurrence_ratio": self.recurrence_ratio,
            "neuromodulation_coverage": self.neuromodulation_coverage,
            "information_integration_proxy": self.information_integration_proxy,
            "biological_plausibility": self.biological_plausibility,
        }


def structural_complexity(ir) -> float:
    n_nodes = len(ir.populations) + len(ir.neurotransmitter_systems)
    n_edges = len(ir.synapses) + len(ir.modulations)
    total = n_nodes + n_edges
    return 1.0 / (1.0 + math.exp(-0.1 * (total - 25)))


def modularity_score(ir) -> float:
    if len(ir.synapses) < 2:
        return 0.0
    edges = [(s.source, s.target) for s in ir.synapses]
    if not edges:
        return 0.0
    adj = {}
    for src, tgt in edges:
        if src not in adj:
            adj[src] = set()
        adj[src].add(tgt)
    triangles = 0
    possible_triangles = 0
    for src, tgt in edges:
        if src in adj and tgt in adj:
            common = adj[src] & adj[tgt]
            triangles += len(common)
            possible_triangles += 1
    if possible_triangles == 0:
        return 0.0
    return min(1.0, triangles / (possible_triangles * 2.0 + 1e-6))


def recurrence_ratio(ir) -> float:
    if not ir.synapses:
        return 0.0
    recurrent = sum(1 for s in ir.synapses if s.source == s.target)
    return recurrent / len(ir.synapses)


def neuromodulation_coverage(ir) -> float:
    if not ir.populations or not ir.modulations:
        return 0.0
    avg_mods = len(ir.modulations) / max(len(ir.populations), 1)
    return 1.0 / (1.0 + math.exp(-3.0 * (avg_mods - 0.5)))


def information_integration_proxy(ir) -> float:
    if len(ir.populations) < 2:
        return 0.0
    pop_names = {p.name for p in ir.populations}
    edges = [(s.source, s.target) for s in ir.synapses if s.source in pop_names and s.target in pop_names]
    if not edges:
        return 0.0
    n_pairs = len(pop_names) * (len(pop_names) - 1)
    if n_pairs == 0:
        return 0.0
    connectivity = len(edges) / n_pairs
    log_scale = math.log(max(len(pop_names), 2))
    return min(1.0, connectivity * log_scale / 2.0)


def biological_plausibility(ir) -> float:
    score = 0.0
    n_checks = 0
    for pop in ir.populations:
        n_checks += 1
        if 10 <= pop.count <= 10000:
            score += 1.0
        elif pop.count > 0:
            score += 0.5
    for nt in ir.neurotransmitter_systems:
        n_checks += 1
        if 0.0 <= nt.base_concentration <= 1.0:
            score += 1.0
        else:
            score += 0.5
    for mod in ir.modulations:
        n_checks += 1
        if 0.1 <= abs(mod.gain) <= 10.0:
            score += 1.0
        elif 0.0 <= abs(mod.gain) <= 100.0:
            score += 0.7
    if n_checks == 0:
        return 0.5
    return score / n_checks


def compute_fitness(ir) -> FitnessVector:
    return FitnessVector(
        structural_complexity=structural_complexity(ir),
        modularity_score=modularity_score(ir),
        recurrence_ratio=recurrence_ratio(ir),
        neuromodulation_coverage=neuromodulation_coverage(ir),
        information_integration_proxy=information_integration_proxy(ir),
        biological_plausibility=biological_plausibility(ir),
    )


def composite_fitness(fv: FitnessVector, weights: Dict = None) -> float:
    if weights is None:
        weights = {k: 1.0/6.0 for k in ["structural_complexity", "modularity_score", "recurrence_ratio", "neuromodulation_coverage", "information_integration_proxy", "biological_plausibility"]}
    fv_dict = fv.to_dict()
    total = sum(fv_dict.get(k, 0.0) * w for k, w in weights.items())
    return min(1.0, max(0.0, total))


def pareto_dominates(a: FitnessVector, b: FitnessVector) -> bool:
    metrics = [a.structural_complexity >= b.structural_complexity, a.modularity_score >= b.modularity_score,
               a.recurrence_ratio >= b.recurrence_ratio, a.neuromodulation_coverage >= b.neuromodulation_coverage,
               a.information_integration_proxy >= b.information_integration_proxy, a.biological_plausibility >= b.biological_plausibility]
    strict = [a.structural_complexity > b.structural_complexity, a.modularity_score > b.modularity_score,
              a.recurrence_ratio > b.recurrence_ratio, a.neuromodulation_coverage > b.neuromodulation_coverage,
              a.information_integration_proxy > b.information_integration_proxy, a.biological_plausibility > b.biological_plausibility]
    return all(metrics) and any(strict)


def pareto_frontier(population: List[Tuple[Any, FitnessVector]]) -> List[Any]:
    if not population:
        return []
    frontier = []
    for candidate, candidate_fitness in population:
        dominated = False
        for other, other_fitness in population:
            if pareto_dominates(other_fitness, candidate_fitness):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier
