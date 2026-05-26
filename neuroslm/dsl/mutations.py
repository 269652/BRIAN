"""Mutation operators for circuit evolution."""
import random
import re
from .compiler import NeuroMLCompiler


class MutationError(Exception):
    pass


def _validate(source: str) -> str:
    try:
        NeuroMLCompiler.compile(source)
        return source
    except:
        raise MutationError("Invalid circuit")


def add_modulation(source: str, nt: str, target: str, effect: str = "multiplicative", gain: float = 0.3) -> str:
    if target not in source or nt not in source and nt not in ['dopamine', 'serotonin', 'gaba', 'glutamate']:
        raise MutationError("Invalid target or NT")
    return _validate(source + f'\nmodulation {nt} -> {target} {{ effect: "{effect}", gain: {gain} }}')


def add_feedback(source: str, src: str, tgt: str, weight: float = 0.1) -> str:
    if src not in source or tgt not in source:
        raise MutationError("Invalid populations")
    return _validate(source + f'\nsynapse {tgt} -> {src} {{ weight: {weight} }}')


def split_population(source: str, pop: str, n: int = 2) -> str:
    if n < 2:
        raise MutationError("n_sub must be >= 2")
    if f"population {pop}" not in source:
        raise MutationError("Population not found")
    return _validate(source)


def add_gating(source: str, nt: str, pop: str) -> str:
    if pop not in source:
        raise MutationError("Population not found")
    return _validate(source + f'\nmodulation {nt} -> {pop} {{ effect: "multiplicative", gain: 0.3 }}')


def add_plasticity(source: str, src: str, tgt: str, rule: str = "hebb") -> str:
    if rule not in ["hebb", "stdp", "bcm", "none"]:
        raise MutationError("Invalid rule")
    return _validate(source)


def add_prediction_loop(source: str, pop: str, strength: float = 0.1) -> str:
    if f"population {pop}" not in source:
        raise MutationError("Population not found")
    return _validate(source + f'\npopulation {pop}_pred {{ count: 32 }}\nsynapse {pop} -> {pop}_pred {{ weight: 0.5 }}\nsynapse {pop}_pred -> {pop} {{ weight: {strength} }}')


def merge_populations(source: str, p1: str, p2: str) -> str:
    if p1 not in source or p2 not in source:
        raise MutationError("Population not found")
    return _validate(source)


def add_homeostatic_regulation(source: str, pop: str) -> str:
    if f"population {pop}" not in source:
        raise MutationError("Population not found")
    return _validate(source + f'\npopulation {pop}_h {{ count: 16, capacity: 0.5 }}\nsynapse {pop}_h -> {pop} {{ weight: 0.1 }}')


def add_sheaf_consistency(source: str, pop: str, threshold: float = 0.3) -> str:
    if pop not in source:
        raise MutationError("Population not found")
    return _validate(source + f'\nsheaf consistency {{ contradiction_threshold: {threshold}, mechanism: "h1_cohomology_proxy" }}')


def mutate_numeric(source: str, sigma: float = 0.1, rate: float = 0.3) -> str:
    def perturb(m):
        if random.random() > rate:
            return m.group(0)
        num = float(m.group(0))
        new_num = max(0.001, num * (1.0 + random.gauss(0, sigma)))
        return f"{new_num:.6f}".rstrip('0').rstrip('.')
    return _validate(re.sub(r'\b\d+\.?\d*\b', perturb, source))


def crossover(a: str, b: str) -> str:
    lines_a = a.split('\n')
    lines_b = b.split('\n')
    child = lines_a + lines_b[len(lines_a)//2:]
    return _validate('\n'.join(child))


ALL_MUTATIONS = [add_modulation, add_feedback, split_population, add_gating, add_plasticity,
                 add_prediction_loop, merge_populations, add_homeostatic_regulation, add_sheaf_consistency,
                 mutate_numeric, crossover]
