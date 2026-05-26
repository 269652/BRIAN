"""Submechanic building blocks for DSL."""

class Submechanic:
    @staticmethod
    def gate(nt: str, pop: str, gain: float = 0.3, effect: str = "multiplicative") -> str:
        return f'modulation {nt} -> {pop} {{ effect: "{effect}", gain: {gain} }}'

    @staticmethod
    def selection(input_pop: str, output_pop: str, k: int = 1) -> str:
        return f'synapse {output_pop} -> {output_pop} {{ weight: -0.5 }}'

    @staticmethod
    def predictor(source: str, depth: int = 1, feedback: float = 0.1) -> str:
        return f'population {source}_pred {{ count: 32 }}\nsynapse {source} -> {source}_pred {{ weight: 0.5 }}\nsynapse {source}_pred -> {source} {{ weight: {feedback} }}'

    @staticmethod
    def homeostasis(pop: str, setpoint: float = 0.5, tau: float = 0.1) -> str:
        return f'population {pop}_h {{ count: 16, capacity: {setpoint}, timescale: {tau} }}\nsynapse {pop}_h -> {pop} {{ weight: 0.1 }}'

    @staticmethod
    def learn(source: str, target: str, rule: str = "hebb", eta: float = 0.01) -> str:
        return f'synapse {source} -> {target} {{ weight: learnable, plasticity: "{rule}", learning_rate: {eta} }}'

    @staticmethod
    def attractor(pop: str, capacity: float = 1.0, timescale: float = 0.05) -> str:
        return f'synapse {pop} -> {pop} {{ weight: learnable, max_conductance: {capacity} }}'

    @staticmethod
    def router(sources: list, targets: list, n_experts: int = 2) -> str:
        lines = [f'population router_gate {{ count: {n_experts} }}']
        for src in sources:
            lines.append(f'synapse {src} -> router_gate {{ weight: 0.5 }}')
        for tgt in targets:
            lines.append(f'synapse router_gate -> {tgt} {{ weight: 0.5 }}')
        return '\n'.join(lines)

    @staticmethod
    def consistency_check(memory_pop: str, threshold: float = 0.3) -> str:
        return f'sheaf consistency {{ contradiction_threshold: {threshold}, mechanism: "h1_cohomology_proxy" }}'

    @staticmethod
    def value_learner(action: str, state: str, discount: float = 0.99) -> str:
        return f'population reward_signal {{ count: 8 }}\npopulation value_est {{ count: 8, capacity: {discount} }}'

    @staticmethod
    def temporal_binding(sources: list, freq: float = 40.0) -> str:
        tau = 1.0 / (2 * 3.14159 * freq)
        lines = [f'population oscillator {{ count: 32, timescale: {tau} }}']
        for src in sources:
            lines.append(f'synapse oscillator -> {src} {{ weight: 0.05 }}')
        return '\n'.join(lines)

    @staticmethod
    def integrate(modules: list, metric: str = "phi") -> str:
        return f'formal_spec integration {{ rule: "integrated_info", metric: "{metric}" }}'


def compose_serial(*fragments: str) -> str:
    return '\n\n'.join(f for f in fragments if f)


def compose_parallel(*fragments: str) -> str:
    return '\n'.join(f for f in fragments if f)
