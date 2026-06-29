# -*- coding: utf-8 -*-
"""THSD Parser — Parse Topological Hyper-Sheaf Dynamics DSL blocks

Extends NeuroMLCompiler with support for THSD syntax:
- complex blocks (simplicial complexes with sheaf-stalks)
- sheaf blocks (constraint bundles)
- formal_spec blocks (mathematical constraints)
"""
import re
from typing import Dict, List, Optional, Any, Tuple
from neuroslm.dsl.thsd_ir import (
    ComplexIR,
    SheafStalkIR,
    TopologyIR,
    CohomologyIR,
    DynamicsIR,
    EmissionKernelIR,
    ReleaseOperatorIR,
    NEMORIConsolidatorIR,
    SheafIR,
    FormalSpecIR,
    LossEquationIR,
    ConvergenceCriteriaIR,
    InformationBottleneckIR,
)


class THSDParser:
    """Robust recursive-descent parser for THSD DSL blocks."""

    @staticmethod
    def tokenize(text: str) -> List[Tuple[str, str]]:
        """Tokenize DSL text into (token_type, value) pairs.

        Token types: 'IDENT', 'NUMBER', 'STRING', 'LBRACE', 'RBRACE', 'COLON', 'COMMA'
        """
        tokens = []
        i = 0
        while i < len(text):
            # Skip whitespace and comments
            if text[i].isspace():
                i += 1
                continue
            if text[i:i+1] == '#':
                # Skip comment line
                i = text.find('\n', i)
                if i == -1:
                    break
                i += 1
                continue

            # String literals
            if text[i] in ('"', "'"):
                quote = text[i]
                j = i + 1
                while j < len(text) and text[j] != quote:
                    if text[j] == '\\':
                        j += 2
                    else:
                        j += 1
                if j < len(text):
                    tokens.append(('STRING', text[i+1:j]))
                    i = j + 1
                else:
                    raise ValueError(f"Unterminated string at position {i}")
                continue

            # Numbers (int and float)
            if text[i].isdigit() or (text[i] == '-' and i+1 < len(text) and text[i+1].isdigit()):
                j = i + (1 if text[i] == '-' else 0)
                while j < len(text) and (text[j].isdigit() or text[j] == '.'):
                    j += 1
                tokens.append(('NUMBER', text[i:j]))
                i = j
                continue

            # Identifiers and keywords
            if text[i].isalpha() or text[i] == '_':
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                tokens.append(('IDENT', text[i:j]))
                i = j
                continue

            # Punctuation
            if text[i:i+1] == '{':
                tokens.append(('LBRACE', '{'))
                i += 1
            elif text[i:i+1] == '}':
                tokens.append(('RBRACE', '}'))
                i += 1
            elif text[i:i+1] == ':':
                tokens.append(('COLON', ':'))
                i += 1
            elif text[i:i+1] == ',':
                tokens.append(('COMMA', ','))
                i += 1
            elif text[i:i+1] == '[':
                tokens.append(('LBRACKET', '['))
                i += 1
            elif text[i:i+1] == ']':
                tokens.append(('RBRACKET', ']'))
                i += 1
            else:
                i += 1

        return tokens

    @staticmethod
    def parse_value(tokens: List[Tuple[str, str]], pos: int) -> Tuple[Any, int]:
        """Parse a single value (string, number, array, or nested dict).

        Returns: (value, new_position)
        """
        if pos >= len(tokens):
            raise ValueError("Unexpected end of input")

        token_type, token_value = tokens[pos]

        if token_type == 'STRING':
            return token_value, pos + 1
        elif token_type == 'NUMBER':
            if '.' in token_value:
                return float(token_value), pos + 1
            else:
                return int(token_value), pos + 1
        elif token_type == 'IDENT':
            if token_value == 'true':
                return True, pos + 1
            elif token_value == 'false':
                return False, pos + 1
            else:
                return token_value, pos + 1
        elif token_type == 'LBRACKET':
            # Parse array
            items = []
            pos += 1
            while pos < len(tokens) and tokens[pos][0] != 'RBRACKET':
                if tokens[pos][0] == 'COMMA':
                    pos += 1
                    continue
                value, pos = THSDParser.parse_value(tokens, pos)
                items.append(value)
            if pos < len(tokens) and tokens[pos][0] == 'RBRACKET':
                pos += 1
            return items, pos
        elif token_type == 'LBRACE':
            # Parse nested dict
            return THSDParser.parse_dict(tokens, pos)
        else:
            raise ValueError(f"Unexpected token: {token_type} = {token_value}")

    @staticmethod
    def parse_dict(tokens: List[Tuple[str, str]], pos: int) -> Tuple[Dict[str, Any], int]:
        """Parse a dictionary from tokens.

        Expected format: { key: value, key: value, ... } or { key { ... }, ... }
        Returns: (dict, new_position)
        """
        result = {}

        # Expect opening brace
        if pos >= len(tokens) or tokens[pos][0] != 'LBRACE':
            raise ValueError(f"Expected '{{', got {tokens[pos] if pos < len(tokens) else 'EOF'}")

        pos += 1

        while pos < len(tokens) and tokens[pos][0] != 'RBRACE':
            # Parse key
            if tokens[pos][0] != 'IDENT':
                raise ValueError(f"Expected identifier for key, got {tokens[pos]}")

            key = tokens[pos][1]
            pos += 1

            # Check for nested dict (no colon) or key-value (with colon)
            if pos < len(tokens) and tokens[pos][0] == 'LBRACE':
                # Nested dict without colon: key { ... }
                value, pos = THSDParser.parse_dict(tokens, pos)
            elif pos < len(tokens) and tokens[pos][0] == 'COLON':
                # Key-value pair: key: value
                pos += 1  # Skip colon
                # Parse value
                value, pos = THSDParser.parse_value(tokens, pos)
            else:
                raise ValueError(f"Expected ':' or '{{' after key, got {tokens[pos] if pos < len(tokens) else 'EOF'}")

            result[key] = value

            # Optional comma
            if pos < len(tokens) and tokens[pos][0] == 'COMMA':
                pos += 1

        # Expect closing brace
        if pos >= len(tokens) or tokens[pos][0] != 'RBRACE':
            raise ValueError(f"Expected '}}', got {tokens[pos] if pos < len(tokens) else 'EOF'}")

        pos += 1
        return result, pos

    @staticmethod
    def extract_complex_blocks(dsl_code: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract all complex blocks from DSL code.

        Returns: List of (name, parsed_dict) tuples
        """
        complexes = []
        tokens = THSDParser.tokenize(dsl_code)

        i = 0
        while i < len(tokens):
            if tokens[i] == ('IDENT', 'complex'):
                i += 1
                if i >= len(tokens) or tokens[i][0] != 'IDENT':
                    raise ValueError("Expected complex name")
                name = tokens[i][1]
                i += 1

                # Tolerate an optional `:` between name and `{`
                # (e.g. `complex GlobalWorkspace: {`).
                if i < len(tokens) and tokens[i][0] == 'COLON':
                    i += 1

                # Parse the block
                block_dict, i = THSDParser.parse_dict(tokens, i)
                complexes.append((name, block_dict))
            else:
                i += 1

        return complexes

    @staticmethod
    def extract_sheaf_blocks(dsl_code: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Extract all sheaf blocks from DSL code.

        Returns: List of (name, parsed_dict) tuples
        """
        sheaves = []
        tokens = THSDParser.tokenize(dsl_code)

        i = 0
        while i < len(tokens):
            if tokens[i] == ('IDENT', 'sheaf'):
                i += 1
                if i >= len(tokens) or tokens[i][0] != 'IDENT':
                    raise ValueError("Expected sheaf name")
                name = tokens[i][1]
                i += 1

                # Tolerate an optional `:` between name and `{`
                # (e.g. `sheaf narrative_consistency: {`).
                if i < len(tokens) and tokens[i][0] == 'COLON':
                    i += 1

                # Parse the block
                block_dict, i = THSDParser.parse_dict(tokens, i)
                sheaves.append((name, block_dict))
            else:
                i += 1

        return sheaves

    @staticmethod
    def parse_stalk(stalk_dict: Dict[str, Any]) -> SheafStalkIR:
        """Parse stalk block into SheafStalkIR."""
        representation_dim = stalk_dict.get("representation_dim")
        fisher_metric = stalk_dict.get("fisher_information_metric", "information_geometry")
        constraints = stalk_dict.get("local_constraints", [])

        if representation_dim is None:
            raise ValueError("stalk block requires representation_dim")

        return SheafStalkIR(
            representation_dim=int(representation_dim),
            fisher_information_metric=fisher_metric,
            local_constraints=constraints if isinstance(constraints, list) else [constraints],
        )

    @staticmethod
    def parse_topology(topology_dict: Dict[str, Any]) -> TopologyIR:
        """Parse topology block into TopologyIR."""
        kind = topology_dict.get("kind", "Tonnetz")
        spectral_gap = topology_dict.get("spectral_gap", 0.3)
        dimension = topology_dict.get("dimension", 8)
        coherence = topology_dict.get("coherence_threshold", 0.95)

        return TopologyIR(
            kind=kind,
            spectral_gap=float(spectral_gap),
            dimension=int(dimension),
            coherence_threshold=float(coherence),
        )

    @staticmethod
    def parse_formal_spec(formal_spec_dict: Dict[str, Any]) -> CohomologyIR:
        """Parse formal_spec block into CohomologyIR."""
        cohom_floor = formal_spec_dict.get("cohomology_floor", 0.01)
        phi_target = formal_spec_dict.get("phi_target", 0.8)
        phi_method = formal_spec_dict.get("phi_method", "geometric_IIT4")

        # Parse information bottleneck if present
        ib_dict = formal_spec_dict.get("information_bottleneck", {})
        if ib_dict:
            ib = InformationBottleneckIR(
                enabled=ib_dict.get("enabled", False),
                compression_ratio=float(ib_dict.get("compression_ratio", 0.7)),
                prediction_lower_bound=float(ib_dict.get("prediction_lower_bound", 0.95)),
            )
        else:
            ib = InformationBottleneckIR()

        return CohomologyIR(
            cohomology_floor=float(cohom_floor),
            phi_target=float(phi_target),
            phi_method=phi_method,
            information_bottleneck=ib,
        )

    @staticmethod
    def parse_dynamics(dynamics_dict: Dict[str, Any]) -> DynamicsIR:
        """Parse dynamics block into DynamicsIR."""
        emission = None
        release = None
        nemori = None

        # Parse emission kernel
        if "emission" in dynamics_dict:
            emit_dict = dynamics_dict["emission"]
            emission = EmissionKernelIR(
                trigger=emit_dict.get("trigger", "always"),
                payload_dim=int(emit_dict.get("payload_dim", 64)),
                lifetime_steps=int(emit_dict.get("lifetime_steps", 100)),
            )

        # Parse release operator
        if "release" in dynamics_dict:
            rel_dict = dynamics_dict["release"]
            release = ReleaseOperatorIR(
                rule=rel_dict.get("rule", "rank_one_update"),
                learning_rate=float(rel_dict.get("learning_rate", 0.001)),
                target=rel_dict.get("target", "parameter_counts"),
            )

        # Parse NEMORI
        if "nemori" in dynamics_dict:
            nem_dict = dynamics_dict["nemori"]
            nemori = NEMORIConsolidatorIR(
                enabled=nem_dict.get("enabled", True),
                consolidation_interval=int(nem_dict.get("consolidation_interval", 1000)),
                forgetting_floor=float(nem_dict.get("forgetting_floor", 0.01)),
            )

        return DynamicsIR(emission=emission, release=release, nemori=nemori)

    @staticmethod
    def parse_complex_block(complex_dict: Dict[str, Any], name: str) -> ComplexIR:
        """Parse a complex block into ComplexIR."""
        # Required: stalk
        if "stalk" not in complex_dict:
            raise ValueError(f"complex {name}: stalk block is required")

        stalk = THSDParser.parse_stalk(complex_dict["stalk"])

        # Optional: topology
        topology = None
        if "topology" in complex_dict:
            topology = THSDParser.parse_topology(complex_dict["topology"])

        # Optional: formal_spec
        formal_spec = None
        if "formal_spec" in complex_dict:
            formal_spec = THSDParser.parse_formal_spec(complex_dict["formal_spec"])

        # Optional: dynamics
        dynamics = None
        if "dynamics" in complex_dict:
            dynamics = THSDParser.parse_dynamics(complex_dict["dynamics"])

        complex_ir = ComplexIR(
            name=name,
            stalk=stalk,
            topology=topology,
            formal_spec=formal_spec,
            dynamics=dynamics,
        )
        complex_ir.validate()
        return complex_ir

    @staticmethod
    def parse_dsl_for_thsd(dsl_code: str) -> Tuple[List[ComplexIR], List[SheafIR]]:
        """Parse DSL code and extract all THSD blocks using tokenization.

        Raises ValueError for invalid THSD blocks (missing stalk, out-of-range
        spectral_gap, out-of-range phi_target, etc.).  Returns empty lists
        only on tokenization / extraction failures (falls back to v2.0).
        """
        complexes_list = []
        sheaves_list = []

        # Try to tokenize - if it fails, fall back to v2.0 parser entirely
        try:
            tokens = THSDParser.tokenize(dsl_code)
        except Exception:
            # If tokenization fails completely, return empty (fall back to v2.0)
            return complexes_list, sheaves_list

        # Extract complex blocks — extraction failure ⇒ fall back to v2.0.
        # Once extracted, however, validation errors MUST propagate so that
        # malformed THSD blocks (missing stalk, negative spectral_gap,
        # phi_target out of [0,1], …) surface to the caller as ValueError.
        try:
            complex_blocks = THSDParser.extract_complex_blocks(dsl_code)
        except Exception:
            complex_blocks = []

        for name, block_dict in complex_blocks:
            # Check if this is THSD syntax (has THSD-specific fields as
            # dicts, not strings).  v2.0 has topology:"Tonnetz" (string),
            # THSD has topology { ... } (dict).
            thsd_dict_fields = {"stalk", "topology", "formal_spec", "dynamics"}
            is_thsd = any(
                isinstance(block_dict.get(field), dict)
                for field in thsd_dict_fields
            )
            if is_thsd:
                # THSD syntax — full validation, ValueError propagates.
                complex_ir = THSDParser.parse_complex_block(block_dict, name)
                complexes_list.append(complex_ir)
            # else: v2.0-style complexes are handled by the legacy
            # _extract_complexes path in compiler.py.

        # Parse sheaf blocks (independent of complex blocks).
        try:
            sheaf_blocks = THSDParser.extract_sheaf_blocks(dsl_code)
            for name, block_dict in sheaf_blocks:
                sheaf_ir = SheafIR(
                    name=name,
                    base_complex=block_dict.get("base_complex", ""),
                    sections=block_dict.get("sections", []),
                    consistency_check=block_dict.get("consistency_check"),
                )
                sheaves_list.append(sheaf_ir)
        except Exception:
            pass

        return complexes_list, sheaves_list