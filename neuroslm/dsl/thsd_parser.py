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
    """Parser for THSD DSL blocks."""

    @staticmethod
    def extract_balanced_braces(text: str, start_pos: int) -> Tuple[str, int]:
        """Extract balanced brace block starting at position.

        Returns: (content, end_position)
        """
        depth = 0
        start = text.find("{", start_pos)
        if start == -1:
            raise ValueError("No opening brace found")

        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start + 1 : i].strip(), i + 1

        raise ValueError("Unbalanced braces")

    @staticmethod
    def parse_key_value(line: str) -> Tuple[str, Any]:
        """Parse key: value or key: {...} line.

        Returns: (key, value)
        """
        match = re.match(r"(\w+)\s*:\s*(.*)", line.strip())
        if not match:
            raise ValueError(f"Invalid key-value syntax: {line}")

        key = match.group(1)
        value_str = match.group(2).strip()

        # Try to parse as various types
        if value_str == "true":
            return key, True
        elif value_str == "false":
            return key, False
        elif value_str.startswith('"') and value_str.endswith('"'):
            return key, value_str[1:-1]
        elif re.match(r"-?\d+(\.\d+)?", value_str):
            if "." in value_str:
                return key, float(value_str)
            else:
                return key, int(value_str)
        elif value_str.endswith((",", "}")):
            value_str = value_str.rstrip(",}")
            if value_str.startswith("[") and value_str.endswith("]"):
                # Parse array
                items_str = value_str[1:-1]
                items = [s.strip().strip('"') for s in items_str.split(",") if s.strip()]
                return key, items
            else:
                # Single value
                if value_str.startswith('"') and value_str.endswith('"'):
                    return key, value_str[1:-1]
                else:
                    try:
                        return key, float(value_str)
                    except ValueError:
                        try:
                            return key, int(value_str)
                        except ValueError:
                            return key, value_str

        return key, value_str

    @staticmethod
    def parse_nested_block(block_text: str, block_name: str) -> Dict[str, Any]:
        """Parse a nested block like { ... } and return dict of fields."""
        result = {}
        # Remove outer braces if present
        content = block_text.strip()
        if content.startswith("{"):
            content = content[1:]
        if content.endswith("}"):
            content = content[:-1]

        lines = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("#"):
                i += 1
                continue

            # Check if this line has a nested block
            if "{" in line:
                # Extract key and nested block
                key_part = line[: line.index("{")].strip().rstrip(":")
                if ":" in key_part:
                    key = key_part.split(":")[-1].strip()
                else:
                    key = key_part

                # Extract nested braces content - just get from { to matching }
                start = line.index("{")
                brace_content = line[start:]
                depth = 0
                end_pos = 0
                for j, c in enumerate(brace_content):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end_pos = j
                            break
                nested_content = brace_content[1:end_pos]
                result[key] = THSDParser.parse_nested_block(nested_content, key)
                i += 1
            else:
                # Regular key-value pair
                try:
                    key, value = THSDParser.parse_key_value(line)
                    result[key] = value
                except ValueError:
                    pass
                i += 1

        return result

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
    def extract_complex_blocks(dsl_code: str) -> List[Tuple[str, str]]:
        """Extract all complex blocks from DSL code.

        Returns: List of (name, block_content) tuples
        """
        complexes = []
        pattern = r"complex\s+(\w+)\s*\{"
        matches = list(re.finditer(pattern, dsl_code))

        for match in matches:
            name = match.group(1)
            start_pos = match.start(0)
            block_content, _ = THSDParser.extract_balanced_braces(dsl_code, start_pos)
            complexes.append((name, block_content))

        return complexes

    @staticmethod
    def extract_sheaf_blocks(dsl_code: str) -> List[Tuple[str, str]]:
        """Extract all sheaf blocks from DSL code.

        Returns: List of (name, block_content) tuples
        """
        sheaves = []
        pattern = r"sheaf\s+(\w+)\s*\{"
        matches = list(re.finditer(pattern, dsl_code))

        for match in matches:
            name = match.group(1)
            start_pos = match.start(0)
            block_content, _ = THSDParser.extract_balanced_braces(dsl_code, start_pos)
            sheaves.append((name, block_content))

        return sheaves

    @staticmethod
    def parse_dsl_for_thsd(dsl_code: str) -> Tuple[List[ComplexIR], List[SheafIR]]:
        """Parse DSL code and extract all THSD blocks.

        Returns: (complexes, sheaves)
        """
        complexes_list = []
        sheaves_list = []

        # Parse complex blocks
        complex_blocks = THSDParser.extract_complex_blocks(dsl_code)
        for name, block_content in complex_blocks:
            complex_dict = THSDParser.parse_nested_block(block_content, "complex")
            complex_ir = THSDParser.parse_complex_block(complex_dict, name)
            complexes_list.append(complex_ir)

        # Parse sheaf blocks
        sheaf_blocks = THSDParser.extract_sheaf_blocks(dsl_code)
        for name, block_content in sheaf_blocks:
            sheaf_dict = THSDParser.parse_nested_block(block_content, "sheaf")
            sheaf_ir = SheafIR(
                name=name,
                base_complex=sheaf_dict.get("base_complex", ""),
                sections=sheaf_dict.get("sections", []),
                consistency_check=sheaf_dict.get("consistency_check"),
            )
            sheaves_list.append(sheaf_ir)

        return complexes_list, sheaves_list
