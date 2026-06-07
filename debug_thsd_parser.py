#!/usr/bin/env python3
"""Debug THSD parser."""
from neuroslm.dsl.thsd_parser import THSDParser

dsl = """
complex LanguageCortex {
    stalk {
        representation_dim: 512,
        fisher_information_metric: "information_geometry"
    }
}
"""

print("Testing THSDParser...")
print(f"DSL input:\n{dsl}\n")

# Test extract_complex_blocks
complex_blocks = THSDParser.extract_complex_blocks(dsl)
print(f"Found {len(complex_blocks)} complex blocks:")
for name, content in complex_blocks:
    print(f"  Name: {name}")
    print(f"  Content preview: {content[:80]}...")

# Test parse_nested_block
print("\nTesting parse_nested_block...")
for name, content in complex_blocks:
    complex_dict = THSDParser.parse_nested_block(content, "complex")
    print(f"Complex dict keys: {list(complex_dict.keys())}")
    if "stalk" in complex_dict:
        print(f"Stalk dict: {complex_dict['stalk']}")
    else:
        print("ERROR: No stalk in complex_dict!")

# Test full parsing
print("\nTesting full parse_dsl_for_thsd...")
complexes, sheaves = THSDParser.parse_dsl_for_thsd(dsl)
print(f"Found {len(complexes)} complexes")
for c in complexes:
    print(f"  - {c.name}: stalk dim={c.stalk.representation_dim}")
