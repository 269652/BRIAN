# -*- coding: utf-8 -*-
"""End-to-end: `Resolver` lazily ingests shared-lib files.

An architecture's ``arch.neuro`` can import equations or features from
``<repo>/lib/``. ``FolderLoader`` only walks the arch directory, so the
``Resolver`` itself must:

1. Detect ``@brian/`` and ``@lib/`` import specifiers during the
   import-resolution pass.
2. Resolve them to absolute paths via PathResolver. ``@brian/<x>``
   is repo-root anchored (so ``@brian/lib/equations`` →
   ``<repo>/lib/equations.neuro``). ``@lib/<x>`` is the cleaner
   shorthand anchored directly at ``<repo>/lib/``.
3. Lazily load + parse those files, registering them in
   ``program.modules`` so the standard cross-file resolution machinery
   (Pass 2 imports, Pass 3 dynamics/function bodies) sees them.
4. Recursively: a shared-lib file may itself import from the shared
   lib (under either prefix) or from ``./`` siblings inside
   ``<repo>/lib/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from neuroslm.dsl.multifile import Resolver, ResolverError


def _mkfile(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def fake_repo(tmp_path: Path):
    """Materialise a repo skeleton (2026-06-15 layout — lib at repo root):

        <tmp>/
            pyproject.toml
            lib/
                equations.neuro        (exports eq_a, eq_b)
                features/
                    hyperbolic.neuro   (imports eq_a from @lib/equations)
            architectures/
                master/
                    arch.neuro         (imports eq_a from @lib/equations)
    """
    _mkfile(tmp_path / "pyproject.toml", "[project]\nname='brian'\n")

    _mkfile(
        tmp_path / "lib" / "equations.neuro",
        """
        export equation eq_a {
            params: [x],
            formula: "y = x"
        }
        export equation eq_b {
            params: [x, w],
            formula: "y = w * x"
        }
        """,
    )

    _mkfile(
        tmp_path / "lib" / "features" / "hyperbolic.neuro",
        """
        import { eq_a } from "@lib/equations"

        export feature hyperbolic {
            equation: eq_a,
            active: false,
            params: { d_model: 64 }
        }
        """,
    )

    _mkfile(
        tmp_path / "architectures" / "master" / "arch.neuro",
        """
        import { eq_a } from "@lib/equations"

        architecture master {
            d_sem: 64,
            dt: 0.01
        }

        population dummy {
            count: 8,
            dynamics: "rate_code"
        }

        feature shared_feat {
            equation: eq_a,
            active: false
        }
        """,
    )

    arch = tmp_path / "architectures" / "master"
    return tmp_path, arch


class TestResolverLazyLoadsBrianLib:
    def test_arch_can_import_from_lib_equations(self, fake_repo):
        """The minimum: arch.neuro imports an equation from
        ``@lib/equations`` and the resolver loads that file even
        though it's outside the arch directory."""
        repo_root, arch = fake_repo
        program = Resolver(arch).resolve()
        # The shared lib file is now part of program.modules
        lib_eq = repo_root / "lib" / "equations.neuro"
        assert lib_eq.resolve() in program.modules

    def test_lib_imports_resolve_to_correct_export(self, fake_repo):
        """The lookup machinery sees the shared equation."""
        _, arch = fake_repo
        program = Resolver(arch).resolve()
        arch_neuro = (arch / "arch.neuro").resolve()
        decl = program.lookup(arch_neuro, "eq_a")
        assert "eq_a" in decl  # the raw declaration text

    def test_transitive_lib_imports_are_loaded(self, fake_repo):
        """A shared-lib file's own ``@lib/`` import is followed."""
        repo_root, arch = fake_repo
        # Add a third layer: arch imports hyperbolic from
        # @lib/features/hyperbolic, which itself imports eq_a from
        # @lib/equations. The resolver must load both.
        _mkfile(
            arch / "arch.neuro",
            """
            import { hyperbolic } from "@lib/features/hyperbolic"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        program = Resolver(arch).resolve()
        lib_feat = repo_root / "lib" / "features" / "hyperbolic.neuro"
        lib_eq = repo_root / "lib" / "equations.neuro"
        assert lib_feat.resolve() in program.modules
        assert lib_eq.resolve() in program.modules

    def test_brian_long_form_also_works(self, fake_repo):
        """``@brian/lib/<x>`` is the long form for ``@lib/<x>``;
        both prefixes must reach the same file."""
        repo_root, arch = fake_repo
        _mkfile(
            arch / "arch.neuro",
            """
            import { eq_a } from "@brian/lib/equations"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        program = Resolver(arch).resolve()
        lib_eq = repo_root / "lib" / "equations.neuro"
        assert lib_eq.resolve() in program.modules

    def test_unknown_lib_import_raises_resolver_error(self, fake_repo):
        repo_root, arch = fake_repo
        _mkfile(
            arch / "arch.neuro",
            """
            import { missing_eq } from "@lib/does_not_exist"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        with pytest.raises(ResolverError, match="does_not_exist"):
            Resolver(arch).resolve()

    def test_lib_import_of_undefined_export_raises(self, fake_repo):
        _, arch = fake_repo
        _mkfile(
            arch / "arch.neuro",
            """
            import { not_exported } from "@lib/equations"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        with pytest.raises(ResolverError, match="not_exported"):
            Resolver(arch).resolve()
