# -*- coding: utf-8 -*-
"""End-to-end: `Resolver` lazily ingests `@brian/` shared-lib files.

An architecture's ``arch.neuro`` can import equations or features from
``<repo>/lib/``. ``FolderLoader`` only walks the arch directory, so the
``Resolver`` itself must:

1. Detect ``@brian/`` import specifiers during the import-resolution pass.
2. Resolve them to absolute paths via the (now @brian-aware) PathResolver.
3. Lazily load + parse those files, registering them in
   ``program.modules`` so the standard cross-file resolution machinery
   (Pass 2 imports, Pass 3 dynamics/function bodies) sees them.
4. Recursively: a shared-lib file may itself import from ``@brian/`` or
   from ``./`` siblings inside ``<repo>/lib/``.
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
    """Materialise a repo skeleton:

        <tmp>/
            pyproject.toml
            architectures/
                lib/
                    equations.neuro        (exports eq_a, eq_b)
                    features/
                        hyperbolic.neuro   (imports eq_a from @brian/equations)
                master/
                    arch.neuro             (imports eq_a from @brian/equations)
    """
    _mkfile(tmp_path / "pyproject.toml", "[project]\nname='brian'\n")

    _mkfile(
        tmp_path / "architectures" / "lib" / "equations.neuro",
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
        tmp_path / "architectures" / "lib" / "features" / "hyperbolic.neuro",
        """
        import { eq_a } from "@brian/equations"

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
        import { eq_a } from "@brian/equations"

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
    def test_arch_can_import_from_brian_equations(self, fake_repo):
        """The minimum: arch.neuro imports an equation from
        ``@brian/equations`` and the resolver loads that file even
        though it's outside the arch directory."""
        repo_root, arch = fake_repo
        program = Resolver(arch).resolve()
        # The shared lib file is now part of program.modules
        lib_eq = repo_root / "architectures" / "lib" / "equations.neuro"
        assert lib_eq.resolve() in program.modules

    def test_brian_imports_resolve_to_correct_export(self, fake_repo):
        """The lookup machinery sees the shared equation."""
        _, arch = fake_repo
        program = Resolver(arch).resolve()
        arch_neuro = (arch / "arch.neuro").resolve()
        decl = program.lookup(arch_neuro, "eq_a")
        assert "eq_a" in decl  # the raw declaration text

    def test_transitive_brian_imports_are_loaded(self, fake_repo):
        """A shared-lib file's own ``@brian/`` import is followed."""
        repo_root, arch = fake_repo
        # Add a third layer: arch imports hyperbolic from
        # @brian/features/hyperbolic, which itself imports eq_a from
        # @brian/equations. The resolver must load both.
        _mkfile(
            arch / "arch.neuro",
            """
            import { hyperbolic } from "@brian/features/hyperbolic"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        program = Resolver(arch).resolve()
        lib_feat = (
            repo_root / "architectures" / "lib" / "features"
            / "hyperbolic.neuro"
        )
        lib_eq = repo_root / "architectures" / "lib" / "equations.neuro"
        assert lib_feat.resolve() in program.modules
        assert lib_eq.resolve() in program.modules

    def test_unknown_brian_import_raises_resolver_error(self, fake_repo):
        repo_root, arch = fake_repo
        _mkfile(
            arch / "arch.neuro",
            """
            import { missing_eq } from "@brian/does_not_exist"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        with pytest.raises(ResolverError, match="does_not_exist"):
            Resolver(arch).resolve()

    def test_brian_import_of_undefined_export_raises(self, fake_repo):
        _, arch = fake_repo
        _mkfile(
            arch / "arch.neuro",
            """
            import { not_exported } from "@brian/equations"

            architecture master { d_sem: 64, dt: 0.01 }
            population dummy { count: 8, dynamics: "rate_code" }
            """,
        )
        with pytest.raises(ResolverError, match="not_exported"):
            Resolver(arch).resolve()
