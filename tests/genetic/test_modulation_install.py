# -*- coding: utf-8 -*-
"""Install banked discovery winners into the next training run.

The probe banks site-tagged winners (`<run>_L<k>_step<n>`) with measured Δ.
Installing them must be evidence-gated: a winner is installable only when it
RECURS (same site + semantically-equal program across probes/batches) with
positive mean Δ — a single-batch Δ is a candidate, not a mechanism. At install
time each selection is re-validated on a real batch (keep only if CE does not
get worse) so a stale winner from an older checkpoint can't hurt the new run.
"""
import torch

from neuroslm.genetic.language import Instruction, Program
from neuroslm.genetic.modulation_store import ModulationRecord, ModulationStore
from neuroslm.genetic.modulation_install import (
    parse_site,
    select_installable,
    install_from_store,
)


def _prog(const=1.0):
    return Program([Instruction("cscale", "t5", ("t0",), const=const)],
                   n_scalar=8, n_tensor=16, out_reg="t5")


def _rec(name, prog, delta):
    return ModulationRecord(name=name, program=prog,
                            metrics={"delta_ce": delta, "step": 1.0})


class TestParseSite:
    def test_extracts_layer_from_probe_names(self):
        assert parse_site("trunk_p_L4_step500") == 4
        assert parse_site("trunk_rcc_bowtie_30m_p4_L11_step2000") == 11

    def test_none_for_siteless_names(self):
        assert parse_site("run_0_step4000") is None


class TestSelectInstallable:
    def test_requires_recurrence(self):
        recs = [_rec("a_L2_step1", _prog(1.1), 0.003)]      # seen once → no
        assert select_installable(recs, min_count=2) == []
        recs.append(_rec("a_L2_step2", _prog(1.1), 0.002))  # recurs → yes
        picked = select_installable(recs, min_count=2)
        assert len(picked) == 1
        assert picked[0].layer == 2
        assert picked[0].count == 2

    def test_requires_positive_mean_delta(self):
        recs = [_rec("a_L2_step1", _prog(1.1), 0.001),
                _rec("a_L2_step2", _prog(1.1), -0.005)]      # mean < 0 → no
        assert select_installable(recs, min_count=2) == []

    def test_groups_by_site_and_program_semantics(self):
        # same program at DIFFERENT sites must not pool evidence
        recs = [_rec("a_L1_step1", _prog(1.1), 0.002),
                _rec("a_L3_step2", _prog(1.1), 0.002)]
        assert select_installable(recs, min_count=2) == []
        # different programs at the SAME site must not pool either
        recs = [_rec("a_L1_step1", _prog(1.1), 0.002),
                _rec("a_L1_step2", _prog(0.5), 0.002)]
        assert select_installable(recs, min_count=2) == []

    def test_best_group_per_layer(self):
        recs = [_rec("a_L1_step1", _prog(1.1), 0.001),
                _rec("a_L1_step2", _prog(1.1), 0.001),
                _rec("a_L1_step3", _prog(0.5), 0.010),
                _rec("a_L1_step4", _prog(0.5), 0.010)]
        picked = select_installable(recs, min_count=2)
        assert len(picked) == 1                    # one winner per layer
        assert abs(picked[0].mean_delta - 0.010) < 1e-9


class TestSingleShotStrictGate:
    """A winner banked ONCE can still install — but only by strictly improving
    a FRESH batch at install time (probe batch + install batch = 2-fold
    cross-batch validation on the same weights). Recurring winners (count>=2)
    keep the lenient not-worse gate."""

    def test_single_shot_installs_only_on_strict_improvement(self):
        from neuroslm.genetic.modulation_install import (Selection,
                                                         install_modulations)

        class _LM:
            _layer_modulations: dict = {}
        sel = Selection(layer=1, program=_prog(1.0), mean_delta=0.01,
                        count=1, name="t_L1_step1")

        vals = iter([2.0, 1.9])            # improves clearly → installs
        _LM._layer_modulations = {}
        rep = install_modulations(_LM, [sel], val_fn=lambda: next(vals))
        assert len(rep["installed"]) == 1

        vals = iter([2.0, 1.99995])        # "not worse" is NOT enough for n=1
        _LM._layer_modulations = {}
        rep = install_modulations(_LM, [sel], val_fn=lambda: next(vals))
        assert rep["installed"] == [] and len(rep["rejected"]) == 1

    def test_recurring_keeps_the_lenient_not_worse_gate(self):
        from neuroslm.genetic.modulation_install import (Selection,
                                                         install_modulations)

        class _LM:
            _layer_modulations: dict = {}
        sel = Selection(layer=1, program=_prog(1.0), mean_delta=0.01,
                        count=3, name="t_L1_step1")
        vals = iter([2.0, 2.0])            # neutral is fine when it recurs
        rep = install_modulations(_LM, [sel], val_fn=lambda: next(vals))
        assert len(rep["installed"]) == 1


class TestInstallFromStore:
    def _cortex(self):
        from neuroslm.dsl.nn_lang import build_dsl_language_cortex
        torch.manual_seed(0)
        return build_dsl_language_cortex(vocab=61, d_model=24, depth=3,
                                         n_heads=4, max_ctx=16, dropout=0.0)

    def _batch(self, seed=7):
        g = torch.Generator().manual_seed(seed)
        return (torch.randint(0, 61, (2, 12), generator=g),
                torch.randint(0, 61, (2, 12), generator=g))

    def test_validation_gate_rejects_harmful_winners(self, tmp_path):
        store = ModulationStore(tmp_path / "mods")
        # recurring "winner" whose install would blow the hidden up 100x —
        # whatever its recorded Δ claims, the live gate must reject it
        store.save(_rec("t_L1_step1", _prog(100.0), 0.5))
        store.save(_rec("t_L1_step2", _prog(100.0), 0.5))
        m = self._cortex()
        ids, targets = self._batch()
        report = install_from_store(m, tmp_path / "mods", ids, targets,
                                    min_count=2)
        assert report["installed"] == []
        assert len(report["rejected"]) == 1
        assert m._layer_modulations == {}

    def test_neutral_winner_installs_and_takes_effect(self, tmp_path):
        store = ModulationStore(tmp_path / "mods")
        # identity gain (const 1.0 → h×1) — measured-neutral, passes the gate
        neutral = Program([Instruction("const", "t5", (), const=1.0)],
                          n_scalar=8, n_tensor=16, out_reg="t5")
        store.save(_rec("t_L1_step1", neutral, 0.001))
        store.save(_rec("t_L1_step2", neutral, 0.001))
        m = self._cortex()
        ids, targets = self._batch()
        report = install_from_store(m, tmp_path / "mods", ids, targets,
                                    min_count=2)
        assert len(report["installed"]) == 1
        assert 1 in m._layer_modulations

    def test_empty_store_is_a_clean_noop(self, tmp_path):
        m = self._cortex()
        ids, targets = self._batch()
        report = install_from_store(m, tmp_path / "nothing-here", ids, targets)
        assert report["installed"] == [] and report["rejected"] == []
        assert m._layer_modulations == {}
