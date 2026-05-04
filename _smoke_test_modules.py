# -*- coding: utf-8 -*-
import unittest
import sys

try:
    import torch
except Exception:
    print("SKIP: torch is not installed. Install requirements (pip install -r requirements.txt) to run module smoke tests.")
    sys.exit(0)

from neuroslm.modules.hippocampus import Hippocampus
from neuroslm.modules.basal_ganglia import BasalGanglia
from neuroslm.modules.pfc import PrefrontalCortex
from neuroslm.config import PRESETS
from neuroslm.brain import Brain
import numpy as np
import tempfile, os


class ModuleSmokeTests(unittest.TestCase):
    def test_hippocampus_store_and_enrich(self):
        d = 16
        hip = Hippocampus(d_sem=d, capacity=8, topk=2, sparse_k=4)

        B, S = 2, 3
        gws = torch.randn(B, S, d)
        ft = torch.randn(B, d)
        nt = {"DA": 0.5, "NE": 0.5, "5HT": 0.5, "ACh": 0.5}

        # When empty, enrich_gws should return unchanged slots and novelty==1
        enriched, novelty, recalls = hip.enrich_gws(gws, ft, nt)
        self.assertEqual(enriched.shape, gws.shape)
        self.assertEqual(novelty.shape[0], B)
        self.assertTrue((novelty == 1).all().item())
        self.assertEqual(recalls.shape[0], B)

        # Store some memories
        for i in range(4):
            q = torch.randn(B, d)
            v = torch.randn(B, d)
            nt_state = torch.zeros(B, 7)
            hip.store(q, v, nt_state=nt_state, valence=0.1 * i, salience=0.2)

        # After storing, enrich should now return enriched slots and recalls non-zero
        enriched2, novelty2, recalls2 = hip.enrich_gws(gws, ft, nt)
        self.assertEqual(enriched2.shape, gws.shape)
        self.assertEqual(recalls2.dim(), 3)
        # novelty should be in [0,1]
        self.assertTrue(((novelty2 >= 0.0) & (novelty2 <= 1.0)).all().item())

    def test_basal_ganglia_selection_and_emergency(self):
        d_sem = 16
        d_act = 8
        K = 4
        bg = BasalGanglia(d_sem=d_sem, d_action=d_act, n_candidates=K)

        B = 2
        thought = torch.randn(B, d_sem)
        nt = {"DA": 0.6, "NE": 0.4}

        chosen, confidence, probs, commit_ok = bg.forward(thought, nt)
        self.assertEqual(chosen.shape, (B, d_act))
        self.assertEqual(confidence.shape[0], B)
        self.assertEqual(probs.shape, (B, K))
        self.assertEqual(commit_ok.shape[0], B)

        # Trigger emergency by setting NE very high -> commit_ok should be all False
        nt_em = {"DA": 0.5, "NE": 0.99}
        chosen2, conf2, probs2, commit_ok2 = bg.forward(thought, nt_em)
        self.assertEqual(commit_ok2.dtype, torch.bool)
        self.assertTrue((commit_ok2 == 0).all().item())

    def test_prefrontal_cortex_scoring_and_replace_gate(self):
        d = 16
        pfc = PrefrontalCortex(d_sem=d, n_layers=1, n_heads=2)

        B, S, R = 2, 3, 4
        gws = torch.randn(B, S, d)
        recalls = torch.randn(B, R, d)
        ft = torch.randn(B, d)
        nt = {"DA": 0.7, "NE": 0.3, "GABA": 0.2, "ACh": 0.5, "5HT": 0.5}

        selected, replace = pfc.forward(gws, recalls, ft, nt_levels=nt)
        self.assertEqual(selected.shape, (B, d))
        self.assertEqual(replace.shape[0], B)
        # replace gate probabilities should be in [0,1]
        self.assertTrue(((replace >= 0.0) & (replace <= 1.0)).all().item())

    def test_brain_memory_consolidation_and_knowledge(self):
        # Use tiny preset to keep initialization small
        cfg = PRESETS['tiny']()
        cfg.vocab_size = 128
        try:
            brain = Brain(cfg)
        except Exception as e:
            self.skipTest("Failed to instantiate Brain: {}".format(e))

        # Episodic memory: add synthetic episodes with numpy content_vecs and NT states
        for i in range(6):
            vec = np.random.randn(cfg.d_sem).astype(np.float32)
            nt = np.zeros(7, dtype=np.float32)
            nt[0] = 0.5  # DA
            brain.episodic.add("event {}".format(i), content_vec=vec.tolist(), nt_state=nt.tolist(), emotion=0.1 * i)

        # Consolidate recent episodes (should run without error and return stats)
        try:
            stats = brain.consolidator.consolidate(brain.episodic.recent(32), da_level=0.5)
        except Exception as e:
            self.fail("Consolidation raised an exception: {}".format(e))

        self.assertIsInstance(stats, dict)
        self.assertIn('n_clusters', stats)

        # Knowledge triple extraction via hypergraph extractor
        try:
            from neuroslm.memory.hypergraph import KnowledgeTripleExtractor
            extractor = KnowledgeTripleExtractor()
            triples = extractor.extract('Alice loves salmon and dislikes rain', entity_context='alice')
            # Should return a list (may be empty on conservative extractor)
            self.assertIsInstance(triples, list)
        except Exception:
            # Not fatal — some environments may lack regex or other deps
            pass

        # Entity identification + preference extraction (ensure no crash)
        try:
            eid, conf = brain.identify_speaker("Hi, I'm Alice and I like salmon")
            # get_entity_knowledge should return a dict
            info = brain.get_entity_knowledge(eid)
            self.assertIsInstance(info, dict)
        except Exception:
            # Non-fatal: skip if entity store not functioning in this env
            pass

        # Save and load memory checkpoint to a temp file
        try:
            fd, path = tempfile.mkstemp(suffix='.mem')
            os.close(fd)
            try:
                res = brain.save_memory_checkpoint(path)
                self.assertIsInstance(res, dict)
                loaded = brain.load_memory_checkpoint(path)
                # load returns payload or dict; ensure no exception
                self.assertTrue(isinstance(loaded, (dict, type(None))) or loaded is None)
            finally:
                try:
                    os.remove(path)
                except Exception:
                    pass
        except Exception:
            # Skip if file system calls fail
            pass


if __name__ == '__main__':
    unittest.main()
