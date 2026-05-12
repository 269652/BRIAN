"""Comprehension gate for episodic memory writing.

Decides whether an observation is worth storing as an episode.
Combines three signals:
  1. Surprise      — NLL of observation under the model (high = unexpected)
  2. Comprehension — cosine similarity of observation vs predicted embedding
                     (high = model can integrate it into existing schema)
  3. Novelty       — 1 - max cosine sim to consolidated nodes
                     (high = concept not already stored)

write_score = surprise * comprehension * novelty
write       = write_score > threshold   (~10% write rate target)

This product selects observations that are *new*, *surprising*, AND
*understandable* — the operational definition of a learning insight.
An adaptive threshold tracks the target write rate via EMA.
"""
from __future__ import annotations
import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a / (np.linalg.norm(a) + 1e-9)) @
                 (b / (np.linalg.norm(b) + 1e-9)))


class ComprehensionGate:
    """Decides whether to store an observation as an episode."""

    def __init__(self, threshold: float = 0.05,
                 novelty_topk: int = 16,
                 target_write_rate: float = 0.10,
                 ema_alpha: float = 0.01,
                 nemori_floor: float = 0.0):
        self.threshold        = threshold
        self.novelty_topk     = novelty_topk
        self.target_write_rate = target_write_rate
        self.ema_alpha        = ema_alpha
        self._write_rate_ema  = target_write_rate
        self._n_evaluated     = 0
        self._n_written       = 0
        # NEMORI prior: only information that is *not* predicted by the
        # current anticipatory schema deserves retention. Concretely:
        # gate write iff (raw_NLL − anticipated_NLL) > nemori_floor.
        # Setting nemori_floor to 0 keeps backwards-compatible behaviour
        # (no filtering); a positive floor enforces predictive forgetting.
        self.nemori_floor = float(nemori_floor)

    def evaluate(self,
                 obs_vec: np.ndarray,
                 predicted_vec: np.ndarray | None,
                 surprise: float,
                 consolidated,
                 anticipated_surprise: float | None = None) -> dict:
        # Comprehension: how well the model's prediction matched reality
        if predicted_vec is None:
            comprehension = 0.5
        else:
            obs_v  = np.asarray(obs_vec,       dtype=np.float32).flatten()
            pred_v = np.asarray(predicted_vec, dtype=np.float32).flatten()
            d = min(obs_v.size, pred_v.size)
            comprehension = max(0.0, _cos(obs_v[:d], pred_v[:d]))

        # Novelty: 1 - max cosine sim to existing consolidated nodes
        novelty = 1.0
        try:
            obs_v = np.asarray(obs_vec, dtype=np.float32).flatten()
            sims  = []
            nodes = list(consolidated.graph.nodes(data=True))[-256:]
            for _, data in nodes:
                cv = data.get("content_vec")
                if cv is None:
                    continue
                cv = np.asarray(cv, dtype=np.float32).flatten()
                d  = min(obs_v.size, cv.size)
                sims.append(_cos(obs_v[:d], cv[:d]))
            if sims:
                novelty = float(1.0 - max(sims))
        except Exception:
            pass

        # Normalise surprise from raw NLL range
        surp  = max(0.0, min(1.0, surprise / 6.0))

        # ── NEMORI distillation prior ─────────────────────────────────────
        # Predictive-forgetting principle: only retain what the model could
        # not anticipate. We compute unpredicted_surprise = surprise −
        # anticipated_surprise; if it's below the floor, the event is
        # already explained by the model's current schema and we reject the
        # write regardless of the surprise×comprehension×novelty score.
        nemori_kept = True
        unpredicted_surprise = surprise
        if anticipated_surprise is not None:
            unpredicted_surprise = float(surprise) - float(anticipated_surprise)
            if unpredicted_surprise < self.nemori_floor:
                nemori_kept = False

        score = surp * comprehension * novelty
        write = bool(score > self.threshold) and nemori_kept

        # Adaptive threshold to track target_write_rate
        self._n_evaluated += 1
        self._n_written   += int(write)
        cur_rate = self._n_written / max(self._n_evaluated, 1)
        self._write_rate_ema = ((1 - self.ema_alpha) * self._write_rate_ema
                                + self.ema_alpha * cur_rate)
        if self._write_rate_ema > self.target_write_rate * 1.2:
            self.threshold *= 1.005
        elif self._write_rate_ema < self.target_write_rate * 0.8:
            self.threshold *= 0.995
        self.threshold = max(1e-4, min(0.5, self.threshold))

        return {
            "write":                write,
            "score":                score,
            "surprise":             surp,
            "comprehension":        comprehension,
            "novelty":              novelty,
            "threshold":            self.threshold,
            "write_rate_ema":       self._write_rate_ema,
            "unpredicted_surprise": unpredicted_surprise,
            "nemori_kept":          nemori_kept,
        }

    def stats(self) -> dict:
        return {
            "threshold":      self.threshold,
            "write_rate_ema": self._write_rate_ema,
            "n_evaluated":    self._n_evaluated,
            "n_written":      self._n_written,
        }
