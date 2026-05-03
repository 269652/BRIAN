"""Comprehension gate for episodic writing.

Not every observation should become a memory. The gate decides which
events are worth storing as episodes (and later as consolidated nodes).
This is what makes QA training produce *concept memories* rather than
verbatim memorization.

The gate combines three signals:

  1. **Surprise** — KL or NLL of the observation under the model. High
     surprise = the model didn't already predict this; might be worth
     remembering.

  2. **Comprehension** — semantic-coherence score: cosine similarity
     between the observation embedding and the model's predicted next
     embedding. High coherence = the model can integrate the observation
     into its existing schema (it understood it).

  3. **Novelty** — minimum cosine distance to existing consolidated
     nodes. High novelty = the concept is not already stored.

Decision rule:

    write_score = surprise * comprehension * novelty
    write       = write_score > theta

Why all three? Surprise alone leads to memorization of noise (high-
surprise = random tokens). Comprehension alone biases toward already-
known content. Novelty alone biases toward weird inputs. The product
selects observations that are *new*, *surprising*, *and understandable*
— the operational definition of an "insight".
"""
from __future__ import annotations
import math
import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float((a / (np.linalg.norm(a) + 1e-9)) @
                 (b / (np.linalg.norm(b) + 1e-9)))


class ComprehensionGate:
    """Decides whether to store an observation as an episode.

    Args:
        threshold:    write_score above this triggers a write.
        novelty_topk: # of consolidated nodes used to compute novelty.
        ema_alpha:    smoothing for adaptive threshold (auto-tunes to
                      keep write rate near `target_write_rate`).
        target_write_rate: fraction of observations expected to be stored.
    """

    def __init__(self, threshold: float = 0.05,
                 novelty_topk: int = 16,
                 target_write_rate: float = 0.10,
                 ema_alpha: float = 0.01):
        self.threshold = threshold
        self.novelty_topk = novelty_topk
        self.target_write_rate = target_write_rate
        self.ema_alpha = ema_alpha
        self._write_rate_ema = target_write_rate
        self._n_evaluated = 0
        self._n_written = 0

    def evaluate(self,
                 obs_vec: np.ndarray,
                 predicted_vec: np.ndarray | None,
                 surprise: float,
                 consolidated) -> dict:
        """Evaluate a single observation. Returns a dict with the
        component scores and a `write: bool` decision.
        """
        # Comprehension: how well our prediction matched
        if predicted_vec is None:
            comprehension = 0.5
        else:
            obs_v = np.asarray(obs_vec, dtype=np.float32).flatten()
            pred_v = np.asarray(predicted_vec, dtype=np.float32).flatten()
            d = min(obs_v.size, pred_v.size)
            comprehension = max(0.0, _cos(obs_v[:d], pred_v[:d]))

        # Novelty: 1 - max cosine sim to existing nodes
        novelty = 1.0
        try:
            obs_v = np.asarray(obs_vec, dtype=np.float32).flatten()
            sims = []
            nodes = list(consolidated.graph.nodes(data=True))[-256:]
            for _, data in nodes:
                cv = data.get("content_vec")
                if cv is None: continue
                cv = np.asarray(cv, dtype=np.float32).flatten()
                d = min(obs_v.size, cv.size)
                sims.append(_cos(obs_v[:d], cv[:d]))
            if sims:
                novelty = float(1.0 - max(sims))
        except Exception:
            pass

        # Surprise: clamp and normalize a typical NLL range
        surp = max(0.0, min(1.0, surprise / 6.0))

        score = surp * comprehension * novelty
        write = score > self.threshold

        # Adaptive threshold to track target_write_rate
        self._n_evaluated += 1
        self._n_written += int(write)
        cur_rate = self._n_written / max(self._n_evaluated, 1)
        self._write_rate_ema = (1 - self.ema_alpha) * self._write_rate_ema \
                               + self.ema_alpha * cur_rate
        if self._write_rate_ema > self.target_write_rate * 1.2:
            self.threshold *= 1.005
        elif self._write_rate_ema < self.target_write_rate * 0.8:
            self.threshold *= 0.995
        self.threshold = max(1e-4, min(0.5, self.threshold))

        return {
            "write": write,
            "score": score,
            "surprise": surp,
            "comprehension": comprehension,
            "novelty": novelty,
            "threshold": self.threshold,
            "write_rate_ema": self._write_rate_ema,
        }

    def stats(self) -> dict:
        return {
            "threshold": self.threshold,
            "write_rate_ema": self._write_rate_ema,
            "n_evaluated": self._n_evaluated,
            "n_written": self._n_written,
        }
