"""Episodic memory: stores recent events, thoughts, and interactions."""
from collections import deque
import threading
import time

class EpisodicMemory:
    def __init__(self, maxlen=2048):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def add(self, content, content_vec=None, nt_state=None, emotion=None, tags=None, context=None):
        """Append an episode. content_vec can be a numeric vector (optional)."""
        episode = {
            'content': content,
            'timestamp': time.time(),
            'content_vec': content_vec,
            'nt_state': nt_state,
            'emotion': emotion,
            'tags': tags or [],
            'context': context or {},
        }
        with self.lock:
            self.buffer.append(episode)

    def recent(self, n=32):
        with self.lock:
            return list(self.buffer)[-n:]

    def all(self):
        with self.lock:
            return list(self.buffer)

    def retrieve_scored(self, query_vec, k=4):
        """Similarity read with scores: top-k ``(cosine, episode)``
        pairs by similarity between ``query_vec`` and each stored
        ``content_vec``. Episodes without a vector are skipped — they
        were stored before the embedding path existed and cannot
        participate in similarity recall. Pure-python math so the
        memory stays importable without numpy/torch. The scores make
        this usable both for recall (which episodes) and for semantic
        novelty (HOW similar is the nearest one).
        """
        def _cos(a, b):
            if len(a) != len(b):
                return -1.0
            dot = sum(x * y for x, y in zip(a, b))
            na = sum(x * x for x in a) ** 0.5
            nb = sum(x * x for x in b) ** 0.5
            if na == 0.0 or nb == 0.0:
                return -1.0
            return dot / (na * nb)

        q = list(query_vec)
        with self.lock:
            episodes = [e for e in self.buffer
                        if e.get("content_vec") is not None]
        scored = [(_cos(q, list(e["content_vec"])), e) for e in episodes]
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[:max(0, int(k))]

    def retrieve(self, query_vec, k=4):
        """Similarity read (hippocampal recall): top-k episodes only —
        see :meth:`retrieve_scored` for the scored variant this
        delegates to."""
        return [e for _, e in self.retrieve_scored(query_vec, k)]
