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
