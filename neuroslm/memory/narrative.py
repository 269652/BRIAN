"""Autobiographic and world narrative buffers, updated as new memories form."""
import threading

class NarrativeBuffer:
    def __init__(self, maxlen=4096):
        self.buffer = []
        self.maxlen = maxlen
        self.lock = threading.Lock()

    def update(self, text):
        with self.lock:
            self.buffer.append(text)
            if len(self.buffer) > self.maxlen:
                self.buffer = self.buffer[-self.maxlen:]

    def get(self, n=32):
        with self.lock:
            return self.buffer[-n:]

    def all(self):
        with self.lock:
            return list(self.buffer)
