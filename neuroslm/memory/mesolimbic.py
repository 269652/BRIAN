"""Mesolimbic system: tags memories with dopaminergic reward/insight signals."""
import threading

class MesolimbicTagger:
    def __init__(self):
        self.lock = threading.Lock()
        self.tags = {}

    def tag(self, memory_id, reward, insight=None):
        with self.lock:
            self.tags[memory_id] = {
                'reward': reward,
                'insight': insight,
            }

    def get_tag(self, memory_id):
        with self.lock:
            return self.tags.get(memory_id, None)

    def all_tags(self):
        with self.lock:
            return dict(self.tags)
