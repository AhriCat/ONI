import threading, time
import numpy as np
from typing import Dict, Any

# Field: small vector + metadata
class Field:
    def __init__(self, dim=256):
        self.vec = np.zeros(dim, dtype=np.float32)
        self.lock = threading.Lock()
        self.version = 0
        self.meta = {}

    def read(self):
        return self.vec.copy(), self.version, dict(self.meta)

    def commit(self, delta: np.ndarray, author: str):
        with self.lock:
            self.vec += delta
            self.version += 1
            self.meta['last_author'] = author
            self.meta['last_time'] = time.time()
            return self.version

# Module adapter: returns proposal and a confidence score
class ModuleAdapter:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.energy = 1.0  # simple motivation scalar

    def read_and_propose(self, field_vec):
        # model maps field -> delta, confidence
        delta, conf = self.model.infer(field_vec)
        return delta, float(conf)

# Simple arbiter: pick highest UCB-like score * energy
class Arbiter:
    def __init__(self, modules):
        self.modules = modules
        self.counts = {m.name:1 for m in modules}
        self.values = {m.name:0.0 for m in modules}
        self.total = sum(self.counts.values())

    def select(self, proposals):
        # proposals: {name: (delta, conf, energy)}
        best = None; best_score = -1e9
        for name,(delta,conf,energy) in proposals.items():
            ucb = conf + 0.1 * np.sqrt(np.log(self.total)/self.counts[name])
            score = ucb * (energy**0.5)
            if score>best_score:
                best_score=score; best=name
        self.counts[best]+=1; self.total+=1
        return best

# Example usage omitted for brevity.
