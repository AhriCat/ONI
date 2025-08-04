# memory.py
import torch
import torch.nn as nn

class WorkingMemoryModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.memory_bank = torch.zeros(config.working_memory_size, config.hidden_dim)
        self.timestamps = torch.zeros(config.working_memory_size)
    
    def update_memory(self, embedding, timestamp):
        idx = self._find_slot_to_update()
        self.memory_bank[idx] = embedding
        self.timestamps[idx] = timestamp
    
    def retrieve_memory(self, query_embedding, current_timestamp):
        time_deltas = current_timestamp - self.timestamps
        decay_weights = torch.exp(-time_deltas / self.config.temporal_context_size)
        weighted_memories = self.memory_bank * decay_weights.unsqueeze(-1)
        return weighted_memories.mean(dim=0)

    def _find_slot_to_update(self):
        # Simple FIFO logic or least-recently-used logic
        return torch.argmin(self.timestamps)
