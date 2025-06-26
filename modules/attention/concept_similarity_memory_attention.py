import torch
import torch.nn as nn
import torch.nn.functional as F

class ConceptSimilarityMemoryAttention(nn.Module):
    """
    Attends over episodic memory based on similarity to current state.
    Can be weighted by emotional salience or recency.
    """

    def __init__(self, hidden_dim, memory_slots):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_slots = memory_slots

        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, memory_keys, memory_values, emotion_weights=None):
        """
        Args:
            query: [B, D] - current agent state or attention focus
            memory_keys: [B, M, D] - memory access keys (e.g., summaries)
            memory_values: [B, M, D] - episodic content or representations
            emotion_weights: Optional [B, M] - emotional tagging (valence/priority)

        Returns:
            context: [B, D] - weighted memory readout
        """
        q = self.query_proj(query).unsqueeze(1)  # [B, 1, D]
        k = self.key_proj(memory_keys)           # [B, M, D]
        v = self.value_proj(memory_values)       # [B, M, D]

        # Cosine similarity
        sim = F.cosine_similarity(q, k, dim=-1)  # [B, M]

        if emotion_weights is not None:
            sim = sim * emotion_weights

        attn = F.softmax(sim, dim=-1).unsqueeze(-1)  # [B, M, 1]
        context = torch.sum(attn * v, dim=1)         # [B, D]

        return context
