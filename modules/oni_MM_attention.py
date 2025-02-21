import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalAttention(nn.Module):
    """
    Computes attention over multiple modalities (vision, audio) with respect to NLP features.

    This implementation performs attention separately for each modality to improve scalability
    and then combines the attended outputs.

    Args:
        dim (int): The input feature dimension for all modalities.
    """
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)  # Query projection for NLP
        self.k_proj = nn.Linear(dim, dim)  # Key projection for vision and audio
        self.v_proj = nn.Linear(dim, dim)  # Value projection for vision and audio

    def forward(self, x_nlp, x_vision, x_audio):
        """
        Performs multi-modal attention.

        Args:
            x_nlp (torch.Tensor): NLP input features.
            x_vision (torch.Tensor): Vision input features.
            x_audio (torch.Tensor): Audio input features.

        Returns:
            torch.Tensor: The weighted sum of vision and audio features based on attention.
        """
        q = self.q_proj(x_nlp)  # Project NLP features into query space
        k_vision = self.k_proj(x_vision)  # Project vision features into key space
        v_vision = self.v_proj(x_vision)  # Project vision features into value space
        k_audio = self.k_proj(x_audio)  # Project audio features into key space
        v_audio = self.v_proj(x_audio)  # Project audio features into value space

        # Calculate attention weights for each modality separately
        attn_vision = F.softmax(torch.bmm(q.unsqueeze(1), k_vision.transpose(1, 2)).squeeze(1), dim=-1)
        attn_audio = F.softmax(torch.bmm(q.unsqueeze(1), k_audio.transpose(1, 2)).squeeze(1), dim=-1)

        # Perform weighted sum for each modality
        attended_vision = torch.bmm(attn_vision.unsqueeze(1), v_vision).squeeze(1)
        attended_audio = torch.bmm(attn_audio.unsqueeze(1), v_audio).squeeze(1)

        # Combine attended outputs (you can adjust the combination method)
        combined_output = attended_vision + attended_audio  # Simple summation

        return combined_output
