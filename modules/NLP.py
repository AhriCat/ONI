import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import time
import re
import logging
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Union, List, Dict, Optional, Tuple, Set
from tqdm import tqdm
import os
import json
import base64
from PIL import Image
import numpy as np
import io

hidden_dim = 896
num_heads = 24
efficient_attention = EfficientAttention(hidden_dim, num_heads)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)  # Add layer norm
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual  # Add residual connection

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("Dimension must be even")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
            
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i, j -> ij", t, self.inv_freq)

        cos_emb = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin_emb = torch.sin(freqs).repeat_interleave(2, dim=-1)

        cos_emb = cos_emb.unsqueeze(0).unsqueeze(0)
        sin_emb = sin_emb.unsqueeze(0).unsqueeze(0)

        if len(x.size()) == 3:
            x = x.unsqueeze(1)

        x_rotated = (x * cos_emb) + (self.rotate_half(x) * sin_emb)

        return x_rotated.squeeze(1)

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., :x.size(-1) // 2], x[..., x.size(-1) // 2:]
        return torch.cat((-x2, x1), dim=-1)
class MultiModalEmbedding(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int,
                 dropout: float,
                 vision_feature_dim: int = 1024,  # Example, adjust to your actual feature size
                 audio_feature_dim: int = 512,   # Example, adjust to your actual feature size
                 emotion_feature_dim: int = 64  # Example, adjust to your actual feature size
                ):
        super(MultiModalEmbedding, self).__init__()

        self.hidden_dim = hidden_dim

        # Modality-specific feature projection
        self.vision_projection = nn.Linear(vision_feature_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_feature_dim, hidden_dim)
        self.emotion_projection = nn.Linear(emotion_feature_dim, hidden_dim)
        
        # Shared embedding space
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        nn.init.xavier_normal_(self.embedding.weight)

        # Positional encoding for modality inputs
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attn = SparseFocusedGroupAttention(hidden_dim, 32, 8, .25)

    def forward(self, 
                vision_tensor: Optional[torch.Tensor] = None, 
                audio_tensor: Optional[torch.Tensor] = None, 
                emotion_tensor: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                meta_tokens: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Projects inputs into a shared embedding space

        Args:
            vision_tensor (Optional[torch.Tensor]): Tensor from vision module
            audio_tensor (Optional[torch.Tensor]): Tensor from audio module
            emotion_tensor (Optional[torch.Tensor]): Tensor from emotion module
            input_ids (Optional[torch.Tensor]): Input IDs from text input.
        Returns:
           torch.Tensor: Combined multi-modal embedding.
        """

        device = next(self.parameters()).device
        embedded_inputs = []
        
        if vision_tensor is not None:
            # Project vision features
            projected_vision = self.vision_projection(vision_tensor.float())
            embedded_inputs.append(projected_vision.unsqueeze(1))
        else:
             if input_ids is not None:
                batch_size, seq_len = input_ids.size()
             else:
                 batch_size = 1
                 seq_len = 1
             projected_vision = torch.zeros(batch_size, self.vision_projection.in_features, device=device).unsqueeze(1)
             embedded_inputs.append(projected_vision)


        if audio_tensor is not None:
            # Project audio features
            projected_audio = self.audio_projection(audio_tensor.float())
            embedded_inputs.append(projected_audio.unsqueeze(1))
        else:
            if input_ids is not None:
                batch_size, seq_len = input_ids.size()
            else:
                batch_size = 1
                seq_len = 1
            projected_audio = torch.zeros(batch_size, self.audio_projection.in_features, device=device).unsqueeze(1)
            embedded_inputs.append(projected_audio)


        if emotion_tensor is not None:
            # Project emotion features
            if isinstance(emotion_tensor, list):
                emotion_tensor = torch.tensor([item['score'] for item in emotion_tensor], dtype=torch.float32, device=device)
            
            if emotion_tensor.size() == torch.Size([0]):
                  emotion_tensor = torch.zeros(1, self.emotion_projection.in_features, device=device)
            elif emotion_tensor.dim() == 1:
                emotion_tensor = emotion_tensor.unsqueeze(0)
                
            # Check if the emotion tensor matches the expected size
            if emotion_tensor.size(1) != self.emotion_projection.in_features:
                
                if emotion_tensor.size(1) < self.emotion_projection.in_features:
                    padding = torch.zeros(emotion_tensor.size(0), self.emotion_projection.in_features - emotion_tensor.size(1), device=device)
                    emotion_tensor = torch.cat([emotion_tensor, padding], dim=1)
                else:
                    emotion_tensor = emotion_tensor[:, :self.emotion_projection.in_features]
                    
            projected_emotion = self.emotion_projection(emotion_tensor.float())
            embedded_inputs.append(projected_emotion.unsqueeze(1))
        else:
            if input_ids is not None:
                batch_size, seq_len = input_ids.size()
            else:
                batch_size = 1
                seq_len = 1
            projected_emotion = torch.zeros(batch_size, self.emotion_projection.in_features, device=device).unsqueeze(1)
            embedded_inputs.append(projected_emotion)
        
        if input_ids is not None:
            # Embed the input IDs from text
            x = self.embedding(input_ids.to(torch.long))
            if meta_tokens is not None:
                meta_embed = self.meta_token_embedding(meta_tokens)
                x = x + meta_embed
                x = self.attn(x)
            embedded_inputs.append(x)
        
        if not embedded_inputs:
             raise ValueError("At least one input modality must be provided")

        # Concatenate all embeddings
        combined = torch.cat(embedded_inputs, dim=1)

        # Apply positional encoding
        if input_ids is not None:
          combined = self.positional_encoding(combined)
        
        # Layer norm after positional embedding
        combined = self.layer_norm(combined)

        return combined
    
class OnlineLearner(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, lr: float = 0.01):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def forward(self, x):
        return self.fc(x)

    def update(self, input_embedding, target_embedding, reward):
        self.optimizer.zero_grad()
        output = self.forward(input_embedding)
        loss = F.mse_loss(output, target_embedding) * reward
        loss.backward()
        self.optimizer.step()

    
class ReformerAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super(ReformerAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(dropout)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, mask=None):
        batch_size, seq_length, dim = x.size()
        assert dim == self.dim, f"Input dimension {dim} does not match layer dimension {self.dim}"

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        if mask is not None:
            attn_mask = mask.unsqueeze(1).repeat(1, seq_length, 1)
        else:
            attn_mask = None

        attn_output, _ = self.multihead_attn(q, k, v, attn_mask=attn_mask)
        attn_output = self.dropout(attn_output)
        output = self.out_proj(attn_output)

        return output + x


class FatDiffuser(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        hidden_dim: int, 
        num_heads: int = 16, 
        timesteps: int = 30, 
        max_seq_len: int = 512, 
        embedding: nn.Embedding = None  # Option to pass custom embedding
    ):
        super(FatDiffuser, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.timesteps = timesteps
        self.max_seq_len = max_seq_len

        # Use provided embedding or initialize a new one
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
            nn.init.xavier_uniform_(self.embedding.weight)

        # Positional and time embeddings
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, hidden_dim))
        nn.init.xavier_uniform_(self.positional_encoding)
        self.time_embedding = nn.Parameter(torch.randn(timesteps, hidden_dim))
        nn.init.xavier_uniform_(self.time_embedding)

        # Diffusion layers
        self.diffusion_layers = nn.ModuleList([
            ReformerAttention(dim=self.hidden_dim, num_heads=self.num_heads)
            for _ in range(self.timesteps)
        ])

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        # Noise schedule with clipping for gradient stability
        self.register_buffer('alpha_schedule', torch.clamp(torch.linspace(0.1, 1.0, timesteps), min=0.1, max=0.9))

        # Early stopping parameters
        self.early_stop_threshold = 1e-4  # Stop if the change in logits is below this
        self.max_early_stop_checks = 5   # Maximum steps to check for early stopping

    def forward(self, input_ids):
        device = input_ids.device
        batch_size, seq_len = input_ids.size()

        # Embed input tokens
        h = self.embedding(input_ids.to(torch.long))

        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).to(device)
        h = h + pos_encoding

        # Precompute time encodings and noise
        time_encodings = self.time_embedding.unsqueeze(1).unsqueeze(1).to(device)
        alphas = self.alpha_schedule.view(self.timesteps, 1, 1, 1).to(device)
        noise = torch.randn(self.timesteps, batch_size, seq_len, self.hidden_dim, device=device)

        logits_list = []  # Keep track of logits for early stopping
        for t in range(self.timesteps):
            # Add time encoding
            h = h + time_encodings[t]

            # Apply attention and normalization
            attn_output = self.diffusion_layers[t](h)
            h = h + attn_output
            h = self.layer_norm(h)

            # Add noise scaled by alpha
            h = h + alphas[t] * noise[t]

            # Calculate logits for current timestep
            logits = self.output_projection(h)
            logits_list.append(logits)

            # Early stopping: Check if logits change significantly
            if t > 0 and self._check_early_stopping(logits_list, t):
                break

        return logits

class MetaCognitionModule(nn.Module):
    def __init__(self, hidden_dim):
        """
        MetaCognitionModule with contextual, nuanced conflict reasoning and interaction graphs.

        Args:
            hidden_dim (int): Dimensionality of the hidden input tensor.
        """
        super(MetaCognitionModule, self).__init__()
        
        # Reflection, confidence, and normalization layers
        self.self_reflection = nn.Linear(hidden_dim, hidden_dim)
        self.confidence_estimation = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dynamic storage for principles
        self.principles = nn.ParameterList()

        # Contextual projection layer for principles
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)

        # Alignment layer
        self.adaptive_alignment = nn.Linear(hidden_dim, hidden_dim)

    def add_principle(self, principle_vector):
        """
        Dynamically add a new principle to the module.

        Args:
            principle_vector (torch.Tensor): A tensor of shape (hidden_dim,).
        """
        if principle_vector.dim() != 1 or principle_vector.size(0) != self.self_reflection.in_features:
            raise ValueError("Principle vector must be of shape (hidden_dim,).")
        self.principles.append(nn.Parameter(principle_vector.clone(), requires_grad=True))

    def contextual_conflict_score(self, principle_a, principle_b, context):
        """
        Compute a nuanced conflict score between two principles in the given context.

        Args:
            principle_a (torch.Tensor): Tensor representing principle A.
            principle_b (torch.Tensor): Tensor representing principle B.
            context (torch.Tensor): The input context tensor.

        Returns:
            score (float): Conflict score, where higher values indicate stronger conflict.
        """
        # Project principles into the context space
        proj_a = self.context_projection(principle_a)
        proj_b = self.context_projection(principle_b)
        context_weight = F.normalize(context, p=2, dim=-1)  # Normalize context vector

        # Measure directional conflict weighted by context
        score = torch.dot(context_weight, proj_a - proj_b).abs()
        return score

    def detect_nuanced_conflicts(self, context, threshold=0.5):
        """
        Detect nuanced, context-aware conflicts between principles.

        Args:
            context (torch.Tensor): The input context tensor.
            threshold (float): Threshold for determining significant conflict.

        Returns:
            conflicts (list): List of conflicting principle index pairs and their scores.
        """
        conflicts = []
        num_principles = len(self.principles)
        if num_principles < 2:
            return conflicts

        # Compare all principle pairs
        for i in range(num_principles):
            for j in range(i + 1, num_principles):
                score = self.contextual_conflict_score(self.principles[i], self.principles[j], context)
                if score > threshold:
                    conflicts.append((i, j, score.item()))
        return conflicts

    def forward(self, x, conflict_threshold=0.5):
        """
        Forward pass with context-aware principle alignment and conflict detection.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, hidden_dim)
            conflict_threshold (float): Threshold for detecting nuanced conflicts.

        Returns:
            x (torch.Tensor): Updated input tensor after reflection.
            confidence (torch.Tensor): Confidence score for self-reflection.
            conflicts (list): List of nuanced conflicts detected.
        """
        # Self-reflection step
        reflection = torch.tanh(self.self_reflection(x))

        # Aggregate principles dynamically
        if len(self.principles) > 0:
            principles = torch.stack(self.principles)  # Shape: (num_principles, hidden_dim)
            principle_weights = torch.softmax(torch.matmul(x, principles.T), dim=-1)  # Attention scores
            
            # Weighted principle alignment
            principle_alignment = torch.matmul(principle_weights, principles)  # Shape: (batch_size, hidden_dim)
            adaptive_reflection = self.adaptive_alignment(reflection + principle_alignment)
        else:
            adaptive_reflection = self.adaptive_alignment(reflection)

        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_estimation(adaptive_reflection))

        # Residual connection and layer normalization
        output = self.layer_norm(x + adaptive_reflection)

        # Detect nuanced conflicts
        conflicts = self.detect_nuanced_conflicts(x.mean(dim=0), threshold=conflict_threshold)

        return output, confidence, conflicts


class ReasoningModule(nn.Module):
    def __init__(self, hidden_dim):
        super(ReasoningModule, self).__init__()
        self.deductive = nn.Linear(hidden_dim, hidden_dim)
        self.inductive = nn.Linear(hidden_dim, hidden_dim)
        self.abductive = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)  # Added dropout
        
    def forward(self, x):
        # Add proper residual connections and combination
        deductive = self.dropout(F.relu(self.deductive(x)))
        inductive = self.dropout(F.relu(self.inductive(x)))
        abductive = self.dropout(F.relu(self.abductive(x)))
        combined = (deductive + inductive + abductive) / 3  # Average the reasoning paths
        return self.layer_norm(x + combined)


class CompositionalLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # First sub-layer with residual connection
        residual = x
        x = self.norm1(x)
        mixed_concepts, _ = self.concept_mixing(x, x, x)
        x = residual + self.dropout(mixed_concepts)
        
        # Second sub-layer with residual connection
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout(x)
        return x
    
    def concept_mixing(self, query, key, value):
        mixed_concepts, attn_weights = self.multihead_attention(query, key, value)
        return mixed_concepts, attn_weights


class ChainOfThought(nn.Module):
    def __init__(self, input_dim, memory_size):
        super().__init__()
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.memory = nn.Parameter(torch.randn(memory_size, input_dim) / math.sqrt(input_dim))  # Proper initialization
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        self.current_size = 0
        self.pointer = 0

    def forward(self, x):
        x = x.float()
        batch_size = x.size(0)
        
        # Update memory with new information
        x_flat = x.view(-1, self.input_dim)
        num_new_items = x_flat.size(0)
        indices = torch.arange(num_new_items) % self.memory_size
        self.memory.data[indices] = x_flat.to(self.memory.dtype)
        
        # Use attention to access memory
        memory_batch = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        attended_memory, _ = self.attention(x, memory_batch, memory_batch)
        
        # Combine with input through residual connection
        output = self.norm(x + attended_memory)
        
        self.current_size = min(self.current_size + num_new_items, self.memory_size)
        self.pointer = (self.pointer + num_new_items) % self.memory_size
        
        return output

class MultiHopReasoning(nn.Module):
    def __init__(self, hidden_dim, num_hops=3):
        super(MultiHopReasoning, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_hops = num_hops
        
        # Create separate transformations for each hop
        self.hop_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_hops)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, external_sources):
        batch_size = x.size(0)
        current_state = x
        
        # Perform multiple hops of reasoning
        for i, hop_layer in enumerate(self.hop_layers):
            # Process external source if available
            if i < len(external_sources):
                source = external_sources[i]
                source = source.expand(batch_size, -1, -1) if len(source.size()) == 2 else source
                current_state = current_state + hop_layer(source)
            
            # Process current state
            current_state = current_state + hop_layer(current_state)
        
        return self.final_norm(current_state)
    
class ConceptualIntegrationLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(ConceptualIntegrationLayer, self).__init__()
        self.concept_mixing = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        mixed_concepts, _ = self.concept_mixing(x, x, x)
        return self.norm(x + mixed_concepts)

class CausalInferenceModule(nn.Module):
    def __init__(self, hidden_dim):
        super(CausalInferenceModule, self).__init__()
        # Structural Causal Model layers
        self.cause_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.effect_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.causal_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Intervention and counterfactual layers
        self.intervention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.counterfactual_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, intervention=None):
        # Encode cause-effect relationships
        causes = self.cause_encoder(x)
        effects = self.effect_encoder(x)
        
        # Learn causal attention
        causal_map, _ = self.causal_attention(causes, effects, effects)
        x = self.layer_norm1(x + self.dropout(causal_map))
        
        # Apply intervention if provided
        if intervention is not None:
            intervention_effect = self.intervention_layer(intervention)
            x = x + intervention_effect
        
        # Generate counterfactual
        counterfactual_input = torch.cat([x, causal_map], dim=-1)
        counterfactual = self.counterfactual_layer(counterfactual_input)
        
        output = self.layer_norm2(x + self.dropout(counterfactual))
        return output
    
def test_positional_encoding():
    pos_enc = PositionalEncoding(dim=512)
    x = torch.randn(32, 100, 512)
    output = pos_enc(x)
    assert output.shape == x.shape


class HybridProcessingBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ssm_state_dim: int, dropout: float):
        super().__init__()
        
        # Hymba block components
        self.ssm_head = SSMHead(hidden_dim, ssm_state_dim)
        self.attention_head = AttentionHead(hidden_dim, num_heads)
        
        # Gate mechanisms
        self.ssm_gate_norm = nn.LayerNorm(hidden_dim)
        self.attn_gate_norm = nn.LayerNorm(hidden_dim)
        
        # Mixing network
        self.mixer = MixingNetwork(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory_state=None):
        # SSM processing branch
        ssm_out = self.ssm_head(x)
        ssm_gate = self.ssm_gate_norm(ssm_out)
        
        # Attention processing branch
        attn_out = self.attention_head(x)
        attn_gate = self.attn_gate_norm(attn_out)
        
        # Combine branches with memory
        if memory_state is not None:
            combined = self.mixer(ssm_gate, attn_gate, memory_state)
        else:
            combined = self.mixer(ssm_gate, attn_gate)
            
        return self.dropout(combined)


class SSMHead(nn.Module):
    def __init__(self, hidden_dim: int, state_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(state_dim, state_dim))
        self.B = nn.Parameter(torch.randn(state_dim, hidden_dim))
        self.C = nn.Parameter(torch.randn(hidden_dim, state_dim))
        
        # Additional processing
        self.input_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.state_dim, device=x.device)
        outputs = []
        
        x = self.input_projection(x)
        
        for t in range(seq_len):
            # Update state
            h = torch.tanh(torch.matmul(h, self.A.T) + torch.matmul(x[:, t], self.B.T))
            # Generate output
            y = torch.matmul(h, self.C.T)
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        return self.output_projection(output)


class AttentionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # Reshape for attention
        x = x.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        attn_out, _ = self.attention(x, x, x)
        attn_out = attn_out.transpose(0, 1)  # [batch, seq_len, hidden_dim]
        return self.projection(attn_out)


class MixingNetwork(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fusion_layer = nn.Linear(hidden_dim * 3, hidden_dim)
        self.gate = nn.Linear(hidden_dim, 2)
        
    def forward(self, ssm_out, attn_out, memory_state=None):
        if memory_state is not None:
            # Concatenate all inputs
            combined = torch.cat([ssm_out, attn_out, memory_state], dim=-1)
        else:
            # Pad with zeros if no memory state
            batch_size, seq_len, _ = ssm_out.shape
            padding = torch.zeros_like(ssm_out)
            combined = torch.cat([ssm_out, attn_out, padding], dim=-1)
            
        # Fuse inputs
        fused = self.fusion_layer(combined)
        
        # Calculate mixing weights
        gates = F.softmax(self.gate(fused), dim=-1)
        
        # Mix SSM and attention outputs
        output = gates[:, :, 0:1] * ssm_out + gates[:, :, 1:2] * attn_out
        
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import re
from typing import Any, List, Dict, Optional, Union, Tuple  # Added 'Any' to imports
from torch.nn.utils.rnn import pad_sequence
from rouge import Rouge
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from rouge import Rouge


class PPOOptimizer:
    """
    Optimizes a model using the Proximal Policy Optimization (PPO) algorithm.

    Args:
        model (nn.Module): The model to optimize.
        lr (float): Learning rate for the optimizer. Defaults to 1e-5.
        gamma (float): Discount factor for rewards. Defaults to 0.99.
        eps_clip (float): Clipping epsilon for PPO. Defaults to 0.2.
        entropy_coef (float): Coefficient for entropy bonus. Defaults to 0.01.
    """
    def __init__(self, 
                 model: nn.Module, 
                 lr: float = 1e-5, 
                 gamma: float = 0.99, 
                 eps_clip: float = 0.2,
                 entropy_coef: float = 0.01):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.entropy_coef = entropy_coef

    def compute_loss(self, 
                     old_log_probs: torch.Tensor, 
                     new_log_probs: torch.Tensor,
                     rewards: torch.Tensor, 
                     advantages: torch.Tensor,
                     states: torch.Tensor,
                     returns: torch.Tensor) -> torch.Tensor:
        """
        Computes the PPO loss with policy, entropy, and value components.

        Args:
            old_log_probs (torch.Tensor): Log probabilities from the old policy.
            new_log_probs (torch.Tensor): Log probabilities from the new policy.
            rewards (torch.Tensor): Rewards obtained.
            advantages (torch.Tensor): Estimated advantages.
            states (torch.Tensor): Input states for value estimation.
            returns (torch.Tensor): Discounted returns.

        Returns:
            torch.Tensor: The total PPO loss.
        """
        # Policy loss with clipped surrogate objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Entropy bonus to encourage exploration
        entropy_loss = self.entropy_coef * (-new_log_probs).mean()

        # Value loss using predicted values vs actual returns
        value_loss = 0.5 * F.mse_loss(self.model.critic(states), returns)

        # Combine losses
        return policy_loss + value_loss - entropy_loss

    def optimize(self, 
                 old_log_probs: torch.Tensor, 
                 new_log_probs: torch.Tensor,
                 rewards: torch.Tensor, 
                 states: torch.Tensor,
                 values: torch.Tensor):
        """
        Performs a PPO optimization step.

        Args:
            old_log_probs (torch.Tensor): Log probabilities from the old policy.
            new_log_probs (torch.Tensor): Log probabilities from the new policy.
            rewards (torch.Tensor): Rewards obtained.
            states (torch.Tensor): Input states.
            values (torch.Tensor): Estimated state values.
        """
        # Compute discounted returns
        returns = torch.zeros_like(rewards, dtype=torch.float32)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return

        # Compute advantages
        advantages = returns - values

        # Compute and apply loss
        loss = self.compute_loss(
            old_log_probs, 
            new_log_probs, 
            rewards, 
            advantages, 
            states, 
            returns
        )
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class RewardFunctionWithROUGE:
    """
    Calculates ROUGE-based rewards for text generation with configurable ROUGE metrics.

    This reward function supports multiple ROUGE metrics and provides flexibility 
    in reward calculation for text generation tasks.
    """
    def __init__(self, 
                 rouge_types: Union[str, list] = ['rouge-l'], 
                 use_stemmer: bool = True):
        """
        Initialize the ROUGE reward function.

        Args:
            rouge_types (Union[str, list]): ROUGE metric types to use. 
                Defaults to ['rouge-l']. Can include 'rouge-1', 'rouge-2', 'rouge-l'.
            use_stemmer (bool): Whether to use stemming. Defaults to True.
        """
        # Ensure rouge_types is a list
        self.rouge_types = [rouge_types] if isinstance(rouge_types, str) else rouge_types
        
        # Initialize ROUGE with specified parameters
        self.rouge = Rouge(
            metrics=self.rouge_types,
            stemming=use_stemmer
        )

    def calculate_rouge_reward(self, 
                                generated_text: str, 
                                reference_text: str, 
                                metric_mode: str = 'f') -> float:
        """
        Calculates ROUGE scores as a reward.

        Args:
            generated_text (str): The text generated by the model.
            reference_text (str): The reference or ground truth text.
            metric_mode (str, optional): Which ROUGE score to use. 
                Can be 'f' (F1-score), 'p' (precision), or 'r' (recall). 
                Defaults to 'f'.

        Returns:
            float: Aggregated ROUGE score across specified ROUGE types.

        Raises:
            ValueError: If an invalid metric mode is provided.
        """
        # Validate metric mode
        if metric_mode not in ['f', 'p', 'r']:
            raise ValueError("metric_mode must be 'f', 'p', or 'r'")

        # Calculate ROUGE scores
        try:
            scores = self.rouge.get_scores(generated_text, reference_text)[0]
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return 0.0  # Default reward if calculation fails

        # Aggregate scores across specified ROUGE types
        total_score = 0.0
        for rouge_type in self.rouge_types:
            if rouge_type in scores:
                # Select the appropriate metric based on metric_mode
                if metric_mode == 'f':
                    total_score += scores[rouge_type]['f']
                elif metric_mode == 'p':
                    total_score += scores[rouge_type]['p']
                else:  # recall
                    total_score += scores[rouge_type]['r']

        # Return average if multiple ROUGE types, otherwise return the score
        return total_score / len(self.rouge_types)

    def calculate_detailed_rouge_reward(self, 
                                        generated_text: str, 
                                        reference_text: str) -> Dict[str, float]:
        """
        Provides a detailed breakdown of ROUGE scores.

        Args:
            generated_text (str): The text generated by the model.
            reference_text (str): The reference or ground truth text.

        Returns:
            Dict[str, float]: A dictionary of ROUGE scores for each metric.
        """
        try:
            scores = self.rouge.get_scores(generated_text, reference_text)[0]
            
            # Extract F1 scores for each ROUGE type
            return {
                rouge_type: scores[rouge_type]['f'] 
                for rouge_type in self.rouge_types 
                if rouge_type in scores
            }
        except Exception as e:
            print(f"Error calculating detailed ROUGE scores: {e}")
            return {}

    def normalize_reward(self, 
                         reward: float, 
                         min_val: float = 0.0, 
                         max_val: float = 1.0) -> float:
        """
        Optionally normalize the reward to a specific range.

        Args:
            reward (float): The original ROUGE-based reward.
            min_val (float, optional): Minimum of the target normalization range. Defaults to 0.0.
            max_val (float, optional): Maximum of the target normalization range. Defaults to 1.0.

        Returns:
            float: Normalized reward.
        """
        return min_val + (reward * (max_val - min_val))

class OptimizedNLPModule(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_heads: int, 
        num_layers: int, 
        dropout: float, 
        tokenizer,
        memory,
        memory_size: int = 8192, 
        output_dim: int = 1000000, 
        buffer_size: int = 100,
        emotions = None,
        num_repeats: int = 3,  # For Hymba blocks
        ssm_state_dim: int = 64,
        memory_decay_rate: float = 0.1
    ):
        super(OptimizedNLPModule, self).__init__()
        
        # Input validation
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided")
        if memory is None:
            raise ValueError("Memory must be provided")
        if not hasattr(memory, 'working_memory') or not hasattr(memory, 'ltm'):
            raise ValueError("Memory must have both 'working_memory' and 'ltm' attributes")

        # Configuration and device
        self.config = {
            "input_dim": input_dim, 
            "hidden_dim": hidden_dim,
            "num_heads": num_heads, 
            "num_layers": num_layers,
            "dropout": dropout, 
            "memory_size": memory_size,
            "output_dim": output_dim,
            "num_repeats": num_repeats,
            "ssm_state_dim": ssm_state_dim
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Core components
        self.tokenizer = tokenizer
        self.memory = memory
        self.working_memory = memory.working_memory
        self.ltm = memory.ltm
        
        # Initialize experience buffer
        self.experience_buffer: List[Dict[str, Union[str, List[str]]]] = []
        self.buffer_size = buffer_size
        self.attn = SparseFocusedGroupAttention(hidden_dim, 32, 8, .25)
        # Embedding and encoding layers
        self.embedding = MultiModalEmbedding(hidden_dim, hidden_dim, hidden_dim, 24, 0.1)
        self.meta_token_embedding = nn.Embedding(1000, hidden_dim)
        #nn.init.xavier_normal_(self.embedding.weights)
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        self.energy = 0 
        # Hybrid processing components
        self.hybrid_blocks = nn.ModuleList([
            HybridProcessingBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                ssm_state_dim=ssm_state_dim,
                dropout=dropout
            ) for _ in range(num_repeats)
        ])

        # Memory systems
        self.fading_memory = FadingMemorySystem(
            hidden_dim=hidden_dim,
            decay_rate=memory_decay_rate
        )
        self.snapshot_memory = SnapshotMemorySystem(
            hidden_dim=hidden_dim,
            memory_size=memory_size
        )

        # Imagination component
        self.imagination = FatDiffuser(
            vocab_size=self.tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            embedding=self.embedding
        )

        # Dynamic layers with residual connections
        self.dynamic_layers = nn.ModuleList([
            DynamicLayer(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])

        # Emotional energy model
        self.emotions = emoMod

        # Core processing modules
        self.reasoning = ReasoningModule(hidden_dim)
        self.composition = CompositionalLayer(hidden_dim)
        self.chain_of_thought = ChainOfThought(input_dim=hidden_dim, memory_size=memory_size)
        self.multi_hop = MultiHopReasoning(hidden_dim)
        self.metacognition = MetaCognitionModule(hidden_dim)
        self.conceptual_integration = ConceptualIntegrationLayer(hidden_dim)
        self.online_learner = OnlineLearner(hidden_dim, output_dim)
        self.causal_inference = CausalInferenceModule(hidden_dim)
        
        # Output processing
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        # Loss and regularization
        self.loss_fn = UpgradedDynamicAdvancedNLPLoss(ignore_index=self.tokenizer.pad_token_id)
        self.repetition_penalty = 0.2

        # Additional components
        self.finder = TextPatternFinder(self.tokenizer)
        self.hopfield_network = SparseHopfieldNetwork(hidden_dim) if 'SparseHopfieldNetwork' in globals() else None
        
        # Dropout and normalization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        # Projection for pattern integration
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_normal_(self.projection.weight)
        self.device = torch.device('cuda')

    def forward(
        self,
        input_text: Union[str, List[str], Dict[Any, str], torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        external_sources: Optional[List[torch.Tensor]] = None,
        max_length: Optional[int] = None,
        use_experience_buffer: bool = False,
        buffer_sampling_rate: float = 0.3,
        min_confidence_threshold: float = 0.7,
        meta_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, float, float]:

        device = next(self.parameters()).device
        
        # Process emotion input before others.
        user_feelings = self.emotions(input_text) if isinstance(self.emotions(input_text), dict) else None
        emotion_energy = user_feelings['emotional_state']['energy'] if user_feelings else 0
        emotion_tensor = user_feelings['output'] if user_feelings else None
        
        # Process input text, audio, visual, and emotion data
        input_texts, input_ids, attention_mask, keys, vision_tensor, audio_tensor, _ = self._process_raw_input(input_text, emotion_tensor=emotion_tensor)
        self._update_pattern_memory(input_texts)
        
        # Embed all modalities
        x = self.embedding(
            vision_tensor=vision_tensor,
            audio_tensor=audio_tensor,
            emotion_tensor=emotion_tensor,
            input_ids=torch.tensor(input_ids).to(device).to(torch.long) if input_ids else None, #convert id to correct type before sending to the embedding
            meta_tokens=meta_tokens,
            attention_mask=torch.tensor(attention_mask).to(device) if attention_mask else None
            )

        # Hybrid processing with memory
        memory_states = []
        for block in self.hybrid_blocks:
            # Update fading memory
            memory_state = self.fading_memory(x)
            memory_states.append(memory_state)
            
            # Process through hybrid block
            x = block(x, memory_state)
            
            # Take memory snapshot
            self.snapshot_memory.update(x)

        # Process through dynamic layers
        x, total_energy = self._process_dynamic_layers(x, total_energy, emotion_energy)
        
        # Apply dropout and attention mask
        x = self.dropout(x)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).to(x.dtype)

        # Imagination integration
        x = self._integrate_imagination(x, input_ids if input_ids else None)
        
        # Advanced reasoning pipeline
        x = self._advanced_reasoning_pipeline(
            x, external_sources, min_confidence_threshold
        )
        
        # Length handling
        if max_length is not None:
            x = x[:, :max_length, :]
            if input_ids is not None:
                input_ids = input_ids[:, :max_length]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_length]

        # Generate outputs
        outputs = self.output_layer(x)
        outputs = self.apply_repetition_penalty(outputs, input_ids if input_ids else None, self.repetition_penalty)

        return outputs, total_energy, user_feelings
    def _process_raw_input(self, input_text, emotion_tensor=None):
        """
        Process input based on type: Tensor, dict, list, string, or multimodal tensors for vision/audio/float tensors.
        Returns input_texts, input_ids, attention_mask, keys, vision_tensor, audio_tensor, and emotion_tensor.
        """
        device = next(self.parameters()).device
        vision_tensor, audio_tensor = None, None
        keys = None
        input_ids = []
        attention_mask = []
        embeddings = []
    
        if isinstance(input_text, torch.Tensor):
            if input_text.dtype in [torch.float32, torch.float64]:
            # Rasterize the tensors and return the tensors and the proper id and attention mask.
                raster = self.tokenizer._rasterize_float(input_text)
                input_ids = raster['input_ids']
                attention_mask = raster['attention_mask']
                embeddings = raster['embedding']
                input_texts = None
                keys = None
                return input_texts, input_ids, attention_mask, keys, vision_tensor, audio_tensor, emotion_tensor

            elif input_text.dtype == torch.long:  # Text tensor (input IDs)
                input_texts = [self.tokenizer.decode(ids.tolist()) for ids in input_text]
                input_ids = input_text
                attention_mask = (input_text != self.tokenizer.pad_token_id).long()
                return input_texts, input_ids, attention_mask, keys, vision_tensor, audio_tensor, emotion_tensor
            else:
                raise ValueError(f"Unsupported tensor data type: {input_text.dtype}")

        elif isinstance(input_text, dict):
            input_texts = list(input_text.values())
            keys = list(input_text.keys())
            # Rasterize all types of input and return.
            rasterized_inputs = [self.tokenizer(text) for text in input_texts]
            input_ids = [x['input_ids'] for x in rasterized_inputs]
            attention_mask = [x['attention_mask'] for x in rasterized_inputs]
            embeddings = [x['embedding'] for x in rasterized_inputs]

            return input_texts, input_ids, attention_mask, keys, vision_tensor, audio_tensor, emotion_tensor
        
        elif isinstance(input_text, str) or isinstance(input_text, list):
            if isinstance(input_text, str):
                input_texts = [input_text]
            else:
                input_texts = input_text
            # Rasterize all types of input and return.
            rasterized_inputs = [self.tokenizer(text) for text in input_texts]
            input_ids = [x['input_ids'] for x in rasterized_inputs]
            attention_mask = [x['attention_mask'] for x in rasterized_inputs]
            embeddings = [x['embedding'] for x in rasterized_inputs]
            return input_texts, input_ids, attention_mask, keys, vision_tensor, audio_tensor, emotion_tensor
            
        else:
            raise TypeError(f"Unsupported input type: {type(input_text)}. Expected Tensor, dict, str, or list.")
        
    def apply_repetition_penalty(self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        """
        Apply a repetition penalty to logits based on previously generated tokens.
        """
        for i in range(logits.size(0)):
            for token_id in set(input_ids[i].tolist()):
                logits[i, :, token_id] /= penalty
        return logits
    
    def _modulate_energy(self, x, emotion_energy, attention_weights=None, max_energy=100.0, min_energy=0.0):
        """
        Modulate energy based on input tensor x and emotion level.

        Args:
            x (torch.Tensor): The main tensor containing model activations or hidden states.
            emotion_energy (float): Energy derived from emotions.
            attention_weights (torch.Tensor, optional): Attention weights that may affect energy modulation.
            max_energy (float): The maximum cap for energy.
            min_energy (float): The minimum cap for energy.

        Returns:
            float: The modulated energy level.
        """
        # Baseline energy modulation factor based on tensor variance
        energy_modulation = torch.var(x).item()

        # Influence of emotions on energy
        energy_modulation += emotion_energy * 0.1  # Scale emotional influence

        # Optional: Modulate based on attention weights if provided
        if attention_weights is not None:
            avg_attention = attention_weights.mean().item()
            energy_modulation += avg_attention * 0.05  # Smaller influence from attention

        # Update the overall energy, capping it between min and max levels
        self.energy = max(min(self.energy + energy_modulation, max_energy), min_energy)

        return self.energy

        # Further processing
    def _update_pattern_memory(self, input_texts: List[str]):
        """Find patterns in input texts and update the pattern memory."""
        # Assuming `self.memory` is an instance of your `Memory` class
        # and `self.finder` is an instance of `TextPatternFinder`

        # Find patterns in the input texts
        corpus_patterns, _ = self.finder.find_patterns(input_texts, self.memory.ltm)

        # Update the pattern memory in the `TextPatternFinder`
        self.finder.corpus_patterns = corpus_patterns

    def _process_input(self, input_text, attention_mask, device):
        """Helper method to process different input types"""
        if isinstance(input_text, torch.Tensor):
            # Assuming input_text is input_ids
            input_ids = input_text.to(device)
            input_texts = [self.tokenizer.decode(ids) for ids in input_ids]
            keys = None
            if attention_mask is None:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
            else:
                attention_mask = attention_mask.to(device)
            return input_texts, input_ids, attention_mask, keys
        elif isinstance(input_text, dict):
            return self._process_dict_input(input_text, device)
        elif isinstance(input_text, (str, list)):
            return self._process_text_input(input_text, device)
        else:
            raise TypeError("Unsupported input type")

    def _process_dict_input(self, input_text, device):
        new_input_texts = []
        keys = []
        for key, value in input_text.items():
            new_input_texts.append(value)
            keys.append(key)
        encoding = self.tokenizer(
            new_input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        return new_input_texts, input_ids, attention_mask, keys

    def _process_text_input(self, input_text, device):
        if isinstance(input_text, str):
            input_texts = [input_text]
        else:
            input_texts = input_text
        encoding = self.tokenizer(
            input_texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        keys = None
        return input_texts, input_ids, attention_mask, keys

    def _process_tensor_input(self, input_text):
        self.forward(input_text)

    def _handle_nan_embeddings(self, x):
        """Handle NaN values in embeddings"""
        mask = torch.isnan(x)
        x[mask] = 0.0
        return self.layer_norm(x)  # Normalize after fixing NaNs

    def _integrate_patterns(self, x, device):
        """Integrate patterns from TextPatternFinder into the model's embeddings."""
        try:
            # Consolidate patterns from corpus and LTM
            unique_patterns = self.finder.consolidate_patterns()
            if not unique_patterns:
                return x

            # Process patterns to get embeddings
            pattern_vectors = self._process_patterns(unique_patterns, device, x.size(1))
            if pattern_vectors is not None:
                x = x + pattern_vectors  # Add pattern vectors to input embeddings

            return self.layer_norm(x)  # Normalize after pattern integration
        except Exception as e:
            print(f"Pattern integration failed: {str(e)}")
            return x

    def _process_patterns(self, patterns: Dict[str, List[int]], device, seq_len):
        """
        Convert patterns to vectors (embeddings) that can be added to the input embeddings.
        """
        if not patterns:
            return None

        # Get the most frequent patterns
        sorted_patterns = sorted(patterns.items(), key=lambda x: len(x[1]), reverse=True)
        top_patterns = [pattern for pattern, indices in sorted_patterns[:10]]  # Adjust the number as needed

        # Tokenize patterns
        encoded_patterns = self.tokenizer(
            top_patterns,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=seq_len
        ).to(device)
        pattern_ids = encoded_patterns['input_ids']  # Shape: [num_patterns, pattern_seq_len]

        # Get embeddings
        pattern_embeddings = self.embedding(pattern_ids)  # Shape: [num_patterns, pattern_seq_len, hidden_dim]

        # Aggregate embeddings (mean over sequence length)
        pattern_embeddings = torch.mean(pattern_embeddings, dim=1)  # Shape: [num_patterns, hidden_dim]

        # Aggregate embeddings across all patterns (mean or sum)
        aggregated_pattern_vector = torch.mean(pattern_embeddings, dim=0)  # Shape: [hidden_dim]

        # Expand to match input embeddings
        batch_size = x.size(0)
        pattern_vectors = aggregated_pattern_vector.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1)

        return pattern_vectors


    def _process_dynamic_layers(self, x, total_energy, emotion_energy):
        """Process input through dynamic layers"""
        for layer in self.dynamic_layers:
            x_prev = x  # Store previous state
            x, energy = layer(x)
            
            # Modulate energy
            energy = self._modulate_energy(energy, emotion_energy)
            total_energy += energy
            
            # Add residual connection
            x = x + x_prev
            x = self.layer_norm(x)  # Normalize after each layer
            
        return x, total_energy

    def _integrate_imagination(self, x, input_ids):
        """Integrate imagination module outputs"""
        imagination_logits = self.imagination(input_ids)
        imagination_probs = F.softmax(imagination_logits, dim=-1)
        imagination_embeds = self.embedding(imagination_probs.argmax(dim=-1))
        return self.layer_norm(x + imagination_embeds)

    def _advanced_reasoning_pipeline(self, x, external_sources, min_confidence_threshold):
        """Process input through advanced reasoning modules"""
        x = self.chain_of_thought(x)
        x, initial_confidence = self.metacognition(x)

        # Adaptive reasoning based on confidence
        if initial_confidence.mean().item() < min_confidence_threshold:
            reasoning_iterations = max(1, min(5, int((1 - initial_confidence.mean().item()) * 5)))
            for _ in range(reasoning_iterations):
                x = self.chain_of_thought(x)
                x = self.layer_norm(x)

        # Process through remaining modules
        module_sequence = [
            self.metacognition,
            self.conceptual_integration,
            self.causal_inference,
            self.reasoning,
            self.composition
        ]

        for module in module_sequence:
            if module == self.metacognition:
                x, _ = module(x)
            else:
                x = module(x)
            x = self.layer_norm(x)

        if external_sources:
            x = self.multi_hop(x, external_sources)
            x = self.layer_norm(x)

        return x

    def generate(
        self,
        input_text: Union[str, List[str], torch.Tensor],
        max_length: int = 512,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        **kwargs
    ) -> Union[str, List[str], torch.Tensor]:
        """Enhanced generation method"""
        self.eval()
        with torch.no_grad():
            # Get initial outputs
            outputs, _, _ = self.forward(
                input_text=input_text,
                max_length=max_length,
                **kwargs
            )
            
            # Apply temperature and filtering
            if isinstance(outputs, torch.Tensor):
                outputs = outputs / temperature
                outputs = self.top_k_top_p_filtering(
                    outputs, top_k=top_k, top_p=top_p
                )
            
            return outputs

    
    def top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int = 50, top_p: float = 0.95, filter_value: float = -float('Inf')) -> torch.Tensor:
        """
        Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            filter_value: value to set filtered logits to
        Returns:
            logits after filtering
        """
        top_k = min(top_k, logits.size(-1))  # Safety check

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value

        return logits

    def beam(
        self, 
        input_ids: torch.Tensor, 
        max_length: int, 
        initial_beam_width: int = 3, 
        pattern_boost: float = 1.5, 
        length_penalty: float = 0.7
    ) -> torch.Tensor:
        """
        Perform beam search to generate text.

        :param input_ids: Tensor of input token IDs, shape (batch_size, seq_len)
        :param max_length: Maximum length of the generated sequence.
        :param initial_beam_width: Number of beams to keep.
        :param pattern_boost: Boost factor for pattern-matched tokens.
        :param length_penalty: Penalty applied based on sequence length.
        :return: Tensor of the best generated token IDs.
        """
        beam = [(input_ids, 0)]  # Each element is a tuple (sequence, score)

        # Fetch patterns from the TextPatternFinder
        patterns = self.finder.consolidate_patterns()
        pattern_vectors = [self.finder._pattern_to_vector(p) for p in patterns.keys()]

        for _ in range(max_length):
            all_candidates = []
            
            for seq, score in beam:
                outputs, _, _ = self.forward(seq.unsqueeze(0))  # Add batch dimension
                next_token_logits = outputs[0, -1, :]  # Get logits for the last token
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)  # Log probabilities
                
                # Get top initial_beam_width token probabilities and their indices
                top_probs, top_indices = torch.topk(next_token_probs, initial_beam_width)
                
                for i in range(initial_beam_width):
                    candidate = (torch.cat([seq, top_indices[i].unsqueeze(0)]), score + top_probs[i].item())
                    all_candidates.append(candidate)
            
            # Sort candidates by score and select the top initial_beam_width ones
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beam = ordered[:initial_beam_width]

            # Adjust scores based on pattern matching using Hopfield network
            pattern_scores = {p: 0 for p in patterns}
            for i, (seq, score) in enumerate(beam):
                seq_str = self.tokenizer.decode(seq.squeeze(0).tolist())
                seq_vector = self.finder._pattern_to_vector(seq_str)

                # Use Hopfield network to check pattern similarities
                if self.hopfield_network:
                    for pattern, pattern_vector in zip(patterns.keys(), pattern_vectors):
                        similarity = self.hopfield_network.similarity(seq_vector, pattern_vector)
                        if similarity > 0.5:  # Threshold can be tuned
                            pattern_scores[pattern] += similarity * pattern_boost

            # Modify scores based on pattern matching
            for i in range(len(beam)):
                seq, score = beam[i]
                seq_str = self.tokenizer.decode(seq.squeeze(0).tolist())
                pattern_match_score = sum(pattern_scores.get(p, 0) for p in patterns if p in seq_str)
                adjusted_score = score + pattern_match_score - length_penalty * len(seq)
                beam[i] = (seq, adjusted_score)

            # Dynamically adjust beam width based on confidence in pattern matching
            top_pattern_score = max(pattern_scores.values())
            if top_pattern_score > 1.0:  # Threshold for strong pattern detection
                initial_beam_width = max(1, initial_beam_width - 1)  # Narrow the beam

        # Return the sequence with the highest score
        best_sequence = sorted(beam, key=lambda x: x[1], reverse=True)[0][0]
        return best_sequence

    def train_step(self, src, tgt, optimizer, criterion):
        optimizer.zero_grad()
        
        logits, energy, _ = self.forward(src)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_value = loss.item()  # Call item() only once
        energy_value = energy.item()
        
        # Update weights and adjust learning rates for each dynamic layer
        with torch.no_grad():
            performance_metric = 1.0 - loss_value
            for layer in self.dynamic_layers:
                layer.update_weights(loss_value)  # Ensure update_weights is defined
                layer.adjust_learning_rate(performance_metric)  # Ensure adjust_learning_rate is defined
        
        return loss_value, energy_value

    def beam_search_inference(
        self, 
        user_input: Union[str, List[str]], 
        max_length: int = 50, 
        initial_beam_width: int = 3, 
        pattern_boost: float = 1.5, 
        length_penalty: float = 0.7
    ) -> List[str]:
        """
        Perform beam search inference on the given user input.

        :param user_input: The input text prompt(s).
        :param max_length: Maximum length of the generated text.
        :param initial_beam_width: Initial beam width for beam search.
        :param pattern_boost: Boost factor for pattern-matched tokens.
        :param length_penalty: Penalty applied based on sequence length.
        :return: List of generated text sequences.
        """
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        # Handle single or multiple inputs
        if isinstance(user_input, str):
            user_input = [user_input]

        # Initialize list to store generated texts
        generated_texts = []

        for prompt in user_input:
            # Tokenize the prompt
            encoding = self.tokenizer(
                texts=[prompt],
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            input_ids = encoding['input_ids'].to(device)

            # Perform beam search
            best_sequence = self.beam(input_ids, max_length, initial_beam_width, pattern_boost, length_penalty)

            # Decode the generated sequence
            generated_text = self.tokenizer.decode(best_sequence.tolist())
            generated_texts.append(generated_text)

        return generated_texts

    def add_to_experience_buffer(self, input_text: str, decoded_output: str):
        """
        Add the input-output pair to the experience buffer, maintaining a FIFO order.
        """
        if len(self.experience_buffer) >= self.buffer_size:
            self.experience_buffer.pop(0)  # Remove the oldest experience
        self.experience_buffer.append({'input_text': input_text, 'output_text': decoded_output})

    def sample_from_experience_buffer(self, batch_size: int) -> List[Dict[str, str]]:
        """
        Sample a batch of experiences from the experience buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            List[Dict[str, str]]: A list of sampled experiences, or an empty list if the buffer is empty.
        """
        if not self.experience_buffer:
            return []
        sampled = random.sample(self.experience_buffer, min(batch_size, len(self.experience_buffer)))
        return sampled
    
    def reason_step(self, current_prompt, feedback=None):
        """
        Generates the next reasoning step using the transformer model.
        
        Args:
            current_prompt (str): The current state of reasoning.
            feedback (str): Feedback to guide the generation.
        
        Returns:
            next_step (str): The generated next step.
        """
        # Combine prompt and feedback
        input_prompt = current_prompt if not feedback else f"{current_prompt}\nFeedback: {feedback}"
        
        # Use the transformer to generate the next step
        generated_output = self.generate(input_prompt)
        return generated_output
    
    def train_step(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """
        Perform a single training step.

        :param src: Source input tensor.
        :param tgt: Target output tensor.
        :param optimizer: Optimizer.
        :param criterion: Loss function.
        :return: Tuple of loss value and energy value.
        """
        optimizer.zero_grad()
        
        logits, energy, _ = self.forward(src)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_value = loss.item()
        energy_value = energy.item()
        
        # Update weights and adjust learning rates for each dynamic layer
        with torch.no_grad():
            performance_metric = 1.0 - loss_value
            for layer in self.dynamic_layers:
                if hasattr(layer, 'update_weights'):
                    layer.update_weights(loss_value)  # Ensure update_weights is defined in DynamicLayer
                if hasattr(layer, 'adjust_learning_rate'):
                    layer.adjust_learning_rate(performance_metric)  # Ensure adjust_learning_rate is defined
    
        return loss_value, energy_value

    def beam_search(
        self, 
        input_ids: torch.Tensor, 
        max_length: int = 50, 
        beam_width: int = 3, 
        repetition_penalty: float = 1.0
    ) -> str:
        """
        Perform beam search generation.

        :param input_ids: Tensor of input token IDs, shape (1, seq_len)
        :param max_length: Maximum length of the generated sequence.
        :param beam_width: Number of beams to keep.
        :param repetition_penalty: Repetition penalty factor.
        :return: Generated text.
        """

        beam = [(input_ids, 0)]  # Each element is a tuple (sequence, score)

        for _ in range(max_length):
            all_candidates = []

            for seq, score in beam:
                outputs, _, _ = self.forward(seq.unsqueeze(0))
                next_token_logits = outputs[0, -1, :]  # Shape: (vocab_size,)
                next_token_logits = self.apply_repetition_penalty(next_token_logits, seq.unsqueeze(0), penalty=repetition_penalty)
                next_token_probs = F.log_softmax(next_token_logits, dim=-1)
                top_probs, top_indices = torch.topk(next_token_probs, beam_width)

                for i in range(beam_width):
                    candidate_seq = torch.cat([seq, top_indices[i].unsqueeze(0)])
                    candidate_score = score + top_probs[i].item()
                    all_candidates.append((candidate_seq, candidate_score))

            # Select top beam_width candidates
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            beam = ordered[:beam_width]

            # Check for stop tokens
            for seq, score in beam:
                if seq[-1].item() == self.tokenizer.eos_token_id:
                    return self.tokenizer.decode(seq.tolist())

        # If no stop token was found, return the highest scoring sequence
        best_seq = beam[0][0]
        return self.tokenizer.decode(best_seq.tolist())

    def generate_ongoing_thoughts(self):
        while True:
            thought, avg_confidence = self.create_thought()
            response = self.process_thought(thought)
            print(f"Thought: {thought}")
            print(f"Response: {response}")
            print("---")
            time.sleep(30)  # Generate a new thought every 30 seconds

    def identify_tasks(self, text: str) -> List[str]:
        """
        Identify task-related keywords in the input text.

        :param text: Input text.
        :return: List of detected task keywords.
        """
        task_indicators = [
            r'/find',
            r'help me find\b',
            r'/get',
            r'retrieve information on\b',
            r'book a reservation for\b',
            r'/screen',
            r'convert this to\b',
            r'/exchange',
            r'make a reservation for\b',
            r'/reserve',
            r'schedule a time for\b',
            r'plan the next steps for\b',
            r'/organize',
            r'create a new\b',
            r'purchase\b',
            r'lookup information on\b',
            r'/calculate',
            r'remind me to\b',
            r'send an email to\b',
            r'/email',
            r'/call',
            r'notify me when\b',
            r'/alert',
            r'check for updates on\b',
            r'/pay',
            r'set up\b',
            r'design a plan for\b',
            r'write a document about\b',
            r'build a structure for\b',
            r'/install',
            r'/fix',
            r'/update',
            r'/download',
            r'/upgrade',
            r'/debug',
            r'configure the settings for\b',
            r'deploy to environment\b',
            r'analyze the results of\b',
            r'track progress on\b',
            r'/test',
            r'launch project\b',
            r'monitor the system for\b',
            r'submit a report on\b',
            r'review the results of\b',
            r'/order',
            r'cancel the\b',
            r'return item to\b',
            r'inquire about\b',
            r'/register',
            r'arrange items in\b',
            r'collect data on\b',
            r'conduct research on\b',
            r'open file\b',
            r'exit application\b',
            r'/help',
            r'solve problem with\b',
            r'view details of\b',
            r'summarize information on\b',
            r'elaborate on details of\b',
            r'compare items\b',
            r'contrast results between\b',
            r'/sort',
            r'translate document to\b',
            r'/animate',
            r'select from options\b',

            # Added Commands
            r'/draw',
            r'create a sketch of\b',
            r'draw a picture of\b',
            r'/design',
            r'/illustrate',
            r'compose a piece on\b',
            r'/model',
            r'/photograph',
            r'/think',
            r'provide reasoning for\b',
            r'consider implications of\b',
            r'/reason',
            r'logically analyze\b',
            r'develop a theory on\b',

            # Music and Composition Commands
            r'make a song about\b',
            r'compose music for\b',
            r'/record',
            r'arrange sounds in\b',
            r'produce a track for\b',
            r'/mix',
            r'master audio of\b',
            r'/play',

            # Trading and Finance Commands
            r'/trade',
            r'/buy',
            r'sell assets in\b',
            r'/invest',
            r'execute trade on\b',

            # Cybersecurity and Monitoring
            r'/monitor network',
            r'test for vulnerabilities in\b',
            r'scan for security threats\b',
            r'detect potential breaches in\b',

            # Workforce and Customer Service Commands
            r'assist customer with\b',
            r'process payment for\b',
            r'/serve',
            r'handle luggage for\b',
            r'guide patron to\b',

            # Math and Graphing Functions
            r'add values in\b',
            r'subtract values from\b',
            r'multiply values in\b',
            r'/divide',
            r'plot a graph of\b',
            r'/compute',
            r'solve equation for\b',
            r'calculate the value of\b',

            # Visualization and Conceptualization
            r'visualize concept of\b',
            r'/wireframe',
            r'generate a model of\b',
            r'conceptualize the idea of\b',

            # Additional AGI Tasks
            r'/compose',
            r'write a song about\b'
            r'/learn',
            r'acquire knowledge about\b',
            r'/adapt',
            r'modify approach based on\b',
            r'/plan',
            r'develop a strategy for\b',
            r'/simulate',
            r'run a simulation on\b',
            r'/predict',
            r'forecast outcomes of\b',
            r'/assist',
            r'provide assistance with\b',
            r'/recommend',
            r'suggest options for\b',
            r'/summarize',
            r'give a summary of\b',
            r'/classify',
            r'group items based on\b',
            r'/optimize',
            r'find the best solution for\b',
            r'/navigate',
            r'find a route to\b',
            r'/schedule',
            r'add to calendar\b',
            r'/remember',
            r'set a reminder for\b',
            r'/explore',
            r'investigate possibilities in\b',
            r'/communicate',
            r'send a message to\b',
            r'/control',
            r'manage settings for\b',
            r'/execute',
            r'run command\b',
            r'/evaluate',
            r'assess performance of\b',
            r'/decide',
            r'choose the best option for\b',
            r'/understand',
            r'comprehend the meaning of\b',
            r'/interpret',
            r'explain the significance of\b',
            r'/question',
            r'ask about\b',
            r'/answer',
            r'respond to inquiry about\b',
            r'/measure',
            r'determine the size of\b',
            r'/connect',
            r'establish connection with\b',
            r'/store',
            r'save information on\b',
            r'/retrieve',
            r'access stored data on\b',
            r'/verify',
            r'confirm the accuracy of\b',
            r'/maintain',
            r'keep system running for\b',
            r'/secure',
            r'protect information about\b',
            r'/emulate',
            r'imitate behavior of\b',
            r'/coordinate',
            r'align tasks related to\b',
            r'/negotiate',
            r'find agreement on\b',
            r'/collaborate',
            r'work together on\b',
            r'/delegate',
            r'assign task regarding\b',
            r'/reflect',
            r'think deeply about\b',
            r'/discover',
            r'find new information about\b',
            r'/invent',
            r'create a new method for\b',
            r'/build',
            r'construct a system for\b',
            r'/analyze',
            r'examine the components of\b',
            r'/validate',
            r'ensure correctness of\b',
            r'/research',
            r'study in-depth about\b',
            r'/document',
            r'create documentation for\b',
            r'/train',
            r'learn from data on\b',
            r'/deploy',
            r'implement solution for\b',
            r'/adjust',
            r'modify parameters of\b',
            r'/observe',
            r'watch the behavior of\b',
            r'/perceive',
            r'become aware of\b',
            r'/imagine',
            r'visualize scenario of\b',
            r'/plan',
            r'layout strategy for\b',
            r'/react',
            r'respond to changes in\b',
            r'/integrate',
            r'combine elements of\b',
            r'/synthesize',
            r'merge ideas from\b',
            r'/transform',
            r'change the form of\b',
            r'/encode',
            r'convert data to\b',
            r'/decode',
            r'interpret data from\b',
            r'/recognize',
            r'identify patterns in\b',
            r'/classify',
            r'categorize aspects of\b',
            r'/filter',
            r'select items matching\b',
            r'/search',
            r'look for information on\b',
            r'/compress',
            r'reduce size of\b',
            r'/decompress',
            r'restore data from\b',
            r'/map',
            r'create a map of\b',
            r'/drive',
            r'control movement towards\b',
            r'/terminate',
            r'stop process on\b',
            r'/initialize',
            r'start process for\b',
            r'/configure',
            r'set up parameters for\b',
            r'/backup',
            r'create a backup of\b',
            r'/restore',
            r'retrieve backup of\b',
            r'/log',
            r'record events of\b',
            r'/audit',
            r'check compliance of\b',
            r'/report',
            r'provide status on\b',
            r'/broadcast',
            r'disseminate information on\b',
            r'/subscribe',
            r'sign up for updates on\b',
            r'/unsubscribe',
            r'opt out of updates on\b',
            r'/feedback',
            r'provide thoughts on\b',
            r'/review',
            r'assess quality of\b',
            r'/approve',
            r'give permission for\b',
            r'/deny',
            r'reject request for\b',
            r'/authorize',
            r'grant access to\b',
            r'/authenticate',
            r'verify identity for\b',
            r'/archive',
            r'store old data on\b',
            r'/share',
            r'distribute information on\b',
            r'/lead',
            r'guide efforts in\b',
            r'/follow',
            r'adhere to instructions on\b',
            r'/coordinate',
            r'align actions in\b',
            r'/update',
            r'bring up to date\b',
            r'/remember',
            r'store in memory\b',
            r'/forget',
            r'remove from memory\b',
            r'/explain',
            r'clarify the concept of\b',
            r'/provide',
            r'offer information on\b',
            r'/optimize',
            r'improve efficiency of\b',
            r'/simulate',
            r'model the scenario of\b',
            r'/decide',
            r'come to a conclusion on\b',
            r'/predict',
            r'estimate future of\b',
            r'/explore',
            r'look into possibilities of\b',
            r'/communicate',
            r'express ideas about\b',
            r'/innovate',
            r'introduce new ideas for\b',
            r'/feel',
            r'express sentiment about\b',
            r'/dream',
            r'conceptualize possibilities of\b',
            r'/inspire',
            r'motivate actions towards\b',
            r'/assist',
            r'support activities in\b',
            r'/transform',
            r'change the form of\b',
            r'/model',
            r'create a representation of\b',
            r'/simulate',
            r'imitate processes of\b',
            r'/recognize',
            r'identify patterns in\b',
            r'/cluster',
            r'group similar items in\b',
            r'/rank',
            r'order items based on\b',
            r'/search',
            r'look for information on\b',
            r'/retrieve',
            r'get data related to\b',
            r'/store',
            r'save data on\b',
            r'/encrypt',
            r'secure data on\b',
            r'/decrypt',
            r'access secured data on\b',
            r'/drive',
            r'control movement towards\b',
            r'/control',
            r'manage systems for\b',
            r'/execute',
            r'run process on\b',
            r'/terminate',
            r'stop process on\b',
            r'/configure',
            r'set up parameters for\b',
            r'/maintain',
            r'ensure proper function of\b',
            r'/upgrade',
            r'improve version of\b',
            r'/downgrade',
            r'revert to previous version of\b',
            r'/schedule',
            r'plan timing for\b',
            r'/log',
            r'record events of\b',
            r'/audit',
            r'check compliance of\b',
            r'/report',
            r'provide status on\b',
            r'/notify',
            r'send updates on\b',
            r'/broadcast',
            r'disseminate information on\b',
            r'/subscribe',
            r'sign up for updates on\b',
            r'/feedback',
            r'provide thoughts on\b',
            r'/review',
            r'assess quality of\b',
            r'/approve',
            r'give permission for\b',
            r'/deny',
            r'reject request for\b',
            r'/authorize',
            r'grant access to\b',
            r'/authenticate',
            r'verify identity for\b',
            r'/archive',
            r'store old data on\b',
            r'/share',
            r'distribute information on\b',
            r'/collaborate',
            r'work jointly on\b',
        ]
            # Combine the task indicators into a single regex pattern
        task_pattern = '|'.join(task_indicators)
            
            # Find all matches in the input text
        tasks = re.findall(task_pattern, text, re.IGNORECASE)
            
        return tasks

    def learn_from_interaction(self, input_text: str, output_text: str, reward: float):
        """
        Learn from each interaction to improve the model's performance.
        """
        # Store the experience in the buffer
        self.experience_buffer.append((input_text, output_text, reward))

        # Use online learning to update the model
        input_ids = self.tokenizer.encode(input_text)
        output_ids = self.tokenizer.encode(output_text)
        
        with torch.no_grad():
            input_embedding = self.embedding(torch.tensor(input_ids)).unsqueeze(0)
            output_embedding = self.embedding(torch.tensor(output_ids)).unsqueeze(0)

        self.online_learner.update(input_embedding, output_embedding, reward)

        # Update the adaptive layer
        self.adaptive_layer(input_embedding, reward)

        # Periodically update the main model using the experience buffer
        if len(self.experience_buffer) >= self.buffer_size:
            self.update_from_buffer()

    def update_from_buffer(self):
        """
        Update the main model using the experience buffer.
        """
        if not self.experience_buffer:
            return

        # Sample a batch from the experience buffer
        batch = random.sample(self.experience_buffer, min(64, len(self.experience_buffer)))

        # Prepare the data
        inputs, outputs, rewards = zip(*batch)
        input_ids = [self.tokenizer.encode(text) for text in inputs]
        output_ids = [self.tokenizer.encode(text) for text in outputs]

        # Pad sequences
        input_ids_padded = pad_sequence([torch.tensor(ids) for ids in input_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        output_ids_padded = pad_sequence([torch.tensor(ids) for ids in output_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        # Define optimizer if not already defined (ensure optimizer is part of the module or passed externally)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

        # Forward pass
        logits, energy, _ = self.forward(input_ids_padded, attention_mask=(input_ids_padded != self.tokenizer.pad_token_id).long())

        # Compute loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), output_ids_padded.view(-1), ignore_index=self.tokenizer.pad_token_id)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Optionally, remove experiences after training to prevent repeated training on the same data
        # self.experience_buffer = []

    def save_model(self, path: str):
        """
        Save the model state including continuous learning components.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'online_learner_state': self.online_learner.state_dict(),
            'adaptive_layer_state': self.adaptive_layer.state_dict(),
            'experience_buffer': self.experience_buffer,
        }, path)

    def load_model(self, path: str):
        """
        Load the model state including continuous learning components.
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.online_learner.load_state_dict(checkpoint['online_learner_state'])
        self.adaptive_layer.load_state_dict(checkpoint['adaptive_layer_state'])
        self.experience_buffer = checkpoint['experience_buffer']

    # Additional methods like create_thought, process_thought, generate_ongoing_thoughts, etc., should be reintroduced here as needed.
    # Ensure that all referenced methods and modules (e.g., FatDiffuser, ReversibleLayer) are defined and correctly integrated.


input_dim = 1000000 # Ensure input_dim matches the tokenizer's vocabulary size
hidden_dim = 896
nhead = 16
num_layers = 12
output_dim = 1000000
dropout = 0.1
memory_size = 8192

