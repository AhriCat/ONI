from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torchvision.models import resnet50
import librosa
import librosa.display
import numpy as np
import timm
from torchvision import transforms
import torch
from diffusers import DiffusionPipeline
import matplotlib.pyplot as plt
from transformers import pipeline
from collections import deque
import logging 
import shutil
from datetime import datetime
from pathlib import Path
import pyautogui
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import logging
import subprocess

device = torch.device('cuda')

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device = 'cuda')

class SparseFocusedGroupAttention(nn.Module):
    def __init__(self, hidden_dim, num_query_heads=64, num_kv_groups=16, sparsity_ratio=0.25):
        super().__init__()
        assert hidden_dim % num_query_heads == 0, "hidden_dim must be divisible by num_query_heads"
        
        self.hidden_dim = hidden_dim
        self.num_query_heads = num_query_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = hidden_dim // num_query_heads
        self.sparsity_ratio = sparsity_ratio
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, (hidden_dim // num_query_heads) * num_kv_groups)
        self.value_proj = nn.Linear(hidden_dim, (hidden_dim // num_query_heads) * num_kv_groups)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scaling = self.head_dim ** -0.5

    def create_sparse_mask(self, seq_len, device):
        # Create deterministic sparse mask instead of random
        mask = torch.ones(seq_len, seq_len, device=device)
        
        # Create local attention window
        window_size = max(1, int(seq_len * 0.1))
        for i in range(seq_len):
            start = max(0, i - window_size)
            end = min(seq_len, i + window_size + 1)
            mask[i, start:end] = 1.0
            
        # Add strided attention
        stride = max(1, seq_len // 8)
        for i in range(seq_len):
            mask[i, ::stride] = 1.0
            
        # Apply sparsity
        mask = mask * (torch.rand(seq_len, seq_len, device=device) < (1 - self.sparsity_ratio))
        return mask.bool()
        
    def forward(self, x):
        # Handle input dimensions
        orig_dim = x.dim()
        if orig_dim == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif orig_dim == 2:
            x = x.unsqueeze(1)
            
        batch_size, seq_len, _ = x.size()
        
        # Projections
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)
        
        # Reshape
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_groups, self.head_dim)
        
        # Transpose
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat k/v for each query head
        k = k.repeat_interleave(self.num_query_heads // self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_query_heads // self.num_kv_groups, dim=1)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        
        # Create and apply sparse mask
        sparse_mask = self.create_sparse_mask(seq_len, x.device)
        sparse_mask = sparse_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        attn_scores = attn_scores.masked_fill(~sparse_mask, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.hidden_dim)
        out = self.output_proj(out)
        
        # Restore original dimensions
        if orig_dim == 1:
            out = out.squeeze(0).squeeze(0)
        elif orig_dim == 2:
            out = out.squeeze(1)
            
        return out

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
    
class VisionTransformer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        self.fc = nn.Linear(1000, 1536)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)
class ResNetAudioAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 1536)

    def forward(self, x):
        x = self.model(x)
        return self.fc(x)

class PersistentMemory:
    def __init__(self, memory_file="memory.pkl", archive_dir="memory_archives", 
                 max_memory_age_days=30, quality_threshold=0.7, similarity_threshold=0.85, 
                 compression_ratio=0.5, max_compression_rounds=3):
        self.memory_file = memory_file
        self.archive_dir = Path(archive_dir)
        self.max_memory_age_days = max_memory_age_days
        self.memory = {}
        self.memory_timestamps = {}  # Track when memories were added
        self.nn_model = NearestNeighbors(n_neighbors=1, metric="cosine")
        self.embeddings = None
        self.logger = logging.getLogger(__name__)
        self.quality_threshold = quality_threshold
        self.similarity_threshold = similarity_threshold
        self.compression_ratio = compression_ratio
        self.max_compression_rounds = max_compression_rounds

        # Create archive directory if it doesn't exist
        self.archive_dir.mkdir(parents=True, exist_ok=True)

        if os.path.exists(self.memory_file):
            self._load_memory()

    def _load_memory(self):
        """Load memory and timestamps from disk."""
        try:
            with open(self.memory_file, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    # Old format - just memories
                    self.memory = data
                    # Initialize timestamps for old data
                    self.memory_timestamps = {k: datetime.now() for k in self.memory.keys()}
                else:
                    # New format - memories and timestamps
                    self.memory, self.memory_timestamps = data
                self._build_embedding_index()
        except Exception as e:
            self.logger.error(f"Error loading memory: {str(e)}")
            self.memory = {}
            self.memory_timestamps = {}

    def _save_memory(self):
        """Save memory and timestamps to disk."""
        try:
            with open(self.memory_file, "wb") as f:
                # Move to CPU for pickle
                cpu_memory = {k: v.cpu() for k, v in self.memory.items()}
                pickle.dump((cpu_memory, self.memory_timestamps), f)
                
            # Restore CUDA placement
            self.memory = {k: v.to(device) for k, v in self.memory.items()}
        except Exception as e:
            self.logger.error(f"Error saving memory: {str(e)}")

    def clear_memory(self, archive=True):
        """Clear all memory, optionally archiving it first."""
        try:
            if archive:
                self.archive_memory()
            
            # Clear memory and timestamps
            self.memory = {}
            self.memory_timestamps = {}
            self.embeddings = None
            
            # Remove the memory file
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
                
            self.logger.info("Memory cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing memory: {str(e)}")

    def archive_memory(self, archive_name=None):
        """Archive current memory to a timestamped file."""
        try:
            if not self.memory:
                return

            # Generate archive filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = archive_name or f"memory_archive_{timestamp}.pkl"
            archive_path = self.archive_dir / archive_name

            # Save current memory to archive
            with open(archive_path, "wb") as f:
                cpu_memory = {k: v.cpu() for k, v in self.memory.items()}
                pickle.dump((cpu_memory, self.memory_timestamps), f)

            self.logger.info(f"Memory archived to {archive_path}")
            return archive_path
        except Exception as e:
            self.logger.error(f"Error archiving memory: {str(e)}")
            return None

    def restore_from_archive(self, archive_name=None):
        """Restore memory from a specific archive or the most recent one."""
        try:
            if archive_name:
                archive_path = self.archive_dir / archive_name
            else:
                # Find most recent archive
                archives = list(self.archive_dir.glob("memory_archive_*.pkl"))
                if not archives:
                    self.logger.warning("No archives found")
                    return False
                archive_path = max(archives, key=os.path.getctime)

            # Load archive
            with open(archive_path, "rb") as f:
                data = pickle.load(f)
                if isinstance(data, dict):
                    self.memory = data
                    self.memory_timestamps = {k: datetime.now() for k in self.memory.keys()}
                else:
                    self.memory, self.memory_timestamps = data

            # Move to CUDA
            self.memory = {k: v.to(device) for k, v in self.memory.items()}
            self._build_embedding_index()
            self._save_memory()  # Save restored memory as current

            self.logger.info(f"Memory restored from {archive_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error restoring from archive: {str(e)}")
            return False

    def prune_old_memories(self):
        """Remove memories older than max_memory_age_days."""
        try:
            current_time = datetime.now()
            keys_to_remove = []
            
            for text, timestamp in self.memory_timestamps.items():
                age = (current_time - timestamp).days
                if age > self.max_memory_age_days:
                    keys_to_remove.append(text)
            
            if keys_to_remove:
                # Archive old memories before removing
                old_memories = {k: self.memory[k] for k in keys_to_remove}
                old_timestamps = {k: self.memory_timestamps[k] for k in keys_to_remove}
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_path = self.archive_dir / f"pruned_memories_{timestamp}.pkl"
                
                with open(archive_path, "wb") as f:
                    cpu_memories = {k: v.cpu() for k, v in old_memories.items()}
                    pickle.dump((cpu_memories, old_timestamps), f)
                
                # Remove old memories
                for key in keys_to_remove:
                    del self.memory[key]
                    del self.memory_timestamps[key]
                
                self._build_embedding_index()
                self._save_memory()
                
                self.logger.info(f"Pruned {len(keys_to_remove)} old memories")
                return len(keys_to_remove)
        except Exception as e:
            self.logger.error(f"Error pruning old memories: {str(e)}")
            return 0

    def get_memory_stats(self):
        """Get statistics about current memory usage."""
        try:
            current_time = datetime.now()
            stats = {
                "total_memories": len(self.memory),
                "oldest_memory": None,
                "newest_memory": None,
                "average_age_days": 0,
                "memory_size_mb": 0,
                "archive_count": len(list(self.archive_dir.glob("*.pkl")))
            }
            
            if self.memory_timestamps:
                ages = [(current_time - timestamp).days for timestamp in self.memory_timestamps.values()]
                stats["oldest_memory"] = max(ages)
                stats["newest_memory"] = min(ages)
                stats["average_age_days"] = sum(ages) / len(ages)
            
            # Estimate memory size
            if self.memory:
                sample_size = next(iter(self.memory.values())).element_size() * \
                             next(iter(self.memory.values())).nelement()
                stats["memory_size_mb"] = (sample_size * len(self.memory)) / (1024 * 1024)
            
            return stats
        except Exception as e:
            self.logger.error(f"Error getting memory stats: {str(e)}")
            return {}

    def add_to_memory(self, embedding, text):
        try:
            embedding = self._validate_tensor(embedding)
            if embedding is None or not embedding.is_cuda:
                self.logger.error("Failed to validate tensor for memory storage")
                return

            # Quality and novelty checks
            if not self._assess_quality(embedding, text):
                self.logger.info("Content failed quality check")
                return

            if not self._check_novelty(embedding):
                self.logger.info("Content not sufficiently novel")
                return

            # Add to memory with timestamp
            self.memory[text] = embedding
            self.memory_timestamps[text] = datetime.now()
            self._build_embedding_index()
            
            # Trigger compression if memory is getting large
            if len(self.memory) > 100:
                self._compress_memories()
            
            self._save_memory()
                
        except Exception as e:
            self.logger.error(f"Error adding to memory: {str(e)}")

    def analyze_memory_pattern(self):
        """
        Current pattern:
        1. Load from disk (CPU) -> Convert to CUDA
        2. Process in CUDA 
        3. Save to disk: Convert to CPU -> Pickle -> Restore to CUDA
        """
        # Issue: Constant CPU-GPU transfers create overhead
        with open(self.memory_file, "wb") as f:
            cpu_memory = {k: v.cpu() for k, v in self.memory.items()}  # CPU conversion
            pickle.dump(cpu_memory, f)
        self.memory = {k: v.to(device) for k, v in self.memory.items()}  # Back to GPU

    # 2. Tensor Validation Issues
    def _validate_tensor(self, tensor):
        """Current validation has several potential failure points:"""
        # Issue: Multiple CUDA syncs without error handling
        tensor = tensor.to(device, non_blocking=True)
        torch.cuda.synchronize()  # Forced sync
        
        # Issue: Reshaping without size validation
        if tensor.dim() > 2:
            tensor = tensor.reshape(-1, tensor.size(-1))  # Could fail
            
        # Issue: Redundant CUDA checks and transfers
        if not tensor.is_cuda:
            tensor = tensor.to(device, non_blocking=True)
            torch.cuda.synchronize()

    def _assess_quality(self, embedding, text):
        """Assess the quality of new content based on embedding characteristics and text properties."""
        try:
            # Check embedding quality
            if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                return False

            # Calculate embedding magnitude (normalized)
            magnitude = torch.norm(embedding).item()
            if magnitude < 0.1 or magnitude > 10:
                return False

            # Text quality checks
            min_length = 10
            max_length = 1000
            if len(text.split()) < min_length or len(text.split()) > max_length:
                return False

            # Calculate embedding entropy as a measure of information content
            normalized_embedding = embedding / magnitude
            entropy = -torch.sum(normalized_embedding * torch.log2(torch.abs(normalized_embedding) + 1e-10))
            if entropy.item() < self.quality_threshold:
                return False

            return True
            
        except Exception as e:
            self.logger.error(f"Error in quality assessment: {str(e)}")
            return False

    def _check_novelty(self, embedding):
        """Check if the new content is sufficiently different from existing memories."""
        if not self.embeddings is None and len(self.memory) > 0:
            try:
                similarities = torch.cosine_similarity(embedding, self.embeddings)
                max_similarity = torch.max(similarities).item()
                return max_similarity < self.similarity_threshold
            except Exception as e:
                self.logger.error(f"Error in novelty check: {str(e)}")
                return True
        return True

    def _compress_memories(self, round=0):
        """Recursively compress memories by combining similar embeddings."""
        if round >= self.max_compression_rounds or len(self.memory) < 2:
            return

        try:
            texts = list(self.memory.keys())
            embeddings = torch.stack([self.memory[text] for text in texts])
            
            # Calculate pairwise similarities
            similarities = torch.cosine_similarity(embeddings.unsqueeze(1), 
                                                embeddings.unsqueeze(0))
            
            # Find pairs to merge (above compression_ratio but below 1.0)
            pairs_to_merge = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if self.compression_ratio <= similarities[i, j] < 1.0:
                        pairs_to_merge.append((i, j))

            # Merge similar pairs
            merged_count = 0
            for i, j in pairs_to_merge:
                if texts[i] in self.memory and texts[j] in self.memory:
                    # Combine embeddings and texts
                    new_embedding = (self.memory[texts[i]] + self.memory[texts[j]]) / 2
                    new_text = f"{texts[i]} || {texts[j]}"
                    
                    # Remove old entries
                    del self.memory[texts[i]]
                    del self.memory[texts[j]]
                    
                    # Add merged entry
                    self.memory[new_text] = new_embedding
                    merged_count += 1

            if merged_count > 0:
                self._build_embedding_index()
                # Recursive compression with increased round counter
                self._compress_memories(round + 1)
                
        except Exception as e:
            self.logger.error(f"Error in memory compression: {str(e)}")

    def add_to_memory(self, embedding, text):
        try:
            embedding = self._validate_tensor(embedding)
            if embedding is None or not embedding.is_cuda:
                self.logger.error("Failed to validate tensor for memory storage")
                return

            # Quality and novelty checks
            if not self._assess_quality(embedding, text):
                self.logger.info("Content failed quality check")
                return

            if not self._check_novelty(embedding):
                self.logger.info("Content not sufficiently novel")
                return

            # Add to memory
            self.memory[text] = embedding
            self._build_embedding_index()
            
            # Trigger compression if memory is getting large
            if len(self.memory) > 100:  # Arbitrary threshold
                self._compress_memories()
                
            # Save to disk
            with open(self.memory_file, "wb") as f:
                cpu_memory = {k: v.cpu() for k, v in self.memory.items()}
                pickle.dump(cpu_memory, f)
                
            # Restore CUDA placement
            self.memory = {k: v.to(device) for k, v in self.memory.items()}
                
        except Exception as e:
            self.logger.error(f"Error adding to memory: {str(e)}")

    def _build_embedding_index(self):
        if not self.memory:
            self.embeddings = None
            return

        try:
            texts = list(self.memory.keys())
            embeddings = []
            
            for emb in self.memory.values():
                validated = self._validate_tensor(emb)
                if validated is not None and validated.is_cuda:
                    embeddings.append(validated.cpu())
                else:
                    self.logger.error("Failed to validate embedding during index build")
                    return
                    
            self.embeddings = torch.stack(embeddings)
            
            if self.embeddings.dim() > 2:
                self.embeddings = self.embeddings.reshape(-1, self.embeddings.size(-1))
                
            embeddings_np = self.embeddings.cpu().numpy()
            embeddings_np = np.nan_to_num(embeddings_np, nan=0.0)
            
            if embeddings_np.ndim > 2:
                embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
                
            self.nn_model.fit(embeddings_np)
            self.embeddings = self.embeddings.to(device)
            torch.cuda.empty_cache()
            
        except Exception as e:
            self.logger.error(f"Error building embedding index: {str(e)}")
            self.embeddings = None

    def query_memory(self, embedding):
        if self.embeddings is None or len(self.memory) == 0:
            return None

        try:
            embedding = self._validate_tensor(embedding)
            query_np = embedding.cpu().numpy()
            query_np = np.nan_to_num(query_np, nan=0.0)
            
            if query_np.ndim > 2:
                query_np = query_np.reshape(-1, query_np.shape[-1])
            elif query_np.ndim == 1:
                query_np = query_np.reshape(1, -1)
            
            distance, index = self.nn_model.kneighbors(query_np)
            closest_text = list(self.memory.keys())[index[0][0]]
            return closest_text
        except Exception as e:
            self.logger.error(f"Error querying memory: {e}")
            return None

# Multimodal Preprocessor
class MultimodalPreprocessor:
    def __init__(self):
        #self.text_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct").to(device)
        #self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        self.text_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B").to(device)
        self.image_model = VisionTransformer(pretrained=True).to(device)
        self.audio_model = ResNetAudioAdapter().to(device)
        
        # Define separate projection layers for each modality
        self.text_projection = nn.Linear(1536, 1536).to(device)
        self.image_projection = nn.Linear(1536, 1536).to(device)
        self.audio_projection = nn.Linear(1536, 1536).to(device)
        
        self.memory = PersistentMemory()

        # Image preprocessing pipeline
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate_spectrogram(audio_waveform, sample_rate=22050):
        spectrogram = librosa.feature.melspectrogram(y=audio_waveform, sr=sample_rate, n_mels=128, fmax=8000)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_tensor = torch.tensor(spectrogram_db).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
        return spectrogram_tensor


    def preprocess_audio(self, audio_path):
        """
        Preprocess an audio file for input into the ResNetAudioAdapter.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            torch.Tensor: Processed audio feature tensor.
        """
        # Load audio file
        audio_waveform, sample_rate = librosa.load(audio_path, sr=22050)
        # Generate spectrogram
        spectrogram = self.generate_spectrogram(audio_waveform, sample_rate)  # Already moved to device
        with torch.no_grad():
            spectrogram = spectrogram.to(device)  # Ensure it's on the same device as the model
            return self.audio_model(spectrogram)


    def preprocess_image(self, image_path):
        """
        Preprocess an image for input into the VisionTransformer.
        Args:
            image_path (str or PIL.Image.Image): Path to the image file or a PIL Image object.
        Returns:
            torch.Tensor: Processed image tensor.
        """
        if isinstance(image_path, str):
            image = Image.open(image_path).convert("RGB")
        else:
            image = image_path  # Assume it's already a PIL Image object
        image = self.image_transform(image).unsqueeze(0).to(device)  # Add batch dimension
        with torch.no_grad():
            return self.image_model(image)

        
    def preprocess_text(self, text):
        """
        Preprocess text for input into the text model.
        Args:
            text (str): Input text.
        Returns:
            torch.Tensor: Processed text embeddings.
        """
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = self.text_model(**tokens, output_hidden_states=True)
            # Use the last hidden state to generate embeddings
            hidden_states = outputs.hidden_states[-1].to(device)  # Ensure it's on the correct device
            embeddings = hidden_states.mean(dim=1)
        return embeddings

    def process(self, text=None, image=None, audio=None):
        projected_features = []
        if text is not None:
            text_features = self.preprocess_text(text).to(device)
            text_projected = self.text_projection(text_features)
            projected_features.append(text_projected)
        if image is not None:
            image_features = self.preprocess_image(image).to(device)
            image_projected = self.image_projection(image_features)
            projected_features.append(image_projected)
        if audio is not None:
            audio_features = self.preprocess_audio(audio).to(device)
            audio_projected = self.audio_projection(audio_features)
            projected_features.append(audio_projected)

        if len(projected_features) > 0:
            # Combine projected features by summing them
            combined_features = torch.stack(projected_features).sum(dim=0)
            return combined_features
        else:
            raise ValueError("At least one input modality must be provided.")

    def integrate_memory(self, embedding):
        closest_text = self.memory.query_memory(embedding)
        return closest_text

    def add_to_memory(self, embedding, text):
        self.memory.add_to_memory(embedding, text)

    def generate_text(self, final_state, user_input_text, reasoning_texts=None, max_length=1536):
        # Convert final embedding into memory query
        final_reasoning_text = self.integrate_memory(final_state) or "No related memory found."

        # Construct the prompt
        prompt = f"<|im_start|>user\n{user_input_text}\n<|im_end|>\n"

        if reasoning_texts:
            prompt += "<|im_start|>assistant\n"
            prompt += "\n".join(reasoning_texts) + "\n"

        prompt += f"Memory Context: {final_reasoning_text}\n<|im_end|>\n<|im_start|>assistant\n"

        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        outputs = self.text_model.generate(
            input_ids=prompt_tokens["input_ids"],
            attention_mask=prompt_tokens["attention_mask"],
            max_length=max_length + prompt_tokens["input_ids"].shape[1],
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=1,
            top_p=0.95,
        )
        return self.tokenizer.decode(outputs[0][prompt_tokens["input_ids"].shape[1]:], skip_special_tokens=True)


class EmotionalState:
    def __init__(self):
        self.emotion_memory = deque(maxlen=5)
        self.current_emotion = None
        self.emotion_intensity = 0.0
        self.emotion_decay = 0.95
        
    def update_state(self, new_emotion, intensity=1.0):
        self.emotion_memory.append(self.current_emotion)
        self.current_emotion = new_emotion
        self.emotion_intensity = intensity
        
    def decay_emotion(self):
        self.emotion_intensity *= self.emotion_decay

class EmotionalLayer(nn.Module):
    # Class-level emotion mapping
    EMOTION_VA_MAP = {
        'admiration': (0.6, 0.3),
        'amusement': (0.8, 0.4),
        'anger': (-0.6, 0.8),
        'annoyance': (-0.4, 0.4),
        'approval': (0.4, 0.2),
        'caring': (0.6, 0.3),
        'confusion': (-0.2, 0.3),
        'curiosity': (0.3, 0.4),
        'desire': (0.5, 0.6),
        'disappointment': (-0.5, -0.2),
        'disapproval': (-0.4, 0.3),
        'disgust': (-0.7, 0.4),
        'embarrassment': (-0.4, 0.4),
        'excitement': (0.7, 0.8),
        'fear': (-0.7, 0.7),
        'gratitude': (0.7, 0.3),
        'grief': (-0.8, -0.4),
        'joy': (0.8, 0.6),
        'love': (0.9, 0.5),
        'nervousness': (-0.3, 0.7),
        'optimism': (0.6, 0.4),
        'pride': (0.7, 0.5),
        'realization': (0.2, 0.3),
        'relief': (0.4, -0.2),
        'remorse': (-0.6, -0.3),
        'sadness': (-0.7, -0.3),
        'surprise': (0.3, 0.7),
        'neutral': (0.0, 0.0)
    }
    def __init__(self, hidden_dim):
        super(EmotionalLayer, self).__init__()
        self.valence = nn.Parameter(torch.tensor(0.0))
        self.arousal = nn.Parameter(torch.tensor(0.0))
        self.emotional_state = EmotionalState()
        self.emotion_embedding = nn.Embedding(len(self.EMOTION_VA_MAP), hidden_dim)
        self.transition_matrix = nn.Parameter(torch.randn(len(self.EMOTION_VA_MAP), len(self.EMOTION_VA_MAP)))
        self.hidden_dim = hidden_dim
        self.device = device
        
    def map_emotion_to_valence_arousal(self, emotion_label):
        return self.EMOTION_VA_MAP.get(emotion_label, (0.0, 0.0))

    def compute_emotion_influence(self, x, emotion_dict):
        # Convert emotion dictionary to tensor on the correct device
        emotion_vector = torch.zeros(len(self.EMOTION_VA_MAP), device=device)
        
        # Fill the emotion vector with probabilities
        for label, score in emotion_dict.items():
            if label in self.EMOTION_VA_MAP:
                idx = list(self.EMOTION_VA_MAP.keys()).index(label)
                emotion_vector[idx] = score
                
        # Normalize probabilities
        emotion_vector = torch.softmax(emotion_vector, dim=0)
        
        # Ensure transition matrix is on the correct device
        transition_weights = torch.softmax(self.transition_matrix, dim=1)
        next_state_probs = torch.matmul(emotion_vector, transition_weights)
        
        # Update emotional state
        max_emotion_idx = torch.argmax(next_state_probs)
        self.emotional_state.update_state(max_emotion_idx.item(), next_state_probs[max_emotion_idx].item())
        
        return next_state_probs

    def forward(self, x, emotion_input=None):
        # Handle different input dimensions
        original_dim = x.dim()
        
        # Ensure x is 2D: [batch_size, hidden_dim]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3:
            x = x.view(-1, x.size(-1))
            
        # Ensure hidden dimension matches
        if x.size(-1) != self.hidden_dim:
            x = F.pad(x, (0, self.hidden_dim - x.size(-1))) if x.size(-1) < self.hidden_dim else x[:, :self.hidden_dim]
        
        # Process emotions and apply modulation
        if emotion_input is not None:
            emotion_dict = {}
            for item in emotion_input:
                emotion_dict[item['label']] = item['score']
            
            if emotion_dict:
                self.compute_emotion_influence(x, emotion_dict)
                dominant_emotion = max(emotion_dict.items(), key=lambda x: x[1])[0]
                v, a = self.map_emotion_to_valence_arousal(dominant_emotion)
                self.valence.data = torch.tensor(v, dtype=self.valence.dtype)
                self.arousal.data = torch.tensor(a, dtype=self.arousal.dtype)
        
        # Apply emotional modulation
        arousal_gain = torch.sigmoid(self.arousal)
        emotional_modulation = 1.0 + (self.valence * arousal_gain)
        
        # Apply emotional state influence
        if self.emotional_state.current_emotion is not None:
            emotion_idx = torch.tensor(min(self.emotional_state.current_emotion, len(self.EMOTION_VA_MAP) - 1))
            emotion_embedding = self.emotion_embedding(emotion_idx.to(x.device))
            emotional_context = emotion_embedding.unsqueeze(0).expand_as(x) * self.emotional_state.emotion_intensity
            x = x + emotional_context
            
        # Apply modulation and restore original dimensions
        output = x * emotional_modulation.view(-1, 1)
        
        if original_dim == 1:
            output = output.squeeze(0)
        elif original_dim == 3:
            output = output.view(*x.shape[:-1], -1)
            
        return output

class EnergyModule(nn.Module):
    def __init__(self, init_energy=100):
        super(EnergyModule, self).__init__()
        self.energy = nn.Parameter(torch.tensor(init_energy, dtype=torch.float32), requires_grad=False)
        self.max_energy = init_energy
        self.recovery_rate = 0.1
        
    def forward(self, x, arousal, valence, emotional_state):
        energy_loss = torch.norm(x).item() * 0.01 * (1 + emotional_state.emotion_intensity)
        new_energy = self.energy - energy_loss

        if valence > 0:
            recovery = self.recovery_rate * (1 + valence) * (self.max_energy - new_energy)
            new_energy = min(new_energy + recovery, self.max_energy)
        
        fatigue_threshold = 0.2 * self.max_energy
        if new_energy < fatigue_threshold:
            fatigue_factor = (new_energy / fatigue_threshold)
            x = x * fatigue_factor
            
        self.energy.data = new_energy.clone().detach()
        return x
    
class EmotionalFeedbackModule(nn.Module):
    def __init__(self, base_learning_rate=0.001, emotion_sensitivity=0.5):
        super(EmotionalFeedbackModule, self).__init__()
        self.base_lr = base_learning_rate
        self.emotion_sensitivity = emotion_sensitivity
        self.feedback_history = []
        self.preprocessor = MultimodalPreprocessor()
        self.tokenizer = self.preprocessor.tokenizer
        self.device = torch.device("cuda")
    def compute_feedback_weight(self, emotional_state, model = None):
        """
        Compute weight adjustment based on emotional valence and intensity
        Positive emotions -> positive feedback (reward)
        Negative emotions -> negative feedback (punishment)
        """
        valence = emotional_state['valence']
        intensity = emotional_state['intensity']
        
        model = None 
        
        feedback = valence * intensity * self.emotion_sensitivity
        feedback_tensor = self.tokenizer.encode(str(feedback))  # Convert to tensor
        feedback_weight = torch.tanh(feedback_tensor * self.base_lr)
        if model is not None:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.base_lr)
        else:
            return feedback
        def feedback_hook(optimizer):
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    param.data += feedback_weight * param.grad
                    
        optimizer.add_hook(feedback_hook)
        return feedback_weight
    
    def apply_emotional_feedback(self, module, emotional_state):
        """
        Apply emotional feedback to module parameters
        """
        feedback_weight = self.compute_feedback_weight(emotional_state)
        
        # Store feedback for analysis
        self.feedback_history.append({
            'emotion': emotional_state['current_emotion'],
            'feedback_weight': feedback_weight,
            'intensity': emotional_state['intensity']
        })
        
        # Apply feedback to parameters
        with torch.no_grad():
            for param in module.parameters():
                if param.requires_grad:
                    # Positive feedback reinforces current weights
                    # Negative feedback pushes weights toward zero
                    if feedback_weight > 0:
                        param.data += feedback_weight * param.data
                    else:
                        param.data += feedback_weight * (param.data - param.data.mean())
                        
    def get_feedback_stats(self):
        """
        Return statistics about feedback history
        """
        if not self.feedback_history:
            return None
            
        recent_feedback = self.feedback_history[-10:]
        return {
            'average_feedback': sum(f['feedback_weight'] for f in recent_feedback) / len(recent_feedback),
            'strongest_positive': max((f for f in recent_feedback if f['feedback_weight'] > 0), 
                                   key=lambda x: x['feedback_weight'], default=None),
            'strongest_negative': min((f for f in recent_feedback if f['feedback_weight'] < 0), 
                                   key=lambda x: x['feedback_weight'], default=None)
        }

class EmotionalEnergyModel(nn.Module):
    def __init__(self, hidden_dim, init_energy=1536):
        super(EmotionalEnergyModel, self).__init__()
        self.emotion_layer = EmotionalLayer(hidden_dim)
        self.energy_module = EnergyModule(init_energy)
        self.classifier = classifier
        self.feedback_module = EmotionalFeedbackModule()
        self.tokenizer = self.feedback_module.tokenizer
        
    def get_emotional_state(self):
        current_emotion_idx = self.emotion_layer.emotional_state.current_emotion
        if current_emotion_idx is not None:
            emotion_list = list(self.emotion_layer.EMOTION_VA_MAP.keys())
            current_emotion = emotion_list[current_emotion_idx] if current_emotion_idx < len(emotion_list) else "unknown"
        else:
            current_emotion = "none"
            
        return {
            'current_emotion': current_emotion,
            'intensity': self.emotion_layer.emotional_state.emotion_intensity,
            'valence': self.emotion_layer.valence.item(),
            'arousal': self.emotion_layer.arousal.item(),
            'energy': self.energy_module.energy.item()
        }
        
    def forward(self, x):
        if isinstance(x, str):
            # Process text input
            model_outputs = self.classifier([x])[0]  # Get first item since we're processing single string
            processed_tensor = self.tokenizer.encode(str(model_outputs))  # Placeholder tensor
            output = self.emotion_layer(processed_tensor, model_outputs)
            emotional_state = self.get_emotional_state()
            self.feedback_module.apply_emotional_feedback(self.emotion_layer, emotional_state)
            return {
                'output': output,
                'classification': model_outputs,
                'emotional_state': self.get_emotional_state(),
                'feedback_stats': self.feedback_module.get_feedback_stats()
            }
            
        else:
            # Process tensor input
            x = self.emotion_layer(x)
            x = self.energy_module(
                x, 
                self.emotion_layer.arousal.item(),
                self.emotion_layer.valence.item(),
                self.emotion_layer.emotional_state
            )
            return x



class LSTMTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1536, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Initialize attention as an instance of SparseFocusedGroupAttention
        self.attention = SparseFocusedGroupAttention(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden_state=None):
        # Ensure input has correct shape
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply attention first
        attended_x = self.attention(x)
        
        # Process through LSTM
        out, hidden_state = self.lstm(attended_x, hidden_state)
        
        # Get last hidden state
        last_hidden_state = out[:, -1, :]
        
        # Process through linear layers
        out = torch.relu(self.linear(last_hidden_state))
        output = self.output_layer(out)
        
        return output, hidden_state
    
class ChainOfThought:
    def __init__(self, model, preprocessor, max_iterations=1, verbose=False):
        self.model = model.to(device)
        self.preprocessor = preprocessor
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.attention_weights = nn.Parameter(torch.ones(4, device=device)/4)
        self.context_gate = nn.GRU(1536, 1536).to(device)  # Move GRU to device
        self.thought_history = []
        self.emotional_state = torch.zeros(1536, device=device)  # Initialize emotional state
        # Add repetition penalty parameters
        self.repetition_penalty = .9
        self.topic_coherence_weight = 1.1
        self.query_attention_weight = 3
        self.similarity_threshold = 0.5

    def compute_similarity(self, text1, text2):
        """Compute semantic similarity between two texts."""
        try:
            emb1 = self.preprocessor.preprocess_text(text1)
            emb2 = self.preprocessor.preprocess_text(text2)
            similarity = F.cosine_similarity(emb1, emb2)
            return similarity.item()
        except:
            return 0.0
            
    def check_repetition(self, thought):
        """Check if thought is too similar to previous thoughts."""
        for prev_thought in self.thought_history[-2:]:
            if self.compute_similarity(thought, prev_thought) > self.similarity_threshold:
                return True
        return False

    def apply_emotional_modulation(self, thought_tensor):
        """Modulate thoughts based on emotional state."""
        emotional_intensity = torch.norm(self.emotional_state)
        emotional_direction = F.normalize(self.emotional_state, dim=0)
        
        # Project thought onto emotional direction
        thought_projection = torch.matmul(thought_tensor, emotional_direction.unsqueeze(1))
        
        # Modulate thought based on emotional intensity
        modulated_thought = thought_tensor + (thought_projection * emotional_direction * emotional_intensity * 0.1)
        return modulated_thought

    def apply_attention(self, inputs, query_embedding=None):
        weighted_inputs = []
        for i, input_tensor in enumerate(inputs):
            if input_tensor is not None:
                if isinstance(input_tensor, str):
                    input_tensor = self.preprocessor.preprocess_text(input_tensor)
                if isinstance(input_tensor, torch.Tensor):
                    # Move tensor to correct device
                    input_tensor = input_tensor.to(device)
                    
                    if input_tensor.dim() > 2:
                        input_tensor = input_tensor.squeeze()
                    if input_tensor.dim() == 1:
                        input_tensor = input_tensor.unsqueeze(0)
                    
                    if query_embedding is not None and i == 0:
                        query_embedding = query_embedding.to(device)  # Ensure query_embedding is on device
                        query_similarity = F.cosine_similarity(
                            input_tensor, 
                            query_embedding, 
                            dim=-1
                        ).unsqueeze(-1)
                        input_tensor = input_tensor * (1 + self.query_attention_weight * query_similarity)
                    
                    weighted_inputs.append(input_tensor * self.attention_weights[i])
                    
        if weighted_inputs:
            # Ensure all tensors are on the same device before stacking
            weighted_inputs = [x.to(device) for x in weighted_inputs]
            return torch.stack(weighted_inputs).sum(dim=0)
        return None
    

    def decode_thought(self, thought_tensor, input_context, query_text):
        try:
            if isinstance(thought_tensor, torch.Tensor):
                # Ensure tensors are on correct device
                thought_tensor = thought_tensor.to(device)
                
                if thought_tensor.dim() > 2:
                    thought_tensor = thought_tensor.squeeze()
                if thought_tensor.dim() == 1:
                    thought_tensor = thought_tensor.unsqueeze(0)
                
                projected_thought = self.preprocessor.text_projection(thought_tensor)
                
                if isinstance(input_context, torch.Tensor):
                    input_context = input_context.to(device)
                    
                    if input_context.dim() > 2:
                        input_context = input_context.squeeze()
                    if input_context.dim() == 1:
                        input_context = input_context.unsqueeze(0)
                    
                    # Ensure inputs are on the same device as the model
                    projected_thought = projected_thought.to(device)
                    input_context = input_context.to(device)
                    
                    gated_thought, _ = self.context_gate(projected_thought, input_context)
                    
                    history_context = self.thought_history[-2:] if self.thought_history else []
                    context_prompt = (
                        f"Query: {query_text}\n"
                        f"Previous thoughts: {' -> '.join(history_context)}\n"
                        f"Goal: Think in short nonrepetitive concise steps to achieve the best response to the query."
                    )
                    
                    # Ensure gated_thought is on the correct device before generating text
                    gated_thought = gated_thought.to(device)
                    
                    decoded_thought = self.preprocessor.generate_text(
                        gated_thought.squeeze(0),
                        user_input_text=context_prompt,
                        max_length=1536
                    )
                    
                    if self.check_repetition(decoded_thought):
                        decoded_thought = self.preprocessor.generate_text(
                            gated_thought.squeeze(0),
                            user_input_text=context_prompt + "\nThink about the steps that follow the last steps you thought about, be concise",
                            max_length=1536
                        )
                    
                    return decoded_thought.strip()
            return "[Invalid thought tensor]"
        except Exception as e:
            return f"[Decode error: {str(e)}]"
        

    def decode_thought(self, thought_tensor, input_context, query_text):
        try:
            if isinstance(thought_tensor, torch.Tensor):
                thought_tensor = thought_tensor.to(device)
                
                if thought_tensor.dim() > 2:
                    thought_tensor = thought_tensor.squeeze()
                if thought_tensor.dim() == 1:
                    thought_tensor = thought_tensor.unsqueeze(0)
                
                # Apply emotional modulation to thought
                thought_tensor = self.apply_emotional_modulation(thought_tensor)
                projected_thought = self.preprocessor.text_projection(thought_tensor)
                
                if isinstance(input_context, torch.Tensor):
                    input_context = input_context.to(device)
                    
                    if input_context.dim() > 2:
                        input_context = input_context.squeeze()
                    if input_context.dim() == 1:
                        input_context = input_context.unsqueeze(0)
                    
                    projected_thought = projected_thought.to(device)
                    input_context = input_context.to(device)
                    
                    gated_thought, _ = self.context_gate(projected_thought, input_context)
                    
                    history_context = self.thought_history[-2:] if self.thought_history else []
                    context_prompt = (
                        f"Query: {query_text}\n"
                        f"Previous thoughts: {' -> '.join(history_context)}\n"
                        f"Goal: Think in short nonrepetitive concise steps to achieve the best response to the query."
                    )
                    
                    gated_thought = gated_thought.to(device)
                    
                    decoded_thought = self.preprocessor.generate_text(
                        gated_thought.squeeze(0),
                        user_input_text=context_prompt,
                        max_length=1536
                    )
                    
                    if self.check_repetition(decoded_thought):
                        decoded_thought = self.preprocessor.generate_text(
                            gated_thought.squeeze(0),
                            user_input_text=context_prompt + "\nThink about the steps that follow the last steps you thought about, be concise",
                            max_length=1536
                        )
                    
                    return f"<think>{decoded_thought.strip()}</think>"
            return "<think>[Invalid thought tensor]</think>"
        except Exception as e:
            return f"<think>[Decode error: {str(e)}]</think>"

    def update_emotional_state(self, current_state, query_embedding):
        """Update emotional state based on current thought and query."""
        emotional_decay = 0.95  # Decay factor for emotions
        self.emotional_state = self.emotional_state * emotional_decay
        
        # Compute emotional response to current thought
        emotional_response = torch.tanh(F.cosine_similarity(
            current_state,
            query_embedding,
            dim=-1
        ).unsqueeze(-1) * current_state)
        
        # Update emotional state
        self.emotional_state = self.emotional_state + (0.1 * emotional_response.squeeze())

    def reason(self, input_data, query_text, memory_context=None, vision_context=None, show_thinking=False):
        self.thought_history = []
        hidden_state = None

        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)
        else:
            input_data = self.preprocessor.preprocess_text(input_data).to(device)
        
        query_embedding = self.preprocessor.preprocess_text(query_text).to(device)
        
        if isinstance(input_data, torch.Tensor):
            if input_data.dim() == 2:
                input_data = input_data.unsqueeze(1)
            elif input_data.dim() == 1:
                input_data = input_data.unsqueeze(0).unsqueeze(1)
        
        current_state = input_data.squeeze(1).to(device)
        contexts = [current_state, vision_context, memory_context, current_state]
        
        for step in range(self.max_iterations):
            if show_thinking:
                print(f"\nThinking Step {step + 1}/{self.max_iterations}...")
            
            focused_input = self.apply_attention(contexts, query_embedding)
            if focused_input is None:
                break
                
            lstm_input = focused_input.unsqueeze(1).to(device)
            output, hidden_state = self.model(lstm_input, hidden_state)
            
            current_thought = self.decode_thought(output, current_state, str(query_text))
            
            if self.check_repetition(current_thought):
                output = output * (1.0 / self.repetition_penalty)
                current_thought = self.decode_thought(output, current_state, str(query_text))
            
            self.thought_history.append(current_thought)
            
            if show_thinking:
                print(f"{current_thought}")
            
            current_state = self.preprocessor.preprocess_text(current_thought).to(device)
            self.update_emotional_state(current_state, query_embedding)
            
            query_similarity = F.cosine_similarity(
                current_state, 
                query_embedding, 
                dim=-1
            ).unsqueeze(-1)
            current_state = current_state * (1 + self.topic_coherence_weight * query_similarity)
            contexts[-1] = current_state

        return output.to(device), self.emotional_state
    


class ONI(nn.Module):
    def __init__(self):
        super(ONI, self).__init__()
        self.preprocessor = MultimodalPreprocessor()
        self.model = LSTMTransformer(input_dim=1536, hidden_dim=1536, output_dim=1536).to(device)
        self.chain_of_thought = ChainOfThought(self.model, self.preprocessor)
        self.metacognition = MetaCognitionModule(1536).to(device)
        self.emotions = EmotionalEnergyModel(1536).to(device)
        self.text_to_image = DiffusionPipeline.from_pretrained(
            "cagliostrolab/animagine-xl-3.1", 
            torch_dtype=torch.float16, 
            use_safetensors=True, 
        )
        self.browser = None  # Placeholder for browser instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.setup_logging()

    def setup_logging(self):
        """Set up logging for the system."""
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    # Keyboard and Mouse Controls
    def type_text(self, text):
        """Simulate typing text using the keyboard."""
        try:
            pyautogui.typewrite(text)
            self.logger.info(f"Typed text: {text}")
        except Exception as e:
            self.logger.error(f"Error typing text: {str(e)}")

    def move_mouse(self, x, y):
        """Move the mouse to the specified coordinates."""
        try:
            pyautogui.moveTo(x, y)
            self.logger.info(f"Moved mouse to ({x}, {y})")
        except Exception as e:
            self.logger.error(f"Error moving mouse: {str(e)}")

    def click_mouse(self, button='left'):
        """Simulate a mouse click."""
        try:
            pyautogui.click(button=button)
            self.logger.info(f"Clicked mouse button: {button}")
        except Exception as e:
            self.logger.error(f"Error clicking mouse: {str(e)}")

    # Browser Automation
    def open_browser(self, url="https://www.google.com"):
        """Open a browser and navigate to a specified URL."""
        try:
            self.browser = webdriver.Chrome()  # Ensure Chrome WebDriver is installed
            self.browser.get(url)
            self.logger.info(f"Opened browser and navigated to {url}")
        except Exception as e:
            self.logger.error(f"Error opening browser: {str(e)}")

    def search_web(self, query):
        """Perform a web search using the browser."""
        if self.browser is None:
            self.logger.error("Browser is not open.")
            return

        try:
            search_box = self.browser.find_element("name", "q")
            search_box.clear()
            search_box.send_keys(query)
            search_box.send_keys(Keys.RETURN)
            self.logger.info(f"Performed web search for: {query}")
        except Exception as e:
            self.logger.error(f"Error performing web search: {str(e)}")

    def close_browser(self):
        """Close the browser."""
        if self.browser is not None:
            try:
                self.browser.quit()
                self.logger.info("Closed browser.")
            except Exception as e:
                self.logger.error(f"Error closing browser: {str(e)}")

    # Internal Tools
    def open_file(self, file_path):
        """Open a file using the default application."""
        try:
            if os.path.exists(file_path):
                os.startfile(file_path)  # Windows
                # For macOS: subprocess.run(['open', file_path])
                # For Linux: subprocess.run(['xdg-open', file_path])
                self.logger.info(f"Opened file: {file_path}")
            else:
                self.logger.error(f"File not found: {file_path}")
        except Exception as e:
            self.logger.error(f"Error opening file: {str(e)}")

    def monitor_system(self):
        """Monitor system resources (CPU, memory, etc.)."""
        try:
            import psutil
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            self.logger.info(f"System Monitor - CPU: {cpu_usage}%, Memory: {memory_usage}%")
            return {"cpu_usage": cpu_usage, "memory_usage": memory_usage}
        except Exception as e:
            self.logger.error(f"Error monitoring system: {str(e)}")
            return None

    # Existing Methods with Optimizations
    def process_input(self, text=None, image=None):
        """Process input data with error handling."""
        try:
            input_data = self.preprocessor.process(text, image)
            if input_data.dim() == 2:
                input_data = input_data.unsqueeze(1)  # Add sequence dimension
            input_data = input_data.to(torch.float32)
            return input_data
        except Exception as e:
            self.logger.error(f"Error processing input: {str(e)}")
            return None

    def run(self, text=None, image=None, show_thinking=False):
        """Run the system with integrated emotional processing."""
        if text is None:
            raise ValueError("`text` (user input) must be provided.")

        try:
            input_data = self.process_input(text, image)
            if input_data is None:
                return "Error processing input."

            memory_context = self.preprocessor.memory.query_memory(input_data) if text else None
            vision_context = input_data if image else None
            input_data = self.emotions(input_data)
            final_state, emotional_state = self.chain_of_thought.reason(
                input_data=input_data,
                query_text=text,
                memory_context=memory_context,
                vision_context=vision_context, 
                show_thinking=show_thinking
            )

            # Process emotional state
            
            final_state, confidence, conflicts = self.metacognition(final_state)

            if show_thinking and conflicts:
                print("\nDetected Principle Conflicts:")
                for i, j, score in conflicts:
                    print(f"Conflict between principles {i} and {j} (score: {score:.2f})")
                print(f"Confidence in resolution: {confidence.item():.2f}\n")
                print(f"Emotional intensity: {torch.norm(emotional_state).item():.2f}\n")

            response = self.preprocessor.generate_text(final_state, user_input_text=text)
            self.preprocessor.add_to_memory(final_state, text)

            return response
        except Exception as e:
            self.logger.error(f"Error in processing: {str(e)}")
            return "I encountered an error while processing your input."
