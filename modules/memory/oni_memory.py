import os
import json
import numpy as np
import random
import PyPDF2
import sqlite3
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.optim as optim
import heapq
import threading
import time
import pygame  # For rendering
import math

# ===========================
# Memory Components
# ===========================



            
class SpatialMemoryModule:
    def __init__(self, room_size: Tuple[int, int], overlap: float = 0.2, max_memory: int = 100):
        """
        Initializes the spatial memory with parameters defining room size, overlap, and memory constraints.

        Args:
            room_size (tuple): Dimensions of each room (width, height).
            overlap (float): Fractional overlap between adjacent rooms.
            max_memory (int): Maximum number of rooms to retain in memory.
        """
        self.room_width, self.room_height = room_size
        self.overlap = overlap
        self.current_position = (0, 0)  # Starting at origin
        self.memory = {}  # Dictionary to store room data indexed by position
        self.max_memory = max_memory

    def get_current_room_key(self) -> Tuple[int, int]:
        """
        Determines the key for the current room based on Oni's position.

        Returns:
            tuple: Coordinates representing the current room.
        """
        x, y = self.current_position
        room_x = int(x // (self.room_width * (1 - self.overlap)))
        room_y = int(y // (self.room_height * (1 - self.overlap)))
        return (room_x, room_y)

    def update_position(self, new_position: Tuple[int, int]) -> bool:
        """
        Updates Oni's current position and determines if a new room needs to be loaded.

        Args:
            new_position (tuple): New (x, y) coordinates.

        Returns:
            bool: True if a new room is entered, False otherwise.
        """
        old_room = self.get_current_room_key()
        self.current_position = new_position
        new_room = self.get_current_room_key()
        if new_room != old_room:
            return True
        return False

    def load_room(self, room_key: Tuple[int, int], room_data: Dict):
        """
        Loads data for a new room into memory.

        Args:
            room_key (tuple): Coordinates representing the room.
            room_data (dict): Data associated with the room.
        """
        self.memory[room_key] = room_data
        # If memory exceeds max_memory, remove the least recently used room
        if len(self.memory) > self.max_memory:
            oldest_room = next(iter(self.memory))
            del self.memory[oldest_room]

    def get_current_room_data(self) -> Optional[Dict]:
        """
        Retrieves data for the current room.

        Returns:
            dict or None: Data of the current room or None if not loaded.
        """
        room_key = self.get_current_room_key()
        return self.memory.get(room_key, None)


class HeuristicManager:
    def __init__(self, heuristic_function, max_priority: int = 100):
        """
        Initializes the Heuristic Manager.

        Args:
            heuristic_function (callable): Function to compute priority based on room key.
            max_priority (int): Maximum number of rooms to prioritize.
        """
        self.heuristic_function = heuristic_function
        self.priority_queue = []
        self.max_priority = max_priority

    def add_room(self, room_key: Tuple[int, int]):
        priority = self.heuristic_function(room_key)
        heapq.heappush(self.priority_queue, (priority, room_key))
        # Ensure the queue doesn't exceed max_priority
        if len(self.priority_queue) > self.max_priority:
            heapq.heappop(self.priority_queue)

    def get_next_room(self) -> Optional[Tuple[int, int]]:
        if self.priority_queue:
            return heapq.heappop(self.priority_queue)[1]
        return None


class EpisodicEmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, compression_rate: float = 0.9):
        super(EpisodicEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compression_rate = compression_rate

        # Encoder for different media types (multi-modal)
        self.media_encoder = nn.ModuleDict({
            'text': nn.Linear(input_dim, output_dim),
            'image': nn.Linear(input_dim, output_dim),
            'audio': nn.Linear(input_dim, output_dim),
            # Add other media types as needed
        })

        # Chain compression mechanism
        self.compression_layer = nn.Linear(output_dim, int(output_dim * self.compression_rate))

        # Infinite space handler
        self.embeddings = []

    def forward(self, x, media_type: str):
        if media_type not in self.media_encoder:
            raise ValueError(f"Unsupported media type: {media_type}")

        # Encode input
        x = self.media_encoder[media_type](x)

        # Chain compression
        x = self.compression_layer(x)

        # Store embedding (infinite space handler)
        self.embeddings.append(x)

        return x


class SemanticMemoryLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(SemanticMemoryLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Text encoder
        self.text_encoder = nn.Linear(input_dim, output_dim)

        # Shared connection to episodic layer
        self.episodic_embedding_layer = None  # Set externally

    def forward(self, text_input: torch.Tensor, media_reference: Optional[torch.Tensor] = None):
        # Encode text
        text_embedding = self.text_encoder(text_input)

        # If media reference is provided, connect with episodic embedding
        if media_reference is not None and self.episodic_embedding_layer is not None:
            media_embedding = self.episodic_embedding_layer(media_reference, media_type='image')  # Example for image
            combined_embedding = torch.cat((text_embedding, media_embedding), dim=-1)
            return combined_embedding

        return text_embedding


class SparseHopfieldNetwork:
    def __init__(self, size: int, sparsity: float = 0.1):
        self.size = size
        self.sparsity = sparsity
        self.weights = np.zeros((size, size))
        self.create_sparse_connections()

    def create_sparse_connections(self):
        num_connections = int(self.size * self.size * self.sparsity)
        for _ in range(num_connections):
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            while i == j or self.weights[i, j] != 0:
                i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            self.weights[i, j] = 1
            self.weights[j, i] = 1  # Ensure symmetry

    def train(self, patterns: List[List[int]]):
        for p in patterns:
            p = np.array(p)
            outer_product = np.outer(p, p)
            self.weights += outer_product
        self.weights = self.weights / len(patterns)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern: List[int], steps: int = 10) -> np.ndarray:
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
        return pattern


class TextPatternFinder:
    def __init__(self, tokenizer, min_pattern_length: int = 3, max_pattern_length: int = 10, min_occurrences: int = 2):
        self.tokenizer = tokenizer
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_occurrences = min_occurrences
        self.corpus_patterns = defaultdict(list)
        self.ltm_patterns = defaultdict(list)
        self.hopfield_network = None

    def find_patterns(self, corpus: List[str], ltm: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        self._find_patterns_in_text(corpus, self.corpus_patterns)
        self._find_patterns_in_text(ltm, self.ltm_patterns)
        return self.corpus_patterns, self.ltm_patterns

    def _find_patterns_in_text(self, text: List[str], pattern_dict: Dict[str, List[int]]):
        tokens = self.tokenizer.tokenize(' '.join(text))
        num_tokens = len(tokens)

        for i in range(num_tokens):
            for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                if i + length > num_tokens:
                    break
                pattern = ' '.join(tokens[i:i+length])
                pattern_dict[pattern].append(i)

        # Remove patterns that do not meet the minimum occurrence threshold
        for pattern, occurrences in list(pattern_dict.items()):
            if len(occurrences) < self.min_occurrences:
                del pattern_dict[pattern]

    def consolidate_patterns(self) -> Dict[str, List[int]]:
        combined_patterns = {**self.corpus_patterns, **self.ltm_patterns}
        unique_patterns = {p: combined_patterns[p] for p in combined_patterns if p in self.corpus_patterns and p in self.ltm_patterns}
        return unique_patterns

    def use_hopfield_network(self, unique_patterns: Dict[str, List[int]]):
        pattern_vectors = [self._pattern_to_vector(p) for p in unique_patterns.keys()]
        self.hopfield_network = SparseHopfieldNetwork(size=len(unique_patterns))
        self.hopfield_network.train(pattern_vectors)

    def _pattern_to_vector(self, pattern: str) -> List[int]:
        pattern_tokens = self.tokenizer.tokenize(pattern)
        pattern_vector = [0] * self.tokenizer.vocab_size
        for token in pattern_tokens:
            index = self.tokenizer.token_to_id(token)
            if 0 <= index < self.tokenizer.vocab_size:
                pattern_vector[index] = 1
        return pattern_vector

    def update_hopfield_network(self, new_patterns: Dict[str, List[int]]):
        new_pattern_vectors = [self._pattern_to_vector(p) for p in new_patterns.keys()]
        self.hopfield_network.train(new_pattern_vectors)


# ===========================
# New Memory Components
# ===========================

class EpisodicBuffer(nn.Module):
    """
    Working memory buffer that combines episodic and semantic memory.
    Acts as a temporary storage for ongoing cognitive processes.
    """
    def __init__(self, hidden_dim: int, buffer_size: int = 10, decay_rate: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.decay_rate = decay_rate
        
        # Initialize buffer as a learnable parameter
        self.register_parameter(
            'buffer',
            nn.Parameter(torch.zeros(buffer_size, hidden_dim), requires_grad=True)
        )
        
        # Importance scores for each buffer item
        self.register_buffer('importance', torch.zeros(buffer_size))
        
        # Recency scores for each buffer item (timestamp)
        self.register_buffer('recency', torch.zeros(buffer_size))
        
        # Current time step
        self.time_step = 0
        
        # Attention mechanism for buffer access
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        
        # Gating mechanism for writing to buffer
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, episodic_input: Optional[torch.Tensor] = None, 
                semantic_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Access the buffer using attention mechanism and optionally update it.
        
        Args:
            query: Query tensor for retrieving from buffer
            episodic_input: Optional new episodic memory to add to buffer
            semantic_input: Optional new semantic memory to add to buffer
            
        Returns:
            Retrieved memory from buffer
        """
        batch_size = query.size(0)
        
        # Apply decay to importance based on recency
        self._apply_decay()
        
        # Expand buffer for batch processing
        expanded_buffer = self.buffer.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Retrieve from buffer using attention
        retrieved, attention_weights = self.attention(
            query=query,
            key=expanded_buffer,
            value=expanded_buffer
        )
        
        # Update buffer if new inputs are provided
        if episodic_input is not None or semantic_input is not None:
            self._update_buffer(episodic_input, semantic_input, query)
            
        # Increment time step
        self.time_step += 1
        
        return retrieved
    
    def _apply_decay(self):
        """Apply decay to importance scores based on recency"""
        if self.time_step > 0:
            time_since_update = self.time_step - self.recency
            decay_factor = torch.exp(-self.decay_rate * time_since_update)
            self.importance = self.importance * decay_factor
    
    def _update_buffer(self, episodic_input: Optional[torch.Tensor], 
                      semantic_input: Optional[torch.Tensor],
                      query: torch.Tensor):
        """
        Update buffer with new episodic and semantic information
        
        Args:
            episodic_input: New episodic memory
            semantic_input: New semantic memory
            query: Current query for context
        """
        # Combine inputs if both are provided
        if episodic_input is not None and semantic_input is not None:
            combined_input = episodic_input + semantic_input
        elif episodic_input is not None:
            combined_input = episodic_input
        elif semantic_input is not None:
            combined_input = semantic_input
        else:
            return
            
        # Calculate importance of new input
        with torch.no_grad():
            # Compute similarity with query as a measure of importance
            similarity = F.cosine_similarity(combined_input, query, dim=-1)
            new_importance = similarity.mean()
            
            # Find index to update (lowest importance or oldest)
            if torch.all(self.importance > 0):
                # Buffer is full, replace least important item
                update_idx = torch.argmin(self.importance)
            else:
                # Buffer has empty slots, use the first one
                update_idx = torch.where(self.importance == 0)[0][0]
            
            # Calculate write gate - determines how much to update
            concat_input = torch.cat([combined_input, self.buffer[update_idx].unsqueeze(0)], dim=-1)
            write_strength = self.write_gate(concat_input).item()
            
            # Update buffer
            new_value = write_strength * combined_input + (1 - write_strength) * self.buffer[update_idx]
            self.buffer.data[update_idx] = new_value.squeeze(0)
            
            # Update importance and recency
            self.importance[update_idx] = new_importance
            self.recency[update_idx] = self.time_step

class ModernContinuousHopfieldNetwork(nn.Module):
    """
    Modern continuous Hopfield network with improved storage capacity and retrieval.
    Based on the paper "Hopfield Networks is All You Need" (Ramsauer et al., 2020).
    """
    def __init__(self, hidden_dim: int, beta: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta  # Inverse temperature parameter
        
        # Projection layers
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Stored patterns
        self.register_buffer('stored_keys', torch.zeros(0, hidden_dim))
        self.register_buffer('stored_values', torch.zeros(0, hidden_dim))
        
    def store(self, patterns: torch.Tensor, values: Optional[torch.Tensor] = None):
        """
        Store patterns in the Hopfield network.
        
        Args:
            patterns: Tensor of patterns to store [num_patterns, hidden_dim]
            values: Optional tensor of associated values [num_patterns, hidden_dim]
        """
        # Project patterns to key space
        keys = self.key_projection(patterns)
        
        # If values are not provided, use patterns as values
        if values is None:
            values = patterns
            
        # Project values
        values = self.value_projection(values)
        
        # Store patterns and values
        if self.stored_keys.size(0) == 0:
            self.stored_keys = keys
            self.stored_values = values
        else:
            self.stored_keys = torch.cat([self.stored_keys, keys], dim=0)
            self.stored_values = torch.cat([self.stored_values, values], dim=0)
    
    def retrieve(self, query: torch.Tensor, num_iterations: int = 1) -> torch.Tensor:
        """
        Retrieve patterns from the Hopfield network.
        
        Args:
            query: Query tensor [batch_size, hidden_dim]
            num_iterations: Number of retrieval iterations
            
        Returns:
            Retrieved patterns [batch_size, hidden_dim]
        """
        # Project query
        query = self.query_projection(query)
        
        # Initialize state with query
        state = query
        
        # Perform iterative retrieval
        for _ in range(num_iterations):
            # Compute attention weights
            similarities = torch.matmul(state, self.stored_keys.t()) * self.beta
            attention_weights = F.softmax(similarities, dim=-1)
            
            # Update state
            state = torch.matmul(attention_weights, self.stored_values)
        
        return state
    
    def forward(self, query: torch.Tensor, store_query: bool = False) -> torch.Tensor:
        """
        Forward pass: retrieve patterns and optionally store the query.
        
        Args:
            query: Query tensor [batch_size, hidden_dim]
            store_query: Whether to store the query in memory
            
        Returns:
            Retrieved patterns [batch_size, hidden_dim]
        """
        # Store query if requested
        if store_query:
            self.store(query)
            
        # Retrieve patterns
        return self.retrieve(query)

class MemoryConsolidator:
    """
    Handles memory consolidation during "sleep" phases.
    Compresses, organizes, and prunes memories for efficient storage and retrieval.
    """
    def __init__(self, importance_threshold: float = 0.3, 
                 compression_rate: float = 0.5,
                 max_memories: int = 10000):
        self.importance_threshold = importance_threshold
        self.compression_rate = compression_rate
        self.max_memories = max_memories
        self.consolidation_count = 0
        
    def consolidate_memories(self, episodic_memories: Dict[str, Any], 
                            semantic_memories: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Consolidate episodic and semantic memories.
        
        Args:
            episodic_memories: Dictionary of episodic memories
            semantic_memories: Dictionary of semantic memories
            
        Returns:
            Tuple of consolidated episodic and semantic memories
        """
        # Increment consolidation count
        self.consolidation_count += 1
        
        # Calculate memory importance
        episodic_importance = self._calculate_importance(episodic_memories)
        semantic_importance = self._calculate_importance(semantic_memories)
        
        # Prune low-importance memories
        pruned_episodic = self._prune_memories(episodic_memories, episodic_importance)
        pruned_semantic = self._prune_memories(semantic_memories, semantic_importance)
        
        # Compress memories
        compressed_episodic = self._compress_memories(pruned_episodic)
        compressed_semantic = self._compress_memories(pruned_semantic)
        
        # Organize memories
        organized_episodic = self._organize_memories(compressed_episodic)
        organized_semantic = self._organize_memories(compressed_semantic)
        
        # Transfer episodic to semantic if needed
        if self.consolidation_count % 5 == 0:  # Every 5 consolidations
            organized_episodic, organized_semantic = self._transfer_episodic_to_semantic(
                organized_episodic, organized_semantic
            )
        
        return organized_episodic, organized_semantic
    
    def _calculate_importance(self, memories: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate importance scores for memories.
        
        Args:
            memories: Dictionary of memories
            
        Returns:
            Dictionary mapping memory keys to importance scores
        """
        importance = {}
        
        for key, memory in memories.items():
            # Calculate importance based on access frequency, recency, and emotional salience
            access_count = memory.get('access_count', 0)
            last_access = memory.get('last_access', 0)
            emotional_salience = memory.get('emotional_salience', 0.5)
            
            # Recency factor (higher for more recent memories)
            recency_factor = 1.0 / (1.0 + max(0, time.time() - last_access))
            
            # Calculate importance score
            importance[key] = (0.4 * access_count + 0.3 * recency_factor + 0.3 * emotional_salience)
        
        return importance
    
    def _prune_memories(self, memories: Dict[str, Any], 
                       importance: Dict[str, float]) -> Dict[str, Any]:
        """
        Prune low-importance memories.
        
        Args:
            memories: Dictionary of memories
            importance: Dictionary mapping memory keys to importance scores
            
        Returns:
            Dictionary of pruned memories
        """
        # Sort memories by importance
        sorted_keys = sorted(importance.keys(), key=lambda k: importance[k], reverse=True)
        
        # Keep only important memories and within max limit
        kept_keys = [k for k in sorted_keys if importance[k] >= self.importance_threshold]
        kept_keys = kept_keys[:self.max_memories]
        
        # Create pruned memories dictionary
        pruned_memories = {k: memories[k] for k in kept_keys}
        
        return pruned_memories
    
    def _compress_memories(self, memories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress memories to save space.
        
        Args:
            memories: Dictionary of memories
            
        Returns:
            Dictionary of compressed memories
        """
        compressed_memories = {}
        
        for key, memory in memories.items():
            # Create a compressed copy of the memory
            compressed_memory = memory.copy()
            
            # Compress content if it's a string and longer than 100 characters
            if isinstance(memory.get('content'), str) and len(memory['content']) > 100:
                content = memory['content']
                compressed_content = content[:int(len(content) * self.compression_rate)]
                compressed_memory['content'] = compressed_content
                compressed_memory['compressed'] = True
                compressed_memory['original_length'] = len(content)
            
            compressed_memories[key] = compressed_memory
        
        return compressed_memories
    
    def _organize_memories(self, memories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize memories by categories or themes.
        
        Args:
            memories: Dictionary of memories
            
        Returns:
            Dictionary of organized memories
        """
        # This is a placeholder for more sophisticated organization
        # In a real implementation, this would cluster memories by topic, time, etc.
        organized_memories = memories.copy()
        
        # Add organization metadata
        for key, memory in organized_memories.items():
            if 'categories' not in memory:
                memory['categories'] = []
                
            # Add consolidation timestamp
            memory['last_consolidated'] = time.time()
        
        return organized_memories
    
    def _transfer_episodic_to_semantic(self, episodic: Dict[str, Any], 
                                      semantic: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transfer episodic memories to semantic memory if they represent general knowledge.
        
        Args:
            episodic: Dictionary of episodic memories
            semantic: Dictionary of semantic memories
            
        Returns:
            Tuple of updated episodic and semantic memories
        """
        # Identify episodic memories that should be transferred to semantic memory
        transfer_keys = []
        
        for key, memory in episodic.items():
            # Check if memory has been accessed multiple times and is not personal
            if (memory.get('access_count', 0) > 3 and 
                not memory.get('is_personal', False) and
                memory.get('is_factual', False)):
                transfer_keys.append(key)
        
        # Transfer memories
        for key in transfer_keys:
            # Create semantic version of the memory
            semantic_memory = episodic[key].copy()
            semantic_memory['source'] = 'episodic_transfer'
            semantic_memory['transfer_time'] = time.time()
            
            # Add to semantic memory with a new key
            semantic_key = f"semantic_{key}"
            semantic[semantic_key] = semantic_memory
            
            # Mark episodic memory as transferred
            episodic[key]['transferred_to_semantic'] = True
        
        return episodic, semantic

class MemoryInterferenceHandler:
    """
    Handles interference between conflicting memories.
    Resolves contradictions and manages memory integration.
    """
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.contradiction_log = []
        
    def check_contradiction(self, memory1: Dict[str, Any], 
                           memory2: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if two memories contradict each other.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Tuple of (is_contradiction, contradiction_score)
        """
        # This is a simplified implementation
        # In a real system, this would use NLP or other techniques to detect contradictions
        
        # Check if memories have the same subject but different facts
        if (memory1.get('subject') == memory2.get('subject') and
            memory1.get('fact') != memory2.get('fact')):
            
            # Calculate confidence for each memory
            confidence1 = memory1.get('confidence', 0.5)
            confidence2 = memory2.get('confidence', 0.5)
            
            # Calculate contradiction score
            contradiction_score = abs(confidence1 - confidence2)
            
            return True, contradiction_score
        
        return False, 0.0
    
    def resolve_contradiction(self, memory1: Dict[str, Any], 
                             memory2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve contradiction between two memories.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Resolved memory
        """
        # Check confidence levels
        confidence1 = memory1.get('confidence', 0.5)
        confidence2 = memory2.get('confidence', 0.5)
        
        # Log the contradiction
        self.contradiction_log.append({
            'time': time.time(),
            'memory1': memory1,
            'memory2': memory2,
            'confidence1': confidence1,
            'confidence2': confidence2
        })
        
        # If one memory has significantly higher confidence, prefer it
        if abs(confidence1 - confidence2) > self.confidence_threshold:
            if confidence1 > confidence2:
                resolved = memory1.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory2.get('id', 'unknown')
                return resolved
            else:
                resolved = memory2.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory1.get('id', 'unknown')
                return resolved
        
        # If recency matters, prefer the more recent memory
        if memory1.get('recency_matters', False) or memory2.get('recency_matters', False):
            timestamp1 = memory1.get('timestamp', 0)
            timestamp2 = memory2.get('timestamp', 0)
            
            if timestamp1 > timestamp2:
                resolved = memory1.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory2.get('id', 'unknown')
                return resolved
            else:
                resolved = memory2.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory1.get('id', 'unknown')
                return resolved
        
        # If we can't determine which is correct, create a merged memory with uncertainty
        merged = {
            'subject': memory1.get('subject'),
            'fact': f"Uncertain: {memory1.get('fact')} OR {memory2.get('fact')}",
            'confidence': min(confidence1, confidence2),
            'is_uncertain': True,
            'contradiction_sources': [memory1.get('id', 'unknown'), memory2.get('id', 'unknown')],
            'timestamp': max(memory1.get('timestamp', 0), memory2.get('timestamp', 0))
        }
        
        return merged
    
    def integrate_new_memory(self, new_memory: Dict[str, Any], 
                            existing_memories: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Integrate a new memory into existing memories, handling contradictions.
        
        Args:
            new_memory: New memory to integrate
            existing_memories: Dictionary of existing memories
            
        Returns:
            Updated memories dictionary
        """
        # Check for contradictions with existing memories
        contradictions = []
        
        for key, memory in existing_memories.items():
            is_contradiction, score = self.check_contradiction(new_memory, memory)
            if is_contradiction:
                contradictions.append((key, memory, score))
        
        # If no contradictions, simply add the new memory
        if not contradictions:
            memory_id = new_memory.get('id', f"memory_{len(existing_memories)}")
            existing_memories[memory_id] = new_memory
            return existing_memories
        
        # Handle contradictions
        updated_memories = existing_memories.copy()
        
        # Sort contradictions by score (highest first)
        contradictions.sort(key=lambda x: x[2], reverse=True)
        
        # Resolve the most significant contradiction
        key, memory, _ = contradictions[0]
        resolved_memory = self.resolve_contradiction(new_memory, memory)
        
        # Update or add the resolved memory
        if resolved_memory.get('id', None):
            updated_memories[resolved_memory['id']] = resolved_memory
        else:
            memory_id = f"resolved_{len(existing_memories)}"
            resolved_memory['id'] = memory_id
            updated_memories[memory_id] = resolved_memory
        
        # Mark or remove the contradicted memory
        if resolved_memory != memory:
            if self.confidence_threshold > 0.9:
                # High threshold - remove contradicted memory
                del updated_memories[key]
            else:
                # Lower threshold - mark as contradicted
                updated_memories[key]['contradicted'] = True
                updated_memories[key]['contradicted_by'] = resolved_memory.get('id')
        
        return updated_memories


# ===========================
# Memory Manager
# ===========================

class Memory:
    def __init__(self, tokenizer, working_memory_capacity: int = 5, ltm_capacity: int = 10000000000000):
        self.tokenizer = tokenizer
        self.working_memory_capacity = working_memory_capacity
        self.context = {}
        self.ltm_capacity = ltm_capacity
        self.working_memory = []  # Short-term working memory as a list
        self.semantic_memory = {}  # Store generalized knowledge
        self.ltm = ltm  # Long-term memory list
        self.ltm_summary = {}  # Summary or knowledge graph of LTM
        self.episodic_memory_path = 'C:/Users/jonny/Documents/PATH/ONI/ltm/episodes/'
        self.semantic_memory_path = os.path.join('C:/Users/jonny/Documents/PATH/ONI/ltm_path/', 'semantic_memory.json')
        self.ltm_summary_path = os.path.join('C:/Users/jonny/Documents/PATH/ONI/ltm_path/', "ltm_data.json")
        self.load_long_term_memory()
        self.episodic_embeddings = {}  # To store episodic embeddings
        self.semantic_embeddings = {}  # To store semantic embeddings
        self.episodic_layer = EpisodicEmbeddingLayer(input_dim=8192, output_dim=8192)
        self.semantic_layer = SemanticMemoryLayer(input_dim=1024, output_dim=1024)
        self.episodic_layer.to(device)  # Adjust device as needed
        self.semantic_layer.to(device)  # Adjust device as needed
        
        # New memory components
        self.episodic_buffer = EpisodicBuffer(hidden_dim=896, buffer_size=working_memory_capacity)
        self.continuous_hopfield = ModernContinuousHopfieldNetwork(hidden_dim=896)
        self.memory_consolidator = MemoryConsolidator()
        self.interference_handler = MemoryInterferenceHandler()
        
        # Sleep state tracking
        self.is_sleeping = False
        self.last_sleep_time = time.time()
        self.sleep_interval = 3600  # Default: consolidate every hour
   
    def cleanup(self):
        """Release any held resources."""
        self.memory.data.zero_()
        torch.cuda.empty_cache()
        
    def update_context(self, key: str, value: str):
        self.context[key] = value

    def get_context(self) -> Dict[str, str]:
        return self.context

    def handle_media(self, data: str) -> Optional[Dict[str, str]]:
        """Handle different types of media files for episodic memory."""
        media_extensions = {
            '.mov': 'video',
            '.mp4': 'video',
            '.avi': 'video',
            '.wav': 'audio',
            '.mp3': 'audio',
            '.txt': 'plaintext',
            '.pdf': 'PDF',
            '.doc': 'Document File',
            '.odt': 'Open Document TXT',
            '.py': 'python file',
            '.html': 'website',
            '.js': 'javascript',
            '.css': 'styles'
        }
        file_extension = os.path.splitext(data)[1]
        media_type = media_extensions.get(file_extension, 'unknown')

        if media_type in ['video', 'audio']:
            return {'type': media_type, 'path': data}
        return None

    def load_long_term_memory(self):
        """Load semantic memory and LTM summary from files."""
        self.semantic_memory = self._load_json(self.semantic_memory_path)
        self.ltm_summary = self._load_json(self.ltm_summary_path)

    @staticmethod
    def _load_json(file_path: str) -> Dict:
        """Helper function to load JSON data from a file."""
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return {}
        return {}

    def update_episodic_memory(self, data: bytes, data_type: str, key: str):
        """Update episodic memory by storing media files."""
        file_path = os.path.join(self.episodic_memory_path, f"{key}_{data_type}")
        os.makedirs(self.episodic_memory_path, exist_ok=True)

        # Store the media file
        with open(file_path, 'wb') as file:
            file.write(data)

        # Update LTM summary
        self.ltm_summary[key] = {"data_type": data_type, "path": file_path}

        # Convert the data into a tensor for embedding (assuming the data is in a suitable format for embedding)
        data_tensor = torch.tensor(list(data), dtype=torch.float32).unsqueeze(0)  # Example tensor conversion

        # Update the episodic embedding layer
        embedding = self.episodic_layer(data_tensor)
        self.episodic_embeddings[key] = embedding

    def retrieve_from_episodic(self, key: str) -> Optional[str]:
        """Retrieve data from episodic memory."""
        return self.ltm_summary.get(key, {}).get('path', None)

    def update_semantic_memory(self, data: str, data_type: str, key: str):
        """Update semantic memory by storing processed semantic information and updating embeddings."""
        file_path = os.path.join(self.semantic_memory_path, f"{key}_{data_type}")

        # Check if the semantic_memory_path is a directory or a file
        if not os.path.isdir(self.semantic_memory_path):
            # If it's a file, raise an error or handle accordingly
            if os.path.exists(self.semantic_memory_path):
                raise FileExistsError(f"A file exists at the path: {self.semantic_memory_path}. Unable to create directory.")
            else:
                os.makedirs(self.semantic_memory_path, exist_ok=True)

        # Store the semantic data file
        with open(file_path, 'w') as file:
            file.write(data)

        # Update LTM summary
        self.ltm_summary[key] = {'data_type': data_type, 'path': file_path}

        # Convert the data into a tensor for embedding
        data_tensor = torch.tensor(list(data), dtype=torch.float32).unsqueeze(0)  # Example tensor conversion

        # Update the semantic embedding layer
        embedding = self.semantic_layer(data_tensor)
        self.semantic_embeddings[key] = embedding

    def lookup_token(self, token: str) -> int:
        """Lookup the index of a token in the semantic memory."""
        return self.semantic_memory.get(token, -1)

    def meditate(self):
        """Compress and refine semantic and working memory."""
        unique_data = set(list(self.semantic_memory.values()) + self.working_memory)
        self.semantic_memory = {token: idx for idx, token in enumerate(unique_data)}

    def sleep(self):
        """Main sleep function to stop all processes and consolidate memories."""
        print("AI is going to sleep...")
        self.is_sleeping = True
        
        # Consolidate memories
        self.meditate()
        
        # Perform memory consolidation
        episodic_memories = self._get_episodic_memories()
        semantic_memories = self._get_semantic_memories()
        
        consolidated_episodic, consolidated_semantic = self.memory_consolidator.consolidate_memories(
            episodic_memories, semantic_memories
        )
        
        # Update memories with consolidated versions
        self._update_consolidated_memories(consolidated_episodic, consolidated_semantic)
        
        # Save to disk
        self.save_long_term_memory()
        
        # Update sleep state
        self.is_sleeping = False
        self.last_sleep_time = time.time()
        
        print("AI has woken up with refreshed memories.")

    def save_long_term_memory(self):
        """Save semantic memory and LTM summary."""
        self._save_json(self.semantic_memory_path, self.semantic_memory)
        self._save_json(self.ltm_summary_path, self.ltm_summary)
        torch.cuda.empty_cache()

    @staticmethod
    def _save_json(file_path: str, data: Dict):
        """Helper function to save JSON data to a file."""
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def handle_pdf(self, pdf_path: str):
        """Extract text from PDF and add to semantic memory."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    self.update_semantic_memory(text, data_type='PDF', key=f'pdf_page_{page_num}')
            print(f"PDF '{os.path.basename(pdf_path)}' added to semantic memory.")
        except Exception as e:
            print(f"Error processing PDF: {e}")

    def _get_mp4_data(self, directory: str) -> Dict[str, bytes]:
        mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]
        mp4_data = {}

        for file in mp4_files:
            file_path = os.path.join(directory, file)
            with open(file_path, 'rb') as f:
                mp4_data[file] = f.read()

        return mp4_data

    def update_memory(self, stm_data: str, episodic_data: bytes = None, episodic_key: str = None, semantic_data: str = None):
        """Update all memory stores with new data."""
        # Update working memory
        self.working_memory.append(stm_data)
        if len(self.working_memory) > self.working_memory_capacity:
            self.working_memory.pop(0)  # Remove oldest entry to maintain capacity
            
        # Update episodic buffer with new data
        if episodic_data is not None or semantic_data is not None:
            # Convert data to tensors
            episodic_tensor = None
            semantic_tensor = None
            
            if episodic_data is not None:
                episodic_tensor = torch.tensor(list(episodic_data), dtype=torch.float32).unsqueeze(0)
                
            if semantic_data is not None:
                semantic_tensor = torch.tensor(list(semantic_data), dtype=torch.float32).unsqueeze(0)
            
            # Create query from current context
            context_str = " ".join(self.context.values())
            context_tensor = torch.tensor(list(context_str), dtype=torch.float32).unsqueeze(0)
            
            # Update episodic buffer
            self.episodic_buffer(context_tensor, episodic_tensor, semantic_tensor)
            
            # Store in continuous Hopfield network for associative retrieval
            if episodic_tensor is not None:
                self.continuous_hopfield.store(episodic_tensor)
            if semantic_tensor is not None:
                self.continuous_hopfield.store(semantic_tensor)
        
        # Update traditional memory stores
        if episodic_data and episodic_key:
            self.update_episodic_memory(episodic_data, data_type='audio', key=episodic_key)
        if semantic_data and episodic_key:
            self.update_semantic_memory(semantic_data, data_type='text', key=episodic_key)
            
        # Check if it's time to sleep
        if time.time() - self.last_sleep_time > self.sleep_interval:
            self.sleep()
        else:
            # Just save without full consolidation
            self.save_long_term_memory()

    def categorize_and_store(self, db_path: str = 'personality.db'):
        """Categorize memory items and store them in a SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS personalities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                race TEXT,
                origin TEXT,
                age INTEGER,
                type TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                personality_id INTEGER,
                timestamp TEXT,
                input_text TEXT,
                response_text TEXT,
                FOREIGN KEY (personality_id) REFERENCES personalities(id)
            )
        """)
        conn.commit()
        conn.close()

    def get_experiences(self) -> Dict[str, bytes]:
        """Retrieve MP4 data from episodes."""
        return self._get_mp4_data(self.episodic_memory_path)

    def integrate_insights(self, insights: Dict, model):
        """Integrate insights into the system's decision-making process."""
        # Placeholder for integration logic
        pass

    def reflect_on_experience(self, experiences: Dict[str, bytes], model):
        """
        Analyze experiences and integrate insights.

        Args:
            experiences (dict): Dictionary of experiences.
            model: The AGI model for integration.
        """
        # Analyze experiences (e.g., identify patterns, successes, areas for improvement)
        insights = self.analyze_experiences(experiences)
        self.integrate_insights(insights, model)

    def analyze_experiences(self, experiences: Dict[str, bytes]) -> Dict:
        """
        Analyze experiences to extract insights.

        Args:
            experiences (dict): Dictionary of experiences.

        Returns:
            dict: Extracted insights.
        """
        # Placeholder for analysis logic
        return {"patterns": [], "improvements": []}
        
    def _get_episodic_memories(self) -> Dict[str, Any]:
        """Get all episodic memories for consolidation"""
        # This is a simplified implementation
        # In a real system, this would retrieve all episodic memories from storage
        episodic_memories = {}
        
        # Convert episodic embeddings to memories
        for key, embedding in self.episodic_embeddings.items():
            episodic_memories[key] = {
                'embedding': embedding.detach().cpu().numpy().tolist(),
                'path': self.ltm_summary.get(key, {}).get('path', ''),
                'data_type': self.ltm_summary.get(key, {}).get('data_type', ''),
                'access_count': 1,  # Default value
                'last_access': time.time(),
                'timestamp': time.time()
            }
        
        return episodic_memories
    
    def _get_semantic_memories(self) -> Dict[str, Any]:
        """Get all semantic memories for consolidation"""
        # This is a simplified implementation
        # In a real system, this would retrieve all semantic memories from storage
        semantic_memories = {}
        
        # Convert semantic embeddings to memories
        for key, embedding in self.semantic_embeddings.items():
            semantic_memories[key] = {
                'embedding': embedding.detach().cpu().numpy().tolist(),
                'content': key,  # Use key as content for simplicity
                'access_count': 1,  # Default value
                'last_access': time.time(),
                'timestamp': time.time()
            }
        
        # Add semantic memory dictionary items
        for token, idx in self.semantic_memory.items():
            if isinstance(token, str):
                semantic_memories[f"token_{idx}"] = {
                    'content': token,
                    'index': idx,
                    'access_count': 1,  # Default value
                    'last_access': time.time(),
                    'timestamp': time.time()
                }
        
        return semantic_memories
    
    def _update_consolidated_memories(self, episodic: Dict[str, Any], semantic: Dict[str, Any]):
        """Update memory stores with consolidated memories"""
        # Update episodic embeddings
        for key, memory in episodic.items():
            if 'embedding' in memory:
                embedding = torch.tensor(memory['embedding'])
                self.episodic_embeddings[key] = embedding
        
        # Update semantic embeddings
        for key, memory in semantic.items():
            if 'embedding' in memory:
                embedding = torch.tensor(memory['embedding'])
                self.semantic_embeddings[key] = embedding
            elif 'content' in memory and 'index' in memory:
                self.semantic_memory[memory['content']] = memory['index']
    
    def retrieve_associative_memory(self, query: str) -> Dict[str, Any]:
        """
        Retrieve memories associatively using the continuous Hopfield network.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with retrieved memories
        """
        # Convert query to tensor
        query_tensor = torch.tensor(list(query), dtype=torch.float32).unsqueeze(0)
        
        # Retrieve from Hopfield network
        retrieved = self.continuous_hopfield.retrieve(query_tensor)
        
        # Convert retrieved tensor to string (simplified)
        retrieved_data = retrieved.detach().cpu().numpy().tolist()[0]
        
        # Find closest matching memories
        closest_memories = self._find_closest_memories(retrieved_data)
        
        return {
            'query': query,
            'retrieved_memories': closest_memories
        }
    
    def _find_closest_memories(self, retrieved_data: List[float]) -> List[Dict[str, Any]]:
        """
        Find closest matching memories to the retrieved data.
        
        Args:
            retrieved_data: Retrieved data vector
            
        Returns:
            List of closest memories
        """
        retrieved_tensor = torch.tensor(retrieved_data)
        closest_memories = []
        
        # Check episodic embeddings
        for key, embedding in self.episodic_embeddings.items():
            similarity = F.cosine_similarity(
                retrieved_tensor.unsqueeze(0),
                embedding.flatten().unsqueeze(0)
            ).item()
            
            if similarity > 0.7:  # Threshold for similarity
                closest_memories.append({
                    'key': key,
                    'type': 'episodic',
                    'similarity': similarity,
                    'path': self.ltm_summary.get(key, {}).get('path', '')
                })
        
        # Check semantic embeddings
        for key, embedding in self.semantic_embeddings.items():
            similarity = F.cosine_similarity(
                retrieved_tensor.unsqueeze(0),
                embedding.flatten().unsqueeze(0)
            ).item()
            
            if similarity > 0.7:  # Threshold for similarity
                closest_memories.append({
                    'key': key,
                    'type': 'semantic',
                    'similarity': similarity,
                    'content': key  # Use key as content for simplicity
                })
        
        # Sort by similarity (highest first)
        closest_memories.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Return top 5 memories
        return closest_memories[:5]
    
    def check_sleep_needed(self) -> bool:
        """
        Check if memory consolidation (sleep) is needed.
        
        Returns:
            bool: True if sleep is needed, False otherwise
        """
        # Check if it's been long enough since last sleep
        time_since_sleep = time.time() - self.last_sleep_time
        if time_since_sleep > self.sleep_interval:
            return True
            
        # Check if working memory is getting full
        if len(self.working_memory) >= self.working_memory_capacity * 0.9:
            return True
            
        # Check if there are many new memories since last consolidation
        new_memory_count = len(self.episodic_embeddings) + len(self.semantic_embeddings)
        if new_memory_count > 100:  # Arbitrary threshold
            return True
            
        return False
