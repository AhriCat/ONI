import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import numpy as np
from dataclasses import dataclass

@dataclass
class HyperEdge:
    """Represents a hyperedge connecting multiple nodes with weights and metadata"""
    nodes: Set[str]
    weight: float
    relation_type: str
    temporal_strength: float = 1.0
    semantic_distance: float = 0.0
    activation_count: int = 0

class SemanticMemoryLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, max_hyperedges: int = 10000):
        super(HypergraphicSemanticMemoryLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_hyperedges = max_hyperedges
        
        # Core encoders
        self.text_encoder = nn.Linear(input_dim, output_dim)
        self.hypergraph_encoder = nn.Linear(output_dim * 3, output_dim)  # For hyperedge embeddings
        
        # Hypergraph attention mechanism
        self.hypergraph_attention = nn.MultiheadAttention(
            embed_dim=output_dim, 
            num_heads=8, 
            dropout=0.1,
            batch_first=True
        )
        
        # Relation type embeddings
        self.relation_embeddings = nn.Embedding(20, output_dim)  # Support 20 relation types
        
        # Hypergraph structure
        self.hyperedges: Dict[str, HyperEdge] = {}
        self.node_embeddings: Dict[str, torch.Tensor] = {}
        self.node_to_hyperedges: Dict[str, Set[str]] = defaultdict(set)
        
        # Dynamic memory components
        self.working_hypergraph = {}  # Temporary hyperedges for current context
        self.consolidation_threshold = 3  # Hyperedges need 3+ activations to persist
        
        # Shared connection to episodic layer
        self.episodic_embedding_layer = None
        
        # Hypergraph convolution layers for message passing
        self.hgcn_layers = nn.ModuleList([
            HypergraphConvLayer(output_dim, output_dim) for _ in range(3)
        ])
        
    def forward(self, text_input: torch.Tensor, media_reference: Optional[torch.Tensor] = None, 
                context_nodes: Optional[List[str]] = None):
        batch_size = text_input.size(0)
        
        # Encode text
        text_embedding = self.text_encoder(text_input)
        
        # Create or retrieve hypergraph context
        if context_nodes:
            hypergraph_context = self._build_hypergraph_context(context_nodes, batch_size)
            
            # Apply hypergraph attention
            attended_embedding, attention_weights = self.hypergraph_attention(
                text_embedding.unsqueeze(1),
                hypergraph_context,
                hypergraph_context
            )
            text_embedding = attended_embedding.squeeze(1)
        
        # Handle media reference through hypergraph connections
        if media_reference is not None and self.episodic_embedding_layer is not None:
            media_embedding = self.episodic_embedding_layer(media_reference, media_type='image')
            
            # Create multimodal hyperedge
            combined_embedding = self._create_multimodal_hyperedge(
                text_embedding, media_embedding, batch_size
            )
            
            # Apply hypergraph convolution
            for hgcn_layer in self.hgcn_layers:
                combined_embedding = hgcn_layer(combined_embedding, self._get_hypergraph_adjacency())
            
            return combined_embedding
            
        return text_embedding
    
    def _build_hypergraph_context(self, context_nodes: List[str], batch_size: int) -> torch.Tensor:
        """Build hypergraph context from related nodes and hyperedges"""
        context_embeddings = []
        
        for node in context_nodes:
            if node in self.node_embeddings:
                context_embeddings.append(self.node_embeddings[node])
            
            # Add hyperedge embeddings that contain this node
            for hyperedge_id in self.node_to_hyperedges[node]:
                if hyperedge_id in self.hyperedges:
                    hyperedge_emb = self._compute_hyperedge_embedding(hyperedge_id)
                    context_embeddings.append(hyperedge_emb)
        
        if not context_embeddings:
            return torch.zeros(batch_size, 1, self.output_dim)
        
        # Stack and return context
        context_tensor = torch.stack(context_embeddings)
        return context_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    
    def _create_multimodal_hyperedge(self, text_emb: torch.Tensor, media_emb: torch.Tensor, 
                                   batch_size: int) -> torch.Tensor:
        """Create hyperedge connecting text and media modalities"""
        # Compute relation embedding
        relation_emb = self.relation_embeddings(torch.tensor(0))  # "multimodal" relation type
        
        # Combine embeddings through hypergraph encoder
        combined = torch.cat([
            text_emb, 
            media_emb, 
            relation_emb.expand(batch_size, -1)
        ], dim=-1)
        
        return self.hypergraph_encoder(combined)
    
    def _compute_hyperedge_embedding(self, hyperedge_id: str) -> torch.Tensor:
        """Compute embedding for a hyperedge based on its constituent nodes"""
        hyperedge = self.hyperedges[hyperedge_id]
        
        node_embeddings = []
        for node in hyperedge.nodes:
            if node in self.node_embeddings:
                node_embeddings.append(self.node_embeddings[node])
        
        if not node_embeddings:
            return torch.zeros(self.output_dim)
        
        # Aggregate node embeddings with attention to hyperedge properties
        stacked_embeddings = torch.stack(node_embeddings)
        weights = torch.softmax(torch.ones(len(node_embeddings)) * hyperedge.weight, dim=0)
        
        return torch.sum(stacked_embeddings * weights.unsqueeze(-1), dim=0)
    
    def _get_hypergraph_adjacency(self) -> torch.Tensor:
        """Get adjacency matrix for hypergraph convolution"""
        # Simplified adjacency - in practice, this would be more sophisticated
        num_nodes = len(self.node_embeddings)
        adjacency = torch.eye(num_nodes)
        return adjacency
    
    def add_hyperedge(self, nodes: Set[str], relation_type: str, weight: float = 1.0):
        """Add new hyperedge to the hypergraph"""
        hyperedge_id = f"{relation_type}_{hash(frozenset(nodes))}"
        
        if hyperedge_id in self.hyperedges:
            # Strengthen existing hyperedge
            self.hyperedges[hyperedge_id].weight += 0.1
            self.hyperedges[hyperedge_id].activation_count += 1
        else:
            # Create new hyperedge
            hyperedge = HyperEdge(
                nodes=nodes,
                weight=weight,
                relation_type=relation_type
            )
            
            # Add to working memory first
            self.working_hypergraph[hyperedge_id] = hyperedge
            
            # Update node-to-hyperedge mapping
            for node in nodes:
                self.node_to_hyperedges[node].add(hyperedge_id)
    
    def consolidate_working_hypergraph(self):
        """Move frequently activated hyperedges from working to long-term memory"""
        for hyperedge_id, hyperedge in list(self.working_hypergraph.items()):
            if hyperedge.activation_count >= self.consolidation_threshold:
                self.hyperedges[hyperedge_id] = hyperedge
                del self.working_hypergraph[hyperedge_id]
                
                # Prune if we exceed maximum hyperedges
                if len(self.hyperedges) > self.max_hyperedges:
                    self._prune_weak_hyperedges()
    
    def _prune_weak_hyperedges(self):
        """Remove weakest hyperedges to maintain memory limits"""
        sorted_edges = sorted(
            self.hyperedges.items(), 
            key=lambda x: x[1].weight * x[1].temporal_strength
        )
        
        # Remove bottom 10%
        num_to_remove = len(sorted_edges) // 10
        for hyperedge_id, _ in sorted_edges[:num_to_remove]:
            self._remove_hyperedge(hyperedge_id)
    
    def _remove_hyperedge(self, hyperedge_id: str):
        """Remove hyperedge and update mappings"""
        if hyperedge_id in self.hyperedges:
            hyperedge = self.hyperedges[hyperedge_id]
            
            # Remove from node mappings
            for node in hyperedge.nodes:
                self.node_to_hyperedges[node].discard(hyperedge_id)
            
            del self.hyperedges[hyperedge_id]


class HypergraphConvLayer(nn.Module):
    """Hypergraph convolution layer for message passing"""
    def __init__(self, input_dim: int, output_dim: int):
        super(HypergraphConvLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.node_transform = nn.Linear(input_dim, output_dim)
        self.hyperedge_transform = nn.Linear(input_dim, output_dim)
        self.message_transform = nn.Linear(output_dim * 2, output_dim)
        
    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Apply hypergraph convolution"""
        # Transform node features
        transformed_nodes = self.node_transform(node_features)
        
        # Message passing (simplified)
        messages = torch.matmul(adjacency, transformed_nodes)
        
        # Combine original features with messages
        combined = torch.cat([transformed_nodes, messages], dim=-1)
        output = self.message_transform(combined)
        
        return F.relu(output)


class TextPatternFinder:
    """Enhanced pattern finder that creates hypergraphic relationships"""
    
    def __init__(self, tokenizer, min_pattern_length: int = 3, max_pattern_length: int = 10, 
                 min_occurrences: int = 2, semantic_threshold: float = 0.8):
        self.tokenizer = tokenizer
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_occurrences = min_occurrences
        self.semantic_threshold = semantic_threshold
        
        # Hypergraphic structures
        self.pattern_hypergraph: Dict[str, HyperEdge] = {}
        self.semantic_clusters: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_patterns: List[Tuple[str, float]] = []
        
        # Enhanced Hopfield network for hypergraphic associations
        self.hypergraphic_hopfield = None
        
    def find_hypergraphic_patterns(self, corpus: List[str], ltm: List[str]) -> Dict[str, HyperEdge]:
        """Find patterns and create hypergraphic relationships"""
        
        # Find basic patterns
        corpus_patterns = self._find_patterns_in_text(corpus)
        ltm_patterns = self._find_patterns_in_text(ltm)
        
        # Create hyperedges for pattern relationships
        self._create_pattern_hyperedges(corpus_patterns, ltm_patterns)
        
        # Build semantic clusters
        self._build_semantic_clusters(corpus_patterns, ltm_patterns)
        
        # Create temporal hyperedges
        self._create_temporal_hyperedges(corpus, ltm)
        
        return self.pattern_hypergraph
    
    def _find_patterns_in_text(self, text: List[str]) -> Dict[str, List[int]]:
        """Enhanced pattern finding with semantic awareness"""
        patterns = defaultdict(list)
        tokens = self.tokenizer.tokenize(' '.join(text))
        
        for i in range(len(tokens)):
            for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                if i + length > len(tokens):
                    break
                    
                pattern = ' '.join(tokens[i:i+length])
                patterns[pattern].append(i)
        
        # Filter by occurrence threshold
        return {p: pos for p, pos in patterns.items() if len(pos) >= self.min_occurrences}
    
    def _create_pattern_hyperedges(self, corpus_patterns: Dict, ltm_patterns: Dict):
        """Create hyperedges connecting related patterns"""
        all_patterns = set(corpus_patterns.keys()) | set(ltm_patterns.keys())
        
        for pattern1 in all_patterns:
            related_patterns = set()
            
            # Find semantically related patterns
            for pattern2 in all_patterns:
                if pattern1 != pattern2:
                    similarity = self._compute_semantic_similarity(pattern1, pattern2)
                    if similarity > self.semantic_threshold:
                        related_patterns.add(pattern2)
            
            if related_patterns:
                related_patterns.add(pattern1)
                hyperedge_id = f"semantic_cluster_{hash(frozenset(related_patterns))}"
                
                self.pattern_hypergraph[hyperedge_id] = HyperEdge(
                    nodes=related_patterns,
                    weight=len(related_patterns) * 0.1,
                    relation_type="semantic_similarity"
                )
    
    def _build_semantic_clusters(self, corpus_patterns: Dict, ltm_patterns: Dict):
        """Build semantic clusters using hypergraph structure"""
        all_patterns = list(set(corpus_patterns.keys()) | set(ltm_patterns.keys()))
        
        # Create similarity matrix
        similarity_matrix = np.zeros((len(all_patterns), len(all_patterns)))
        
        for i, pattern1 in enumerate(all_patterns):
            for j, pattern2 in enumerate(all_patterns):
                if i != j:
                    similarity_matrix[i][j] = self._compute_semantic_similarity(pattern1, pattern2)
        
        # Find clusters using threshold
        for i, pattern in enumerate(all_patterns):
            cluster_key = f"cluster_{i}"
            similar_indices = np.where(similarity_matrix[i] > self.semantic_threshold)[0]
            
            for idx in similar_indices:
                self.semantic_clusters[cluster_key].add(all_patterns[idx])
    
    def _create_temporal_hyperedges(self, corpus: List[str], ltm: List[str]):
        """Create hyperedges for temporal relationships"""
        combined_text = corpus + ltm
        
        # Simple temporal relationship detection
        for i in range(len(combined_text) - 1):
            current_tokens = set(self.tokenizer.tokenize(combined_text[i]))
            next_tokens = set(self.tokenizer.tokenize(combined_text[i + 1]))
            
            # Find overlapping concepts
            overlap = current_tokens & next_tokens
            if len(overlap) >= 2:  # Sufficient overlap for temporal relationship
                hyperedge_id = f"temporal_{i}_{i+1}"
                self.pattern_hypergraph[hyperedge_id] = HyperEdge(
                    nodes=overlap,
                    weight=len(overlap) * 0.2,
                    relation_type="temporal_sequence",
                    temporal_strength=1.0 / (1 + i)  # Decay over time
                )
    
    def _compute_semantic_similarity(self, pattern1: str, pattern2: str) -> float:
        """Compute semantic similarity between patterns"""
        tokens1 = set(self.tokenizer.tokenize(pattern1))
        tokens2 = set(self.tokenizer.tokenize(pattern2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        # Jaccard similarity as baseline
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_hypergraph_neighborhood(self, query_pattern: str, max_hops: int = 2) -> Set[str]:
        """Get hypergraph neighborhood around a query pattern"""
        neighborhood = {query_pattern}
        current_patterns = {query_pattern}
        
        for hop in range(max_hops):
            next_patterns = set()
            
            for hyperedge in self.pattern_hypergraph.values():
                if any(pattern in current_patterns for pattern in hyperedge.nodes):
                    next_patterns.update(hyperedge.nodes)
            
            neighborhood.update(next_patterns)
            current_patterns = next_patterns - neighborhood
            
            if not current_patterns:
                break
        
        return neighborhood
