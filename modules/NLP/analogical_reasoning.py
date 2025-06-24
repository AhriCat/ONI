import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class StructureMapping(nn.Module):
    """
    Neural structure mapping module for analogical reasoning.
    
    Implements Gentner's structure mapping theory for finding analogies
    between a source and target domain by identifying shared relational
    structure.
    """
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        """
        Initialize structure mapping module.
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for mapping representations
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Relation encoder
        self.relation_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Mapping network
        self.mapping_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Correspondence scorer
        self.correspondence_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def encode_relations(self, entities: torch.Tensor, 
                        relations: torch.Tensor) -> torch.Tensor:
        """
        Encode relations between entities.
        
        Args:
            entities: Entity embeddings of shape [batch_size, num_entities, embedding_dim]
            relations: Relation adjacency matrix of shape [batch_size, num_entities, num_entities]
            
        Returns:
            Encoded relations of shape [batch_size, num_relations, hidden_dim]
        """
        batch_size, num_entities, _ = entities.shape
        
        # Extract relation pairs
        relation_pairs = []
        relation_indices = []
        
        for b in range(batch_size):
            for i in range(num_entities):
                for j in range(num_entities):
                    if relations[b, i, j] > 0:
                        # Concatenate entity embeddings to represent the relation
                        pair = torch.cat([entities[b, i], entities[b, j]], dim=-1)
                        relation_pairs.append(pair)
                        relation_indices.append((b, i, j))
        
        if not relation_pairs:
            # No relations found
            return torch.zeros(batch_size, 0, self.hidden_dim, device=entities.device)
        
        # Stack relation pairs
        relation_pairs = torch.stack(relation_pairs, dim=0)
        
        # Encode relations
        encoded_relations = self.relation_encoder(relation_pairs)
        
        # Reshape to [batch_size, num_relations, hidden_dim]
        result = torch.zeros(batch_size, num_entities * num_entities, self.hidden_dim, 
                           device=entities.device)
        
        for idx, (b, i, j) in enumerate(relation_indices):
            flat_idx = i * num_entities + j
            result[b, flat_idx] = encoded_relations[idx]
        
        return result
    
    def compute_mapping_score(self, source_relations: torch.Tensor, 
                             target_relations: torch.Tensor) -> torch.Tensor:
        """
        Compute mapping score between source and target relations.
        
        Args:
            source_relations: Source relation encodings [batch_size, num_relations, hidden_dim]
            target_relations: Target relation encodings [batch_size, num_relations, hidden_dim]
            
        Returns:
            Mapping scores of shape [batch_size, num_source_relations, num_target_relations]
        """
        batch_size, num_source_relations, _ = source_relations.shape
        _, num_target_relations, _ = target_relations.shape
        
        # Reshape for pairwise comparison
        source_expanded = source_relations.unsqueeze(2).expand(
            -1, -1, num_target_relations, -1
        )
        target_expanded = target_relations.unsqueeze(1).expand(
            -1, num_source_relations, -1, -1
        )
        
        # Concatenate relation pairs
        relation_pairs = torch.cat([source_expanded, target_expanded], dim=-1)
        
        # Reshape for the mapping network
        relation_pairs = relation_pairs.view(-1, self.hidden_dim * 2)
        
        # Compute mapping scores
        scores = self.mapping_network(relation_pairs).view(
            batch_size, num_source_relations, num_target_relations
        )
        
        return scores
    
    def find_correspondences(self, source_entities: torch.Tensor, 
                            target_entities: torch.Tensor) -> torch.Tensor:
        """
        Find correspondences between source and target entities.
        
        Args:
            source_entities: Source entity embeddings [batch_size, num_source, embedding_dim]
            target_entities: Target entity embeddings [batch_size, num_target, embedding_dim]
            
        Returns:
            Correspondence scores [batch_size, num_source, num_target]
        """
        batch_size, num_source, _ = source_entities.shape
        _, num_target, _ = target_entities.shape
        
        # Reshape for pairwise comparison
        source_expanded = source_entities.unsqueeze(2).expand(
            -1, -1, num_target, -1
        )
        target_expanded = target_entities.unsqueeze(1).expand(
            -1, num_source, -1, -1
        )
        
        # Concatenate entity pairs
        entity_pairs = torch.cat([source_expanded, target_expanded], dim=-1)
        
        # Reshape for the correspondence scorer
        entity_pairs = entity_pairs.view(-1, self.embedding_dim * 2)
        
        # Compute correspondence scores
        scores = self.correspondence_scorer(entity_pairs).view(
            batch_size, num_source, num_target
        )
        
        return scores
    
    def forward(self, source_entities: torch.Tensor, source_relations: torch.Tensor,
               target_entities: torch.Tensor, target_relations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform structure mapping between source and target domains.
        
        Args:
            source_entities: Source entity embeddings [batch_size, num_source, embedding_dim]
            source_relations: Source relation matrix [batch_size, num_source, num_source]
            target_entities: Target entity embeddings [batch_size, num_target, embedding_dim]
            target_relations: Target relation matrix [batch_size, num_target, num_target]
            
        Returns:
            Dictionary with mapping results
        """
        # Encode relations
        encoded_source_relations = self.encode_relations(source_entities, source_relations)
        encoded_target_relations = self.encode_relations(target_entities, target_relations)
        
        # Compute mapping scores
        mapping_scores = self.compute_mapping_score(
            encoded_source_relations, encoded_target_relations
        )
        
        # Find entity correspondences
        correspondence_scores = self.find_correspondences(
            source_entities, target_entities
        )
        
        # Normalize scores
        mapping_scores = F.softmax(mapping_scores, dim=-1)
        correspondence_scores = F.softmax(correspondence_scores, dim=-1)
        
        return {
            'mapping_scores': mapping_scores,
            'correspondence_scores': correspondence_scores
        }
    
    def apply_analogy(self, source_entities: torch.Tensor, source_relations: torch.Tensor,
                     target_entities: torch.Tensor, target_relations: torch.Tensor,
                     source_output: torch.Tensor) -> torch.Tensor:
        """
        Apply an analogy to generate a target output.
        
        Args:
            source_entities: Source entity embeddings [batch_size, num_source, embedding_dim]
            source_relations: Source relation matrix [batch_size, num_source, num_source]
            target_entities: Target entity embeddings [batch_size, num_target, embedding_dim]
            target_relations: Target relation matrix [batch_size, num_target, num_target]
            source_output: Source output [batch_size, embedding_dim]
            
        Returns:
            Target output [batch_size, embedding_dim]
        """
        # Get mapping results
        mapping_results = self.forward(
            source_entities, source_relations,
            target_entities, target_relations
        )
        
        # Get correspondence scores
        correspondence_scores = mapping_results['correspondence_scores']
        
        # Compute weighted combination of target entities based on correspondences
        batch_size, num_source, num_target = correspondence_scores.shape
        
        # Reshape source output for entity-wise comparison
        source_output_expanded = source_output.unsqueeze(1).expand(-1, num_source, -1)
        
        # Compute similarity between source output and source entities
        similarity = F.cosine_similarity(
            source_output_expanded, source_entities, dim=-1
        )
        
        # Normalize similarity
        similarity = F.softmax(similarity, dim=-1)
        
        # Use similarity to weight correspondence scores
        weighted_correspondence = similarity.unsqueeze(-1) * correspondence_scores
        
        # Sum over source entities to get target entity weights
        target_weights = weighted_correspondence.sum(dim=1)
        
        # Compute weighted combination of target entities
        target_output = torch.bmm(
            target_weights.unsqueeze(1),
            target_entities
        ).squeeze(1)
        
        return target_output

class AnalogicalReasoning(nn.Module):
    """
    Module for analogical reasoning and transfer learning.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4):
        """
        Initialize analogical reasoning module.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for analogical representations
            num_heads: Number of attention heads for relation extraction
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Relation extractor (multi-head attention)
        self.relation_extractor = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Structure mapping module
        self.structure_mapper = StructureMapping(hidden_dim, hidden_dim)
        
        # Analogy generator
        self.analogy_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Memory for storing analogies
        self.register_buffer('source_memory', torch.zeros(100, hidden_dim))
        self.register_buffer('target_memory', torch.zeros(100, hidden_dim))
        self.memory_counter = 0
    
    def extract_entities(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract entity representations from input.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            mask: Optional mask of shape [batch_size, seq_len]
            
        Returns:
            Entity embeddings of shape [batch_size, seq_len, hidden_dim]
        """
        # Encode entities
        entities = self.entity_encoder(x)
        
        return entities
    
    def extract_relations(self, entities: torch.Tensor, 
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract relation matrix from entities.
        
        Args:
            entities: Entity embeddings of shape [batch_size, seq_len, hidden_dim]
            mask: Optional mask of shape [batch_size, seq_len]
            
        Returns:
            Relation matrix of shape [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, _ = entities.shape
        
        # Transpose for attention
        entities_t = entities.transpose(0, 1)
        
        # Extract relations using self-attention
        _, attn_weights = self.relation_extractor(
            entities_t, entities_t, entities_t,
            key_padding_mask=~mask if mask is not None else None
        )
        
        # Average attention weights across heads
        relations = attn_weights.mean(dim=0)
        
        return relations
    
    def forward(self, source: torch.Tensor, target: torch.Tensor,
               source_mask: Optional[torch.Tensor] = None,
               target_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform analogical reasoning between source and target domains.
        
        Args:
            source: Source domain input of shape [batch_size, source_len, input_dim]
            target: Target domain input of shape [batch_size, target_len, input_dim]
            source_mask: Optional mask for source of shape [batch_size, source_len]
            target_mask: Optional mask for target of shape [batch_size, target_len]
            
        Returns:
            Dictionary with analogical reasoning outputs
        """
        # Extract entities
        source_entities = self.extract_entities(source)
        target_entities = self.extract_entities(target)
        
        # Extract relations
        source_relations = self.extract_relations(source_entities, source_mask)
        target_relations = self.extract_relations(target_entities, target_mask)
        
        # Perform structure mapping
        mapping_results = self.structure_mapper(
            source_entities, source_relations,
            target_entities, target_relations
        )
        
        # Store in memory
        if self.training:
            idx = self.memory_counter % 100
            self.source_memory[idx] = source_entities.mean(dim=1).mean(dim=0).detach()
            self.target_memory[idx] = target_entities.mean(dim=1).mean(dim=0).detach()
            self.memory_counter += 1
        
        return {
            'source_entities': source_entities,
            'target_entities': target_entities,
            'source_relations': source_relations,
            'target_relations': target_relations,
            'mapping_scores': mapping_results['mapping_scores'],
            'correspondence_scores': mapping_results['correspondence_scores']
        }
    
    def generate_analogy(self, source: torch.Tensor, partial_target: torch.Tensor,
                        source_mask: Optional[torch.Tensor] = None,
                        target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate an analogy by completing a partial target.
        
        Args:
            source: Source domain input of shape [batch_size, source_len, input_dim]
            partial_target: Partial target input of shape [batch_size, target_len, input_dim]
            source_mask: Optional mask for source
            target_mask: Optional mask for target
            
        Returns:
            Completed target of shape [batch_size, target_len, input_dim]
        """
        # Get analogical reasoning results
        results = self.forward(source, partial_target, source_mask, target_mask)
        
        # Extract relevant components
        source_entities = results['source_entities']
        target_entities = results['target_entities']
        correspondence_scores = results['correspondence_scores']
        
        batch_size, target_len, _ = partial_target.shape
        
        # Generate completions for each position in the target
        completions = []
        
        for i in range(target_len):
            # Skip if masked
            if target_mask is not None and not target_mask[:, i].any():
                completions.append(partial_target[:, i])
                continue
            
            # Get correspondence scores for this position
            pos_scores = correspondence_scores[:, :, i]
            
            # Weighted sum of source entities
            weighted_source = torch.bmm(
                pos_scores.unsqueeze(1),
                source_entities
            ).squeeze(1)
            
            # Combine with target entity
            combined = torch.cat([weighted_source, target_entities[:, i]], dim=-1)
            
            # Generate completion
            completion = self.analogy_generator(combined)
            completions.append(completion)
        
        # Stack completions
        completed_target = torch.stack(completions, dim=1)
        
        return completed_target
    
    def retrieve_analogy(self, query: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the most similar analogies from memory.
        
        Args:
            query: Query tensor of shape [batch_size, input_dim]
            k: Number of analogies to retrieve
            
        Returns:
            Tuple of (source_analogies, target_analogies) each of shape [batch_size, k, hidden_dim]
        """
        batch_size = query.shape[0]
        
        # Encode query
        query_encoded = self.entity_encoder(query).mean(dim=1)
        
        # Compute similarity with source memory
        similarity = F.cosine_similarity(
            query_encoded.unsqueeze(1),
            self.source_memory.unsqueeze(0),
            dim=-1
        )
        
        # Get top-k indices
        _, indices = torch.topk(similarity, k=min(k, self.memory_counter), dim=-1)
        
        # Gather source and target analogies
        source_analogies = torch.gather(
            self.source_memory.unsqueeze(0).expand(batch_size, -1, -1),
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        
        target_analogies = torch.gather(
            self.target_memory.unsqueeze(0).expand(batch_size, -1, -1),
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        
        return source_analogies, target_analogies
