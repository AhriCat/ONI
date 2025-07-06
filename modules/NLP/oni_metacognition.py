import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from collections import defaultdict
from NLP.conflict_graph import PrincipleConflictGraph

class AbductiveReasoning(nn.Module):
    """Abductive reasoning module for inferring the best explanation for observations."""
    
    def __init__(self, hidden_dim: int, num_hypotheses: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_hypotheses = num_hypotheses
        
        # Hypothesis generator
        self.hypothesis_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_hypotheses * hidden_dim)
        )
        
        # Hypothesis evaluator
        self.hypothesis_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Hypothesis memory
        self.register_buffer('hypothesis_memory', torch.zeros(num_hypotheses, hidden_dim))
        self.register_buffer('hypothesis_scores', torch.zeros(num_hypotheses))
        self.hypothesis_count = 0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate and evaluate hypotheses to explain the input.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            
        Returns:
            best_hypothesis: The best hypothesis
            metadata: Dictionary with hypothesis information
        """
        batch_size = x.size(0)
        
        # Generate hypotheses
        hypotheses_flat = self.hypothesis_generator(x)
        hypotheses = hypotheses_flat.view(batch_size, self.num_hypotheses, self.hidden_dim)
        
        # Evaluate each hypothesis
        scores = []
        for i in range(self.num_hypotheses):
            hypothesis = hypotheses[:, i, :]
            
            # Concatenate input and hypothesis
            eval_input = torch.cat([x, hypothesis], dim=-1)
            
            # Evaluate how well the hypothesis explains the input
            score = self.hypothesis_evaluator(eval_input)
            scores.append(score)
        
        # Stack scores
        scores = torch.cat(scores, dim=-1)
        
        # Get best hypothesis
        best_idx = torch.argmax(scores, dim=-1, keepdim=True)
        best_scores = torch.gather(scores, -1, best_idx)
        
        # Gather best hypotheses
        best_idx_expanded = best_idx.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        best_hypothesis = torch.gather(hypotheses, 1, best_idx_expanded).squeeze(1)
        
        # Update hypothesis memory with new hypotheses if they're better than existing ones
        if self.training:
            for i in range(batch_size):
                # Find the lowest scoring hypothesis in memory
                if self.hypothesis_count < self.num_hypotheses:
                    # Memory not full yet, add new hypothesis
                    idx = self.hypothesis_count
                    self.hypothesis_memory[idx] = best_hypothesis[i].detach()
                    self.hypothesis_scores[idx] = best_scores[i].detach()
                    self.hypothesis_count += 1
                else:
                    # Memory full, replace lowest scoring hypothesis if new one is better
                    min_idx = torch.argmin(self.hypothesis_scores)
                    if best_scores[i] > self.hypothesis_scores[min_idx]:
                        self.hypothesis_memory[min_idx] = best_hypothesis[i].detach()
                        self.hypothesis_scores[min_idx] = best_scores[i].detach()
        
        return best_hypothesis, {
            'all_hypotheses': hypotheses,
            'scores': scores,
            'best_idx': best_idx,
            'best_score': best_scores
        }
    
    def retrieve_similar_hypothesis(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Retrieve the most similar hypothesis from memory."""
        if self.hypothesis_count == 0:
            return torch.zeros_like(x), 0.0
        
        # Calculate similarity with all hypotheses in memory
        similarities = F.cosine_similarity(
            x.unsqueeze(1).expand(-1, self.hypothesis_count, -1),
            self.hypothesis_memory[:self.hypothesis_count].unsqueeze(0),
            dim=2
        )
        
        # Get most similar hypothesis
        best_idx = torch.argmax(similarities, dim=1)
        best_similarity = torch.gather(similarities, 1, best_idx.unsqueeze(1)).squeeze(1)
        
        # Gather best hypotheses
        best_hypothesis = self.hypothesis_memory[best_idx]
        
        return best_hypothesis, best_similarity

class AnalogicalReasoning(nn.Module):
    """Analogical reasoning module for solving problems by drawing parallels to similar problems."""
    
    def __init__(self, hidden_dim: int, memory_size: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # Source encoder
        self.source_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Target encoder
        self.target_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Mapping network
        self.mapping_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # Analogy memory
        self.register_buffer('source_memory', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('target_memory', torch.zeros(memory_size, hidden_dim))
        self.register_buffer('mapping_memory', torch.zeros(memory_size, hidden_dim))
        self.memory_count = 0
    
    def forward(self, source: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform analogical reasoning.
        
        Args:
            source: Source problem tensor of shape [batch_size, hidden_dim]
            target: Optional target problem tensor of shape [batch_size, hidden_dim]
            
        Returns:
            mapping: The analogical mapping
            metadata: Dictionary with additional information
        """
        batch_size = source.size(0)
        
        # Encode source
        source_encoded = self.source_encoder(source)
        
        if target is not None:
            # Encode target
            target_encoded = self.target_encoder(target)
            
            # Create mapping
            mapping_input = torch.cat([source_encoded, target_encoded], dim=-1)
            mapping = self.mapping_network(mapping_input)
            
            # Store in memory if training
            if self.training:
                for i in range(batch_size):
                    if self.memory_count < self.memory_size:
                        idx = self.memory_count
                        self.source_memory[idx] = source_encoded[i].detach()
                        self.target_memory[idx] = target_encoded[i].detach()
                        self.mapping_memory[idx] = mapping[i].detach()
                        self.memory_count += 1
            
            return mapping, {
                'source_encoded': source_encoded,
                'target_encoded': target_encoded
            }
        else:
            # Retrieve most similar source from memory
            if self.memory_count == 0:
                # No memories yet, return zeros
                return torch.zeros_like(source), {
                    'source_encoded': source_encoded,
                    'similarity': torch.zeros(batch_size, device=source.device)
                }
            
            # Calculate similarity with all sources in memory
            similarities = F.cosine_similarity(
                source_encoded.unsqueeze(1).expand(-1, self.memory_count, -1),
                self.source_memory[:self.memory_count].unsqueeze(0),
                dim=2
            )
            
            # Get most similar source
            best_idx = torch.argmax(similarities, dim=1)
            best_similarity = torch.gather(similarities, 1, best_idx.unsqueeze(1)).squeeze(1)
            
            # Gather corresponding mapping
            best_mapping = self.mapping_memory[best_idx]
            
            return best_mapping, {
                'source_encoded': source_encoded,
                'similarity': best_similarity
            }

class CausalInferenceEngine(nn.Module):
    """Causal inference engine for understanding cause-and-effect relationships."""
    
    def __init__(self, hidden_dim: int, max_variables: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_variables = max_variables
        
        # Variable encoder
        self.variable_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Causal graph generator
        self.graph_generator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Intervention predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Causal graph memory
        self.register_buffer('causal_graph', torch.zeros(max_variables, max_variables))
        self.register_buffer('variable_embeddings', torch.zeros(max_variables, hidden_dim))
        self.variable_count = 0
    
    def forward(self, variables: torch.Tensor, interventions: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform causal inference.
        
        Args:
            variables: Variable tensor of shape [batch_size, num_variables, hidden_dim]
            interventions: Optional intervention tensor of shape [batch_size, num_interventions, hidden_dim]
            
        Returns:
            effects: The predicted effects
            metadata: Dictionary with additional information
        """
        batch_size, num_variables, _ = variables.shape
        
        # Encode variables
        encoded_variables = self.variable_encoder(variables)
        
        # Generate causal graph
        causal_graph = torch.zeros(batch_size, num_variables, num_variables, device=variables.device)
        
        for i in range(num_variables):
            for j in range(num_variables):
                if i != j:
                    # Check if variable i causes variable j
                    pair = torch.cat([encoded_variables[:, i], encoded_variables[:, j]], dim=-1)
                    causal_graph[:, i, j] = self.graph_generator(pair).squeeze(-1)
        
        if interventions is not None:
            # Predict effects of interventions
            num_interventions = interventions.size(1)
            effects = torch.zeros(batch_size, num_interventions, num_variables, self.hidden_dim, device=variables.device)
            
            for i in range(num_interventions):
                intervention = interventions[:, i]
                
                # Find variables affected by the intervention
                for j in range(num_variables):
                    # Calculate influence of intervention on variable j
                    pair = torch.cat([intervention, encoded_variables[:, j]], dim=-1)
                    effect = self.intervention_predictor(pair)
                    effects[:, i, j] = effect
            
            # Update causal graph memory if training
            if self.training and batch_size == 1:
                for i in range(min(num_variables, self.max_variables)):
                    if i >= self.variable_count:
                        # Add new variable
                        self.variable_embeddings[i] = encoded_variables[0, i].detach()
                        self.variable_count += 1
                    else:
                        # Update existing variable
                        self.variable_embeddings[i] = 0.9 * self.variable_embeddings[i] + 0.1 * encoded_variables[0, i].detach()
                
                # Update causal graph for existing variables
                for i in range(min(num_variables, self.variable_count)):
                    for j in range(min(num_variables, self.variable_count)):
                        if i != j:
                            self.causal_graph[i, j] = 0.9 * self.causal_graph[i, j] + 0.1 * causal_graph[0, i, j].detach()
            
            return effects, {
                'causal_graph': causal_graph,
                'encoded_variables': encoded_variables
            }
        else:
            # Just return the causal graph
            return causal_graph, {
                'encoded_variables': encoded_variables
            }
    
    def get_causal_graph(self) -> torch.Tensor:
        """Get the learned causal graph."""
        return self.causal_graph[:self.variable_count, :self.variable_count]
    
    def predict_intervention_effect(self, intervention: torch.Tensor, target_variable_idx: int) -> torch.Tensor:
        """Predict the effect of an intervention on a target variable."""
        if self.variable_count == 0:
            return torch.zeros_like(intervention)
        
        if target_variable_idx >= self.variable_count:
            raise ValueError(f"Target variable index {target_variable_idx} out of range (0-{self.variable_count-1})")
        
        # Encode intervention
        encoded_intervention = self.variable_encoder(intervention)
        
        # Get target variable embedding
        target_variable = self.variable_embeddings[target_variable_idx].unsqueeze(0)
        
        # Predict effect
        pair = torch.cat([encoded_intervention, target_variable], dim=-1)
        effect = self.intervention_predictor(pair)
        
        return effect

class MetaCognitionModule(nn.Module):
    """Enhanced MetaCognition module with advanced reasoning capabilities."""
    
    def __init__(self, hidden_dim: int, num_principles: int = 10):
        """
        Initialize the enhanced MetaCognition module.
        
        Args:
            hidden_dim: Dimensionality of the hidden input tensor
            num_principles: Maximum number of principles to track
        """
        super(MetaCognitionModule, self).__init__()
        
        # Core components from original implementation
        self.hidden_dim = hidden_dim
        self.num_principles = num_principles
        
        # Self-reflection and confidence layers
        self.self_reflection = nn.Linear(hidden_dim, hidden_dim)
        self.confidence_estimation = nn.Linear(hidden_dim, 1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Dynamic storage for principles
        self.principles = nn.ParameterList()
        self.principle_descriptions = []
        
        # Contextual projection layer for principles
        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Alignment layer
        self.adaptive_alignment = nn.Linear(hidden_dim, hidden_dim)
        
        # Conflict graph for tracking principle conflicts
        self.conflict_graph = PrincipleConflictGraph(threshold=0.5)
        
        # Advanced reasoning components
        self.abductive_reasoning = AbductiveReasoning(hidden_dim)
        self.analogical_reasoning = AnalogicalReasoning(hidden_dim)
        self.causal_inference = CausalInferenceEngine(hidden_dim)
        
        # Uncertainty quantification
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Aleatoric and epistemic uncertainty
        )
        
        # Meta-memory for tracking reasoning patterns
        self.register_buffer('reasoning_memory', torch.zeros(100, hidden_dim))
        self.reasoning_memory_ptr = 0
        
        # Reasoning strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 reasoning strategies
        )
        
        # Principle importance weighting
        self.importance_weighting = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize principle weights
        self.principle_weights = torch.ones(num_principles) / num_principles
        
        # Metacognitive state tracking
        self.metacognitive_state = {
            'confidence_history': [],
            'conflict_history': [],
            'reasoning_strategies': [],
            'uncertainty_history': []
        }

    def add_principle(self, principle_vector: torch.Tensor, description: str = None):
        """
        Dynamically add a new principle to the module.
        
        Args:
            principle_vector: A tensor of shape (hidden_dim,)
            description: Optional text description of the principle
        """
        if len(self.principles) >= self.num_principles:
            # Replace the least important principle
            weights = [self.importance_weighting(p).item() for p in self.principles]
            min_idx = weights.index(min(weights))
            self.principles[min_idx] = nn.Parameter(principle_vector.clone(), requires_grad=True)
            if description:
                self.principle_descriptions[min_idx] = description
        else:
            if principle_vector.dim() != 1 or principle_vector.size(0) != self.hidden_dim:
                raise ValueError(f"Principle vector must be of shape ({self.hidden_dim},)")
            
            self.principles.append(nn.Parameter(principle_vector.clone(), requires_grad=True))
            self.principle_descriptions.append(description or f"Principle {len(self.principles)}")
            
            # Update principle weights
            self.principle_weights = torch.ones(len(self.principles)) / len(self.principles)

    def contextual_conflict_score(self, principle_a: torch.Tensor, principle_b: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute a nuanced conflict score between two principles in the given context.
        
        Args:
            principle_a: Tensor representing principle A
            principle_b: Tensor representing principle B
            context: The input context tensor
            
        Returns:
            score: Conflict score, where higher values indicate stronger conflict
        """
        # Project principles into the context space
        proj_a = self.context_projection(principle_a)
        proj_b = self.context_projection(principle_b)
        
        # Normalize vectors for more stable dot products
        context_weight = F.normalize(context, p=2, dim=-1)
        proj_a = F.normalize(proj_a, p=2, dim=-1)
        proj_b = F.normalize(proj_b, p=2, dim=-1)
        
        # Calculate directional alignment with context
        align_a = torch.sum(proj_a * context_weight, dim=-1)
        align_b = torch.sum(proj_b * context_weight, dim=-1)
        
        # Calculate principle similarity
        principle_sim = torch.sum(proj_a * proj_b, dim=-1)
        
        # Conflict occurs when principles are dissimilar but both align with context
        conflict_score = (1 - principle_sim) * align_a * align_b
        
        return conflict_score

    def detect_nuanced_conflicts(self, context: torch.Tensor, threshold: float = 0.5) -> List[Tuple[int, int, float]]:
        """
        Detect nuanced, context-aware conflicts between principles.
        
        Args:
            context: The input context tensor
            threshold: Threshold for determining significant conflict
            
        Returns:
            conflicts: List of conflicting principle index pairs and their scores
        """
        conflicts = []
        num_principles = len(self.principles)
        
        if num_principles < 2:
            return conflicts
        
        # Compare all principle pairs
        for i in range(num_principles):
            for j in range(i + 1, num_principles):
                score = self.contextual_conflict_score(
                    self.principles[i], 
                    self.principles[j], 
                    context
                )
                
                if score > threshold:
                    conflicts.append((i, j, score.item()))
                    
                    # Add to conflict graph
                    self.conflict_graph.add_conflict(i, j, score.item(), context)
        
        return conflicts

    def select_reasoning_strategy(self, x: torch.Tensor) -> int:
        """
        Select the most appropriate reasoning strategy for the input.
        
        Args:
            x: Input tensor
            
        Returns:
            strategy_idx: Index of selected strategy
                0: Deductive reasoning
                1: Abductive reasoning
                2: Analogical reasoning
                3: Causal reasoning
        """
        strategy_logits = self.strategy_selector(x)
        strategy_idx = torch.argmax(strategy_logits, dim=-1)
        
        # Record selected strategy
        self.metacognitive_state['reasoning_strategies'].append(strategy_idx.item())
        
        return strategy_idx

    def apply_reasoning_strategy(self, x: torch.Tensor, strategy_idx: int) -> torch.Tensor:
        """
        Apply the selected reasoning strategy.
        
        Args:
            x: Input tensor
            strategy_idx: Index of selected strategy
            
        Returns:
            output: Result of applying the reasoning strategy
        """
        if strategy_idx == 0:
            # Deductive reasoning (standard forward pass)
            return x
        elif strategy_idx == 1:
            # Abductive reasoning (find best explanation)
            hypothesis, _ = self.abductive_reasoning(x)
            return hypothesis
        elif strategy_idx == 2:
            # Analogical reasoning (solve by analogy)
            mapping, _ = self.analogical_reasoning(x)
            return mapping
        elif strategy_idx == 3:
            # Causal reasoning (understand cause-effect)
            # Reshape x to [batch_size, num_variables, hidden_dim]
            batch_size = x.size(0)
            reshaped_x = x.view(batch_size, -1, self.hidden_dim)
            causal_graph, _ = self.causal_inference(reshaped_x)
            
            # Apply causal effects
            return x * (1 + causal_graph.mean(dim=(1, 2)).unsqueeze(-1))
        else:
            return x

    def estimate_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate aleatoric and epistemic uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            aleatoric: Aleatoric uncertainty (data uncertainty)
            epistemic: Epistemic uncertainty (model uncertainty)
        """
        uncertainty = self.uncertainty_estimator(x)
        aleatoric = uncertainty[:, 0].unsqueeze(-1)  # Data uncertainty
        epistemic = uncertainty[:, 1].unsqueeze(-1)  # Model uncertainty
        
        # Record uncertainty
        self.metacognitive_state['uncertainty_history'].append((aleatoric.mean().item(), epistemic.mean().item()))
        
        return aleatoric, epistemic

    def update_reasoning_memory(self, x: torch.Tensor) -> None:
        """Update reasoning memory with new input."""
        idx = self.reasoning_memory_ptr % self.reasoning_memory.size(0)
        self.reasoning_memory[idx] = x.detach().mean(0)
        self.reasoning_memory_ptr += 1

    def get_similar_reasoning(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Find similar past reasoning pattern."""
        if self.reasoning_memory_ptr == 0:
            return torch.zeros_like(x), 0.0
        
        # Calculate similarity with reasoning memory
        memory_size = min(self.reasoning_memory_ptr, self.reasoning_memory.size(0))
        similarities = F.cosine_similarity(
            x.unsqueeze(1).expand(-1, memory_size, -1),
            self.reasoning_memory[:memory_size].unsqueeze(0),
            dim=2
        )
        
        # Get most similar reasoning
        best_idx = torch.argmax(similarities, dim=1)
        best_similarity = torch.gather(similarities, 1, best_idx.unsqueeze(1)).squeeze(1)
        
        # Gather best reasoning
        best_reasoning = self.reasoning_memory[best_idx]
        
        return best_reasoning, best_similarity

    def forward(self, x: torch.Tensor, conflict_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor, List[Tuple[int, int, float]], Dict[str, Any]]:
        """
        Enhanced forward pass with advanced reasoning capabilities.
        
        Args:
            x: Input tensor of shape [batch_size, hidden_dim]
            conflict_threshold: Threshold for detecting nuanced conflicts
            
        Returns:
            output: Updated input tensor after metacognitive processing
            confidence: Confidence score for self-reflection
            conflicts: List of nuanced conflicts detected
            metadata: Dictionary with additional metacognitive information
        """
        batch_size = x.size(0)
        
        # Select reasoning strategy
        strategy_idx = self.select_reasoning_strategy(x)
        
        # Apply reasoning strategy
        reasoned_x = self.apply_reasoning_strategy(x, strategy_idx)
        
        # Check for similar past reasoning
        similar_reasoning, similarity = self.get_similar_reasoning(reasoned_x)
        
        # If similar reasoning found, blend with current reasoning
        if similarity > 0.8:  # High similarity threshold
            reasoned_x = 0.7 * reasoned_x + 0.3 * similar_reasoning
        
        # Update reasoning memory
        self.update_reasoning_memory(reasoned_x)
        
        # Self-reflection step
        reflection = torch.tanh(self.self_reflection(reasoned_x))
        
        # Estimate uncertainty
        aleatoric, epistemic = self.estimate_uncertainty(reflection)
        
        # Aggregate principles dynamically
        if len(self.principles) > 0:
            principles = torch.stack(list(self.principles))  # Shape: [num_principles, hidden_dim]
            
            # Calculate principle importance for this context
            importance = self.importance_weighting(principles).squeeze(-1)  # Shape: [num_principles]
            importance = F.softmax(importance, dim=0)
            
            # Update principle weights with exponential moving average
            self.principle_weights = 0.9 * self.principle_weights + 0.1 * importance
            
            # Calculate attention scores between input and principles
            principle_weights = torch.softmax(torch.matmul(reasoned_x, principles.T), dim=-1)  # Shape: [batch_size, num_principles]
            
            # Weight by importance
            principle_weights = principle_weights * importance.unsqueeze(0)
            
            # Weighted principle alignment
            principle_alignment = torch.matmul(principle_weights, principles)  # Shape: [batch_size, hidden_dim]
            adaptive_reflection = self.adaptive_alignment(reflection + principle_alignment)
        else:
            adaptive_reflection = self.adaptive_alignment(reflection)
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_estimation(adaptive_reflection))
        
        # Adjust confidence based on uncertainty
        confidence = confidence * (1 - epistemic)
        
        # Residual connection and layer normalization
        output = self.layer_norm(reasoned_x + adaptive_reflection)
        
        # Detect nuanced conflicts
        conflicts = self.detect_nuanced_conflicts(reasoned_x.mean(dim=0), threshold=conflict_threshold)
        
        # Record confidence
        self.metacognitive_state['confidence_history'].append(confidence.mean().item())
        
        # Record conflicts
        if conflicts:
            self.metacognitive_state['conflict_history'].append(conflicts)
        
        # Prepare metadata
        metadata = {
            'strategy': strategy_idx.item(),
            'uncertainty': {
                'aleatoric': aleatoric.mean().item(),
                'epistemic': epistemic.mean().item()
            },
            'principle_weights': self.principle_weights.tolist() if len(self.principles) > 0 else [],
            'principle_descriptions': self.principle_descriptions,
            'similar_reasoning': similarity.item(),
            'conflict_patterns': self.conflict_graph.analyze_conflict_patterns()
        }
        
        return output, confidence, conflicts, metadata

    def get_metacognitive_state(self) -> Dict[str, Any]:
        """Get the current metacognitive state."""
        return {
            'confidence': self.metacognitive_state['confidence_history'][-10:] if self.metacognitive_state['confidence_history'] else [],
            'conflicts': self.conflict_graph.get_all_conflicts(),
            'most_conflicted_principles': self.conflict_graph.get_most_conflicted_principles(),
            'reasoning_strategies': self.metacognitive_state['reasoning_strategies'][-10:] if self.metacognitive_state['reasoning_strategies'] else [],
            'uncertainty': self.metacognitive_state['uncertainty_history'][-10:] if self.metacognitive_state['uncertainty_history'] else [],
            'principle_weights': self.principle_weights.tolist() if len(self.principles) > 0 else [],
            'principle_descriptions': self.principle_descriptions,
            'conflict_patterns': self.conflict_graph.analyze_conflict_patterns()
        }

    def explain_reasoning(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Generate an explanation of the metacognitive reasoning process.
        
        Args:
            x: Input tensor
            
        Returns:
            explanation: Dictionary with reasoning explanation
        """
        # Forward pass to get all metacognitive outputs
        output, confidence, conflicts, metadata = self.forward(x)
        
        # Select reasoning strategy
        strategy_idx = metadata['strategy']
        strategy_names = ["Deductive", "Abductive", "Analogical", "Causal"]
        strategy_name = strategy_names[strategy_idx]
        
        # Get uncertainty estimates
        aleatoric = metadata['uncertainty']['aleatoric']
        epistemic = metadata['uncertainty']['epistemic']
        
        # Get principle information
        principle_weights = metadata['principle_weights']
        principle_descriptions = metadata['principle_descriptions']
        
        # Prepare explanation
        explanation = {
            'reasoning_strategy': {
                'name': strategy_name,
                'description': f"Used {strategy_name.lower()} reasoning to process the input"
            },
            'confidence': {
                'score': confidence.mean().item(),
                'description': f"Confidence level: {confidence.mean().item():.2f}"
            },
            'uncertainty': {
                'aleatoric': aleatoric,
                'epistemic': epistemic,
                'description': f"Data uncertainty: {aleatoric:.2f}, Model uncertainty: {epistemic:.2f}"
            },
            'principles': {
                'active_principles': [
                    {
                        'description': desc,
                        'weight': weight
                    }
                    for desc, weight in zip(principle_descriptions, principle_weights)
                ] if principle_descriptions else []
            },
            'conflicts': {
                'detected': [
                    {
                        'principles': [principle_descriptions[i], principle_descriptions[j]],
                        'score': score
                    }
                    for i, j, score in conflicts
                ] if conflicts else []
            }
        }
        
        return explanation
