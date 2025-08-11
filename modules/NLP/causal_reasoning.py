import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class CausalGraph(nn.Module):
    """
    A neural causal graph that learns and represents causal relationships
    between variables in a domain.
    """
    
    def __init__(self, num_variables: int, hidden_dim: int = 64):
        """
        Initialize a causal graph.
        
        Args:
            num_variables: Number of variables in the causal domain
            hidden_dim: Hidden dimension for edge representations
        """
        super().__init__()
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        
        # Edge existence probabilities (i->j means i causes j)
        self.edge_logits = nn.Parameter(torch.zeros(num_variables, num_variables))
        
        # Edge strength representations
        self.edge_strengths = nn.Parameter(torch.zeros(num_variables, num_variables))
        
        # Mask to prevent self-loops
        self.register_buffer('self_loop_mask', 
                            1 - torch.eye(num_variables))
        
        # Edge feature network
        self.edge_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def get_adjacency_matrix(self, hard: bool = False, tau: float = 1.0) -> torch.Tensor:
        """
        Get the adjacency matrix of the causal graph.
        
        Args:
            hard: Whether to use hard (discrete) edges
            tau: Temperature for Gumbel softmax if using hard edges
            
        Returns:
            Adjacency matrix of shape [num_variables, num_variables]
        """
        # Apply mask to prevent self-loops
        masked_logits = self.edge_logits * self.self_loop_mask
        
        if hard:
            # Use straight-through Gumbel softmax for discrete edges
            edges = F.gumbel_softmax(
                torch.stack([torch.zeros_like(masked_logits), masked_logits], dim=-1),
                tau=tau,
                hard=True
            )[..., 1]
        else:
            # Use sigmoid for continuous edges
            edges = torch.sigmoid(masked_logits)
        
        return edges
    
    def get_weighted_adjacency(self, hard: bool = False, tau: float = 1.0) -> torch.Tensor:
        """
        Get the weighted adjacency matrix with edge strengths.
        
        Args:
            hard: Whether to use hard (discrete) edges
            tau: Temperature for Gumbel softmax if using hard edges
            
        Returns:
            Weighted adjacency matrix of shape [num_variables, num_variables]
        """
        adj = self.get_adjacency_matrix(hard, tau)
        edge_weights = torch.tanh(self.edge_strengths)
        return adj * edge_weights
    
    def forward(self, inputs: torch.Tensor, interventions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform causal inference to predict effects.
        
        Args:
            inputs: Input tensor of shape [batch_size, num_variables]
            interventions: Optional intervention tensor of shape [batch_size, num_variables]
                          with 1s indicating intervened variables
            
        Returns:
            Predicted effects of shape [batch_size, num_variables]
        """
        batch_size = inputs.shape[0]
        
        # Get weighted adjacency matrix
        weighted_adj = self.get_weighted_adjacency()
        
        # Initialize with input values
        effects = inputs.clone()
        
        # Apply interventions if provided
        if interventions is not None:
            # Keep only the intervened values, zero out the rest
            effects = effects * interventions
        
        # Propagate effects through the causal graph (simplified)
        # In a real implementation, this would use a more sophisticated
        # causal inference algorithm like do-calculus
        for _ in range(self.num_variables):  # Max path length
            # Propagate one step
            effects = effects + torch.matmul(effects, weighted_adj)
            
            # Re-apply interventions to maintain their values
            if interventions is not None:
                effects = effects * (1 - interventions) + inputs * interventions
        
        return effects
    
    def compute_intervention_effects(self, 
                                    inputs: torch.Tensor, 
                                    intervention_idx: int, 
                                    intervention_value: float) -> torch.Tensor:
        """
        Compute the effects of an intervention on a specific variable.
        
        Args:
            inputs: Input tensor of shape [batch_size, num_variables]
            intervention_idx: Index of the variable to intervene on
            intervention_value: Value to set for the intervened variable
            
        Returns:
            Intervention effects of shape [batch_size, num_variables]
        """
        batch_size = inputs.shape[0]
        
        # Create intervention mask
        interventions = torch.zeros_like(inputs)
        interventions[:, intervention_idx] = 1.0
        
        # Create intervened inputs
        intervened_inputs = inputs.clone()
        intervened_inputs[:, intervention_idx] = intervention_value
        
        # Compute effects
        effects = self.forward(intervened_inputs, interventions)
        
        # Compute counterfactual effects (difference from baseline)
        baseline = self.forward(inputs)
        counterfactual_effects = effects - baseline
        
        return counterfactual_effects

class CausalReasoning(nn.Module):
    """
    Module for causal reasoning, inference, and counterfactual analysis.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_variables: int = 12):
        """
        Initialize causal reasoning module.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for causal representations
            num_variables: Number of causal variables to model
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_variables = num_variables
        
        # Encoder to map input to causal variables
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variables)
        )
        
        # Decoder to map causal variables back to output space
        self.decoder = nn.Sequential(
            nn.Linear(num_variables, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Causal graph for modeling relationships
        self.causal_graph = CausalGraph(num_variables, hidden_dim)
        
        # Intervention generator
        self.intervention_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_variables * 2)  # For both mask and values
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to causal variables"""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode causal variables to output"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor, intervention_query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform causal reasoning on input.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            intervention_query: Optional query for intervention
            
        Returns:
            Dictionary with causal reasoning outputs
        """
        batch_size = x.shape[0]
        
        # Encode input to causal variables
        causal_variables = self.encode(x)
        
        # Perform causal inference
        if intervention_query is not None:
            # Generate intervention from query
            intervention_logits = self.intervention_generator(intervention_query)
            intervention_mask = torch.sigmoid(intervention_logits[:, :self.num_variables])
            intervention_values = intervention_logits[:, self.num_variables:]
            
            # Apply intervention
            intervened_variables = causal_variables * (1 - intervention_mask) + intervention_values * intervention_mask
            
            # Compute effects
            effects = self.causal_graph(intervened_variables, intervention_mask)
        else:
            # No intervention, just propagate effects
            effects = self.causal_graph(causal_variables)
            intervention_mask = None
        
        # Decode back to output space
        output = self.decode(effects)
        
        return {
            'output': output,
            'causal_variables': causal_variables,
            'effects': effects,
            'intervention_mask': intervention_mask,
            'adjacency_matrix': self.causal_graph.get_adjacency_matrix()
        }
    
    def counterfactual(self, x: torch.Tensor, intervention_idx: int, 
                      intervention_value: float) -> torch.Tensor:
        """
        Generate a counterfactual by intervening on a specific variable.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            intervention_idx: Index of variable to intervene on
            intervention_value: Value to set for the intervened variable
            
        Returns:
            Counterfactual output
        """
        # Encode input to causal variables
        causal_variables = self.encode(x)
        
        # Compute counterfactual effects
        counterfactual_effects = self.causal_graph.compute_intervention_effects(
            causal_variables, intervention_idx, intervention_value
        )
        
        # Add effects to original variables
        counterfactual_variables = causal_variables + counterfactual_effects
        
        # Decode to output space
        counterfactual_output = self.decode(counterfactual_variables)
        
        return counterfactual_output
    
    def explain_causal_path(self, x: torch.Tensor, from_var: int, to_var: int) -> List[Tuple[int, float]]:
        """
        Explain the causal path between two variables.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            from_var: Starting variable index
            to_var: Target variable index
            
        Returns:
            List of (variable_idx, strength) tuples representing the causal path
        """
        # Get weighted adjacency matrix
        weighted_adj = self.causal_graph.get_weighted_adjacency().detach().cpu().numpy()
        
        # Find the shortest path using a simple BFS
        visited = set([from_var])
        queue = [(from_var, [])]
        
        while queue:
            node, path = queue.pop(0)
            
            if node == to_var:
                # Found the path
                result = []
                prev = from_var
                for curr in path + [to_var]:
                    strength = float(weighted_adj[prev, curr])
                    result.append((curr, strength))
                    prev = curr
                return result
            
            # Add neighbors to queue
            for next_node in range(self.num_variables):
                if next_node not in visited and weighted_adj[node, next_node] > 0.1:
                    visited.add(next_node)
                    queue.append((next_node, path + [next_node]))
        
        # No path found
        return []
    
    def identify_causal_factors(self, x: torch.Tensor, target_idx: int, 
                               threshold: float = 0.1) -> List[Tuple[int, float]]:
        """
        Identify the causal factors that influence a target variable.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            target_idx: Index of the target variable
            threshold: Minimum strength threshold for causal relationships
            
        Returns:
            List of (variable_idx, strength) tuples representing causal factors
        """
        # Get weighted adjacency matrix
        weighted_adj = self.causal_graph.get_weighted_adjacency().detach().cpu().numpy()
        
        # Find all variables that have a direct causal effect on the target
        causal_factors = []
        for var_idx in range(self.num_variables):
            if var_idx != target_idx and abs(weighted_adj[var_idx, target_idx]) > threshold:
                strength = float(weighted_adj[var_idx, target_idx])
                causal_factors.append((var_idx, strength))
        
        # Sort by absolute strength
        causal_factors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return causal_factors
