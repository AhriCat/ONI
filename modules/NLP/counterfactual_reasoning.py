import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class CounterfactualGenerator(nn.Module):
    """
    Module for generating counterfactual scenarios by modifying input features
    and predicting alternative outcomes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_counterfactuals: int = 5):
        """
        Initialize counterfactual generator.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for counterfactual representations
            num_counterfactuals: Number of counterfactuals to generate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_counterfactuals = num_counterfactuals
        
        # Feature importance estimator
        self.feature_importance = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Counterfactual generator
        self.generator = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * num_counterfactuals)
        )
        
        # Outcome predictor
        self.outcome_predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual scenarios.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            condition: Optional conditioning tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary with counterfactual outputs
        """
        batch_size = x.shape[0]
        
        # Estimate feature importance
        importance = self.feature_importance(x)
        
        # Generate counterfactuals
        if condition is None:
            # Use zeros as default condition
            condition = torch.zeros_like(x)
        
        # Concatenate input and condition
        generator_input = torch.cat([x, condition], dim=-1)
        
        # Generate counterfactual deltas
        counterfactual_deltas = self.generator(generator_input)
        counterfactual_deltas = counterfactual_deltas.view(
            batch_size, self.num_counterfactuals, self.input_dim
        )
        
        # Apply importance-weighted deltas to create counterfactuals
        importance_expanded = importance.unsqueeze(1).expand(
            -1, self.num_counterfactuals, -1
        )
        
        counterfactuals = x.unsqueeze(1) + importance_expanded * counterfactual_deltas
        
        # Predict outcomes for each counterfactual
        counterfactuals_flat = counterfactuals.view(-1, self.input_dim)
        outcomes_flat = self.outcome_predictor(counterfactuals_flat)
        outcomes = outcomes_flat.view(batch_size, self.num_counterfactuals, self.input_dim)
        
        return {
            'counterfactuals': counterfactuals,
            'outcomes': outcomes,
            'importance': importance,
            'deltas': counterfactual_deltas
        }
    
    def generate_targeted_counterfactual(self, x: torch.Tensor, target: torch.Tensor, 
                                        steps: int = 10, step_size: float = 0.1) -> torch.Tensor:
        """
        Generate a counterfactual that achieves a specific target outcome.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            target: Target outcome of shape [batch_size, input_dim]
            steps: Number of optimization steps
            step_size: Step size for optimization
            
        Returns:
            Counterfactual input that achieves the target outcome
        """
        # Start with the original input
        counterfactual = x.clone().detach().requires_grad_(True)
        
        # Estimate feature importance
        importance = self.feature_importance(x).detach()
        
        # Optimize counterfactual to achieve target outcome
        for _ in range(steps):
            # Predict outcome
            outcome = self.outcome_predictor(counterfactual)
            
            # Compute loss
            loss = F.mse_loss(outcome, target)
            
            # Compute gradients
            loss.backward()
            
            # Update counterfactual with importance-weighted gradients
            with torch.no_grad():
                grad = counterfactual.grad * importance
                counterfactual -= step_size * grad
                counterfactual.grad.zero_()
        
        return counterfactual.detach()

class ScenarioGenerator(nn.Module):
    """
    Module for generating "what if" scenarios by exploring alternative
    paths and outcomes.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_scenarios: int = 5):
        """
        Initialize scenario generator.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for scenario representations
            num_scenarios: Number of scenarios to generate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_scenarios = num_scenarios
        
        # Scenario encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Scenario generator
        self.generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * num_scenarios)
        )
        
        # Scenario decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Probability estimator
        self.probability_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Generate alternative scenarios.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            condition: Optional conditioning tensor
            
        Returns:
            Dictionary with scenario outputs
        """
        batch_size = x.shape[0]
        
        # Encode input
        encoded = self.encoder(x)
        
        # Apply conditioning if provided
        if condition is not None:
            condition_encoded = self.encoder(condition)
            encoded = encoded + condition_encoded
        
        # Generate scenario embeddings
        scenario_embeddings = self.generator(encoded)
        scenario_embeddings = scenario_embeddings.view(
            batch_size, self.num_scenarios, self.hidden_dim
        )
        
        # Decode scenarios
        scenario_embeddings_flat = scenario_embeddings.view(-1, self.hidden_dim)
        scenarios_flat = self.decoder(scenario_embeddings_flat)
        scenarios = scenarios_flat.view(batch_size, self.num_scenarios, self.input_dim)
        
        # Estimate probabilities
        probabilities_flat = self.probability_estimator(scenario_embeddings_flat)
        probabilities = probabilities_flat.view(batch_size, self.num_scenarios)
        
        # Normalize probabilities
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        
        return {
            'scenarios': scenarios,
            'probabilities': probabilities,
            'embeddings': scenario_embeddings
        }
    
    def evaluate_scenario(self, scenario: torch.Tensor, condition: torch.Tensor) -> float:
        """
        Evaluate the probability of a scenario given a condition.
        
        Args:
            scenario: Scenario tensor of shape [batch_size, input_dim]
            condition: Condition tensor of shape [batch_size, input_dim]
            
        Returns:
            Probability of the scenario
        """
        # Encode scenario and condition
        scenario_encoded = self.encoder(scenario)
        condition_encoded = self.encoder(condition)
        
        # Compute similarity
        similarity = F.cosine_similarity(scenario_encoded, condition_encoded, dim=-1)
        
        # Estimate probability
        probability = self.probability_estimator(scenario_encoded).squeeze(-1)
        
        # Adjust probability based on similarity
        adjusted_probability = probability * (0.5 + 0.5 * similarity)
        
        return adjusted_probability

class CounterfactualReasoning(nn.Module):
    """
    Module for counterfactual reasoning and "what if" scenario generation.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize counterfactual reasoning module.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for counterfactual representations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Counterfactual generator
        self.counterfactual_generator = CounterfactualGenerator(
            input_dim, hidden_dim, num_counterfactuals=5
        )
        
        # Scenario generator
        self.scenario_generator = ScenarioGenerator(
            input_dim, hidden_dim, num_scenarios=5
        )
        
        # Intervention selector
        self.intervention_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Outcome comparator
        self.outcome_comparator = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, query: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Perform counterfactual reasoning.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            query: Optional query tensor for targeted counterfactuals
            
        Returns:
            Dictionary with counterfactual reasoning outputs
        """
        # Generate counterfactuals
        counterfactual_results = self.counterfactual_generator(x, query)
        
        # Generate scenarios
        scenario_results = self.scenario_generator(x, query)
        
        # Select intervention points
        intervention_mask = self.intervention_selector(x)
        
        # Combine results
        return {
            'counterfactuals': counterfactual_results['counterfactuals'],
            'counterfactual_outcomes': counterfactual_results['outcomes'],
            'scenarios': scenario_results['scenarios'],
            'scenario_probabilities': scenario_results['probabilities'],
            'intervention_mask': intervention_mask
        }
    
    def compare_outcomes(self, outcome1: torch.Tensor, outcome2: torch.Tensor) -> torch.Tensor:
        """
        Compare two outcomes and determine if outcome1 is better than outcome2.
        
        Args:
            outcome1: First outcome of shape [batch_size, input_dim]
            outcome2: Second outcome of shape [batch_size, input_dim]
            
        Returns:
            Comparison score (higher means outcome1 is better)
        """
        # Concatenate outcomes
        combined = torch.cat([outcome1, outcome2], dim=-1)
        
        # Compute comparison score
        score = self.outcome_comparator(combined)
        
        return score
    
    def generate_what_if(self, x: torch.Tensor, intervention: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate "what if" scenarios based on a specific intervention.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            intervention: Intervention tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary with "what if" scenario outputs
        """
        # Create intervention mask (1 where intervention is applied)
        intervention_mask = (intervention.abs() > 1e-6).float()
        
        # Apply intervention
        intervened_input = x * (1 - intervention_mask) + intervention * intervention_mask
        
        # Generate scenarios
        scenario_results = self.scenario_generator(intervened_input)
        
        # Generate counterfactuals
        counterfactual_results = self.counterfactual_generator(intervened_input)
        
        # Predict outcomes
        outcomes = counterfactual_results['outcomes']
        
        # Compare with original outcome
        original_outcome = self.counterfactual_generator.outcome_predictor(x)
        
        comparison_scores = []
        for i in range(outcomes.shape[1]):
            score = self.compare_outcomes(
                outcomes[:, i], original_outcome
            )
            comparison_scores.append(score)
        
        comparison_scores = torch.cat(comparison_scores, dim=-1)
        
        return {
            'intervened_input': intervened_input,
            'scenarios': scenario_results['scenarios'],
            'scenario_probabilities': scenario_results['probabilities'],
            'outcomes': outcomes,
            'comparison_scores': comparison_scores
        }
