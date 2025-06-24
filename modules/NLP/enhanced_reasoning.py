import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any

from .causal_reasoning import CausalReasoning
from .analogical_reasoning import AnalogicalReasoning
from .counterfactual_reasoning import CounterfactualReasoning
from .multi_step_planning import MultiStepPlanning

class EnhancedReasoning(nn.Module):
    """
    Unified module that integrates multiple advanced reasoning capabilities:
    - Causal reasoning
    - Analogical reasoning
    - Counterfactual reasoning
    - Multi-step planning
    
    This module serves as a central hub for all enhanced reasoning systems
    and provides a unified interface for the rest of the ONI system.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize enhanced reasoning module.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for reasoning representations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize reasoning modules
        self.causal_reasoning = CausalReasoning(input_dim, hidden_dim)
        self.analogical_reasoning = AnalogicalReasoning(input_dim, hidden_dim)
        self.counterfactual_reasoning = CounterfactualReasoning(input_dim, hidden_dim)
        self.multi_step_planning = MultiStepPlanning(input_dim, hidden_dim)
        
        # Reasoning type selector
        self.reasoning_selector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),  # 4 reasoning types
            nn.Softmax(dim=-1)
        )
        
        # Output integration
        self.output_integrator = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Perform enhanced reasoning on input.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            context: Optional context dictionary with additional inputs
            
        Returns:
            Dictionary with reasoning outputs
        """
        batch_size = x.shape[0]
        
        # Determine which reasoning types to use
        reasoning_weights = self.reasoning_selector(x)
        
        # Extract context elements if provided
        source_domain = context.get('source_domain', None) if context else None
        target_domain = context.get('target_domain', None) if context else None
        goal = context.get('goal', None) if context else None
        intervention_query = context.get('intervention_query', None) if context else None
        
        # Apply each reasoning type
        causal_output = self.causal_reasoning(x, intervention_query)['output']
        
        # For analogical reasoning, we need both source and target domains
        if source_domain is not None and target_domain is not None:
            analogical_output = self.analogical_reasoning.generate_analogy(
                source_domain, target_domain
            )
        else:
            # Use x as both source and target if not provided
            analogical_output = self.analogical_reasoning.generate_analogy(
                x.unsqueeze(1), x.unsqueeze(1)
            )
        
        # For counterfactual reasoning
        counterfactual_output = self.counterfactual_reasoning(x)['counterfactuals'][:, 0]
        
        # For multi-step planning
        if goal is not None:
            planning_output = self.multi_step_planning(x, goal)['plan'][:, 0]
        else:
            # Use x as goal if not provided
            planning_output = self.multi_step_planning(x, x)['plan'][:, 0]
        
        # Integrate outputs based on reasoning weights
        integrated_output = (
            reasoning_weights[:, 0].unsqueeze(-1) * causal_output +
            reasoning_weights[:, 1].unsqueeze(-1) * analogical_output +
            reasoning_weights[:, 2].unsqueeze(-1) * counterfactual_output +
            reasoning_weights[:, 3].unsqueeze(-1) * planning_output
        )
        
        # Apply final integration
        final_output = self.output_integrator(
            torch.cat([causal_output, analogical_output, 
                      counterfactual_output, planning_output], dim=-1)
        )
        
        return {
            'output': final_output,
            'integrated_output': integrated_output,
            'reasoning_weights': reasoning_weights,
            'causal_output': causal_output,
            'analogical_output': analogical_output,
            'counterfactual_output': counterfactual_output,
            'planning_output': planning_output
        }
    
    def explain_reasoning(self, x: torch.Tensor, context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Generate explanations for the reasoning process.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            context: Optional context dictionary
            
        Returns:
            Dictionary with reasoning explanations
        """
        # Get reasoning outputs
        outputs = self.forward(x, context)
        
        # Get reasoning weights
        reasoning_weights = outputs['reasoning_weights']
        
        # Determine primary reasoning type
        primary_reasoning_idx = reasoning_weights.argmax(dim=-1)
        reasoning_types = ['causal', 'analogical', 'counterfactual', 'planning']
        
        explanations = []
        for i in range(x.shape[0]):
            primary_type = reasoning_types[primary_reasoning_idx[i].item()]
            weight = reasoning_weights[i, primary_reasoning_idx[i]].item()
            
            explanation = {
                'primary_reasoning_type': primary_type,
                'confidence': weight,
                'reasoning_weights': {
                    reasoning_type: reasoning_weights[i, j].item()
                    for j, reasoning_type in enumerate(reasoning_types)
                }
            }
            
            # Add type-specific explanations
            if primary_type == 'causal':
                # Extract causal factors
                explanation['causal_factors'] = [
                    {'variable': j, 'strength': float(outputs['causal_output'][i, j])}
                    for j in range(min(5, self.input_dim))
                ]
            
            elif primary_type == 'analogical':
                # Explain analogy
                explanation['analogy'] = {
                    'source_to_target_mapping': 'Mapped similar structures between domains'
                }
            
            elif primary_type == 'counterfactual':
                # Explain counterfactual
                explanation['counterfactual'] = {
                    'what_if_scenario': 'Generated alternative scenario by changing key factors'
                }
            
            elif primary_type == 'planning':
                # Explain plan
                explanation['plan'] = {
                    'steps': 'Decomposed goal into achievable subgoals and actions'
                }
            
            explanations.append(explanation)
        
        return {
            'explanations': explanations,
            'reasoning_outputs': outputs
        }
