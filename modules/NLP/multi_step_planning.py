import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

class PlanStep(nn.Module):
    """
    Represents a single step in a hierarchical plan.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize plan step.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for plan representations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Step encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Step decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Precondition checker
        self.precondition_checker = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Effect predictor
        self.effect_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process a plan step.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            state: Current state tensor of shape [batch_size, hidden_dim]
            
        Returns:
            Dictionary with step outputs
        """
        # Encode step
        step_embedding = self.encoder(x)
        
        # Check preconditions
        precondition_input = torch.cat([step_embedding, state], dim=-1)
        precondition_satisfied = self.precondition_checker(precondition_input)
        
        # Predict effects
        effects = self.effect_predictor(step_embedding)
        
        # Update state
        new_state = state + effects * precondition_satisfied
        
        # Decode step
        decoded_step = self.decoder(step_embedding)
        
        return {
            'step_embedding': step_embedding,
            'precondition_satisfied': precondition_satisfied,
            'effects': effects,
            'new_state': new_state,
            'decoded_step': decoded_step
        }

class SubgoalDecomposer(nn.Module):
    """
    Module for decomposing a goal into subgoals.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, max_subgoals: int = 5):
        """
        Initialize subgoal decomposer.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for subgoal representations
            max_subgoals: Maximum number of subgoals to generate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_subgoals = max_subgoals
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Subgoal generator
        self.subgoal_generator = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Subgoal decoder
        self.subgoal_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Stop token predictor
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, goal: torch.Tensor, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decompose a goal into subgoals.
        
        Args:
            goal: Goal tensor of shape [batch_size, input_dim]
            state: Current state tensor of shape [batch_size, hidden_dim]
            
        Returns:
            Dictionary with subgoal outputs
        """
        batch_size = goal.shape[0]
        
        # Encode goal
        goal_embedding = self.goal_encoder(goal)
        
        # Initialize hidden state with current state
        h = state
        
        # Generate subgoals
        subgoals = []
        subgoal_embeddings = []
        stop_probs = []
        
        for _ in range(self.max_subgoals):
            # Update hidden state
            h = self.subgoal_generator(goal_embedding, h)
            
            # Predict stop probability
            stop_prob = self.stop_predictor(h)
            stop_probs.append(stop_prob)
            
            # Decode subgoal
            subgoal = self.subgoal_decoder(h)
            
            subgoals.append(subgoal)
            subgoal_embeddings.append(h)
        
        # Stack outputs
        subgoals = torch.stack(subgoals, dim=1)  # [batch_size, max_subgoals, input_dim]
        subgoal_embeddings = torch.stack(subgoal_embeddings, dim=1)  # [batch_size, max_subgoals, hidden_dim]
        stop_probs = torch.cat(stop_probs, dim=-1)  # [batch_size, max_subgoals]
        
        return {
            'subgoals': subgoals,
            'subgoal_embeddings': subgoal_embeddings,
            'stop_probs': stop_probs
        }

class HierarchicalPlanner(nn.Module):
    """
    Module for hierarchical planning with subgoal decomposition.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, max_steps: int = 10, max_subgoals: int = 5):
        """
        Initialize hierarchical planner.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for plan representations
            max_steps: Maximum number of steps in a plan
            max_subgoals: Maximum number of subgoals
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        self.max_subgoals = max_subgoals
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Subgoal decomposer
        self.subgoal_decomposer = SubgoalDecomposer(
            input_dim, hidden_dim, max_subgoals
        )
        
        # Plan step
        self.plan_step = PlanStep(input_dim, hidden_dim)
        
        # Plan decoder
        self.plan_decoder = nn.GRUCell(hidden_dim * 2, hidden_dim)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Success predictor
        self.success_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, initial_state: torch.Tensor, goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate a hierarchical plan.
        
        Args:
            initial_state: Initial state tensor of shape [batch_size, input_dim]
            goal: Goal tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary with planning outputs
        """
        batch_size = initial_state.shape[0]
        
        # Encode state and goal
        state_embedding = self.state_encoder(initial_state)
        goal_embedding = self.goal_encoder(goal)
        
        # Decompose goal into subgoals
        subgoal_results = self.subgoal_decomposer(goal, state_embedding)
        subgoals = subgoal_results['subgoals']
        subgoal_embeddings = subgoal_results['subgoal_embeddings']
        stop_probs = subgoal_results['stop_probs']
        
        # Generate plan for each subgoal
        all_actions = []
        all_states = [state_embedding]
        current_state = state_embedding
        
        for i in range(self.max_subgoals):
            # Skip if stop probability is high
            if i > 0 and stop_probs[:, i-1].mean() > 0.9:
                break
                
            subgoal = subgoals[:, i]
            subgoal_embedding = subgoal_embeddings[:, i]
            
            # Plan steps for this subgoal
            subgoal_actions = []
            
            for j in range(self.max_steps):
                # Initialize decoder state
                decoder_input = torch.cat([current_state, subgoal_embedding], dim=-1)
                decoder_state = self.plan_decoder(decoder_input, current_state)
                
                # Decode action
                action = self.action_decoder(decoder_state)
                subgoal_actions.append(action)
                
                # Update state
                step_results = self.plan_step(action, current_state)
                current_state = step_results['new_state']
                all_states.append(current_state)
                
                # Check if subgoal is achieved
                success = self.success_predictor(
                    torch.cat([current_state, subgoal_embedding], dim=-1)
                )
                
                if success.mean() > 0.9:
                    break
            
            # Add actions for this subgoal
            all_actions.extend(subgoal_actions)
        
        # Stack outputs
        all_actions = torch.stack(all_actions, dim=1)  # [batch_size, num_actions, input_dim]
        all_states = torch.stack(all_states, dim=1)  # [batch_size, num_states, hidden_dim]
        
        # Predict overall success
        final_state = all_states[:, -1]
        overall_success = self.success_predictor(
            torch.cat([final_state, goal_embedding], dim=-1)
        )
        
        return {
            'plan': all_actions,
            'states': all_states,
            'subgoals': subgoals,
            'stop_probs': stop_probs,
            'success': overall_success
        }
    
    def execute_step(self, state: torch.Tensor, goal: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Execute a single step of the plan.
        
        Args:
            state: Current state tensor of shape [batch_size, input_dim]
            goal: Goal tensor of shape [batch_size, input_dim]
            
        Returns:
            Tuple of (next_action, success_probability)
        """
        batch_size = state.shape[0]
        
        # Encode state and goal
        state_embedding = self.state_encoder(state)
        goal_embedding = self.goal_encoder(goal)
        
        # Generate next action
        decoder_input = torch.cat([state_embedding, goal_embedding], dim=-1)
        decoder_state = self.plan_decoder(decoder_input, state_embedding)
        action = self.action_decoder(decoder_state)
        
        # Predict success
        success_prob = self.success_predictor(
            torch.cat([state_embedding, goal_embedding], dim=-1)
        )
        
        return action, success_prob.item()

class MultiStepPlanning(nn.Module):
    """
    Module for multi-step planning with hierarchical goal decomposition.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        """
        Initialize multi-step planning module.
        
        Args:
            input_dim: Dimension of input embeddings
            hidden_dim: Hidden dimension for plan representations
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Hierarchical planner
        self.planner = HierarchicalPlanner(
            input_dim, hidden_dim, max_steps=10, max_subgoals=5
        )
        
        # Plan evaluator
        self.plan_evaluator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Plan optimizer
        self.plan_optimizer = nn.GRUCell(hidden_dim * 2, hidden_dim)
        
        # Plan memory
        self.register_buffer('plan_memory', torch.zeros(100, hidden_dim))
        self.register_buffer('goal_memory', torch.zeros(100, hidden_dim))
        self.memory_counter = 0
    
    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate a multi-step plan.
        
        Args:
            state: Initial state tensor of shape [batch_size, input_dim]
            goal: Goal tensor of shape [batch_size, input_dim]
            
        Returns:
            Dictionary with planning outputs
        """
        # Generate hierarchical plan
        plan_results = self.planner(state, goal)
        
        # Encode goal
        goal_embedding = self.planner.goal_encoder(goal)
        
        # Evaluate plan
        final_state_embedding = plan_results['states'][:, -1]
        plan_quality = self.plan_evaluator(
            torch.cat([final_state_embedding, goal_embedding], dim=-1)
        )
        
        # Store successful plans in memory
        if self.training and plan_quality.mean().item() > 0.8:
            idx = self.memory_counter % 100
            self.plan_memory[idx] = final_state_embedding.mean(dim=0).detach()
            self.goal_memory[idx] = goal_embedding.mean(dim=0).detach()
            self.memory_counter += 1
        
        # Add plan quality to results
        plan_results['plan_quality'] = plan_quality
        
        return plan_results
    
    def retrieve_similar_plan(self, goal: torch.Tensor, k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve similar plans from memory.
        
        Args:
            goal: Goal tensor of shape [batch_size, input_dim]
            k: Number of plans to retrieve
            
        Returns:
            Tuple of (similar_plans, similar_goals)
        """
        batch_size = goal.shape[0]
        
        # Encode goal
        goal_embedding = self.planner.goal_encoder(goal)
        
        # Compute similarity with goal memory
        similarity = F.cosine_similarity(
            goal_embedding.unsqueeze(1),
            self.goal_memory.unsqueeze(0),
            dim=-1
        )
        
        # Get top-k indices
        _, indices = torch.topk(similarity, k=min(k, self.memory_counter), dim=-1)
        
        # Gather similar plans and goals
        similar_plans = torch.gather(
            self.plan_memory.unsqueeze(0).expand(batch_size, -1, -1),
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        
        similar_goals = torch.gather(
            self.goal_memory.unsqueeze(0).expand(batch_size, -1, -1),
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        )
        
        return similar_plans, similar_goals
    
    def optimize_plan(self, state: torch.Tensor, goal: torch.Tensor, 
                     initial_plan: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Optimize a plan to improve its quality.
        
        Args:
            state: Initial state tensor of shape [batch_size, input_dim]
            goal: Goal tensor of shape [batch_size, input_dim]
            initial_plan: Initial plan of shape [batch_size, num_steps, input_dim]
            steps: Number of optimization steps
            
        Returns:
            Optimized plan
        """
        batch_size, num_steps, _ = initial_plan.shape
        
        # Encode state and goal
        state_embedding = self.planner.state_encoder(state)
        goal_embedding = self.planner.goal_encoder(goal)
        
        # Initialize optimized plan
        optimized_plan = initial_plan.clone()
        
        # Optimize plan
        for _ in range(steps):
            # Execute plan to get final state
            current_state = state_embedding
            
            for i in range(num_steps):
                step_results = self.planner.plan_step(optimized_plan[:, i], current_state)
                current_state = step_results['new_state']
            
            # Evaluate plan
            plan_quality = self.plan_evaluator(
                torch.cat([current_state, goal_embedding], dim=-1)
            )
            
            # Compute gradient of plan quality with respect to plan
            plan_quality.sum().backward(retain_graph=True)
            
            # Update plan
            with torch.no_grad():
                grad = optimized_plan.grad
                optimized_plan += 0.01 * grad
                optimized_plan.grad.zero_()
        
        return optimized_plan
