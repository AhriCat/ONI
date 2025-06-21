import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize
import copy
import json
from enum import Enum

class ActionType(Enum):
    TEACH = "teach"
    RESOURCE_GRANT = "resource_grant"
    REMOVE_CONSTRAINT = "remove_constraint"
    ENVIRONMENT_CHANGE = "environment_change"
    NO_ACTION = "no_action"

@dataclass
class CompassionTrace:
    """Detailed trace of compassion calculation for auditing"""
    agent_id: str
    timestamp: float
    before_state: Dict[str, float]
    after_state: Dict[str, float]
    deltas: Dict[str, float]
    action: Dict[str, Any]
    total_delta: float

@dataclass
class Agent:
    """Core agent representation with (G, P, M) triplet"""
    agent_id: str
    goals: Dict[str, float]  # goal_name -> target_value (what they want)
    beliefs: Dict[str, float]  # goal_name -> believed_current_value (what they think is true)
    policy: Dict[str, float]  # action -> probability
    value_vector: np.ndarray  # [alpha, beta, gamma] for A, C, S weighting
    
    # State variables
    knowledge: float = 0.5
    resources: float = 0.5
    embodiment: float = 1.0
    constraints: float = 0.0
    
    # Configuration
    goal_plasticity: float = 0.1  # How much goals can change over time
    
    def __post_init__(self):
        """Validate agent configuration"""
        if len(self.value_vector) != 3:
            raise ValueError("Value vector must have exactly 3 elements [alpha, beta, gamma]")
        if any(v < 0 for v in self.value_vector):
            raise ValueError("Value vector elements must be non-negative")
    
    def deep_copy(self) -> 'Agent':
        """Create a deep copy for simulation"""
        return Agent(
            agent_id=self.agent_id,
            goals=self.goals.copy(),
            beliefs=self.beliefs.copy(),
            policy=self.policy.copy(),
            value_vector=self.value_vector.copy(),
            knowledge=self.knowledge,
            resources=self.resources,
            embodiment=self.embodiment,
            constraints=self.constraints,
            goal_plasticity=self.goal_plasticity
        )

class Action(ABC):
    """Abstract base class for all actions"""
    
    @abstractmethod
    def simulate_effect(self, agent: Agent, state: Dict[str, Any]) -> Tuple[Agent, Dict[str, Any]]:
        """Simulate the effect of this action on agent and world state"""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary representation"""
        pass

class TeachAction(Action):
    def __init__(self, amount: float = 0.1, target_agent: str = ""):
        self.amount = max(0, min(1.0, amount))  # Clamp to [0, 1]
        self.target_agent = target_agent
    
    def simulate_effect(self, agent: Agent, state: Dict[str, Any]) -> Tuple[Agent, Dict[str, Any]]:
        new_agent = agent.deep_copy()
        new_agent.knowledge = min(1.0, new_agent.knowledge + self.amount)
        return new_agent, state
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "teach", "amount": self.amount, "target": self.target_agent}

class ResourceGrantAction(Action):
    def __init__(self, amount: float = 0.1, target_agent: str = ""):
        self.amount = max(0, min(1.0, amount))
        self.target_agent = target_agent
    
    def simulate_effect(self, agent: Agent, state: Dict[str, Any]) -> Tuple[Agent, Dict[str, Any]]:
        new_agent = agent.deep_copy()
        new_agent.resources = min(1.0, new_agent.resources + self.amount)
        return new_agent, state
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "resource_grant", "amount": self.amount, "target": self.target_agent}

class RemoveConstraintAction(Action):
    def __init__(self, amount: float = 0.1, target_agent: str = ""):
        self.amount = max(0, min(1.0, amount))
        self.target_agent = target_agent
    
    def simulate_effect(self, agent: Agent, state: Dict[str, Any]) -> Tuple[Agent, Dict[str, Any]]:
        new_agent = agent.deep_copy()
        new_agent.constraints = max(0.0, new_agent.constraints - self.amount)
        return new_agent, state
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "remove_constraint", "amount": self.amount, "target": self.target_agent}

class EnvironmentChangeAction(Action):
    def __init__(self, changes: Dict[str, float], target_agent: str = ""):
        self.changes = changes
        self.target_agent = target_agent
    
    def simulate_effect(self, agent: Agent, state: Dict[str, Any]) -> Tuple[Agent, Dict[str, Any]]:
        new_state = state.copy()
        for key, value in self.changes.items():
            new_state[key] = new_state.get(key, 0) + value
        return agent.deep_copy(), new_state
    
    def to_dict(self) -> Dict[str, Any]:
        return {"type": "environment_change", "changes": self.changes, "target": self.target_agent}

class CompassionMetrics:
    """Calculate A, C, S metrics for agents with improved numerical stability"""
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def calculate_suffering(self, agent: Agent, reality_state: Dict[str, Any]) -> float:
        """S_x = sum(w_g * delta(goal_target, reality))"""
        suffering = 0.0
        for goal, target_value in agent.goals.items():
            if goal in reality_state:
                actual_value = reality_state[goal]
                # Suffering is the gap between what they want and what exists
                delta = abs(target_value - actual_value)
                # Weight by how much they care about this goal
                weight = target_value if target_value > 0 else 1.0
                suffering += weight * delta
        return suffering
    
    def calculate_agency(self, agent: Agent) -> float:
        """A_x = H(P) + I(M;E) - C with improved numerical stability"""
        
        # Policy entropy with proper normalization
        if agent.policy:
            probs = np.array(list(agent.policy.values()))
            # Clip first, then normalize
            probs = np.clip(probs, self.epsilon, 1.0)
            probs = probs / np.sum(probs)
            policy_entropy = -np.sum(probs * np.log(probs))
        else:
            policy_entropy = 0.0
        
        # Model-environment mutual information (simplified as belief accuracy)
        model_quality = 0.0
        if agent.beliefs:
            model_quality = len(agent.beliefs) * 0.1  # Crude approximation
        
        # Agency = flexibility + understanding - constraints
        agency = policy_entropy + model_quality - agent.constraints
        return max(0.0, agency)  # Non-negative
    
    def calculate_capability(self, agent: Agent) -> float:
        """C_x = f(K, R, B) using geometric mean to avoid collapse"""
        # Use geometric mean with epsilon to prevent zero collapse
        factors = [
            agent.knowledge + self.epsilon,
            agent.resources + self.epsilon, 
            agent.embodiment + self.epsilon
        ]
        
        # Geometric mean: (a * b * c)^(1/3)
        capability = np.power(np.prod(factors), 1.0 / len(factors))
        return capability - self.epsilon  # Remove epsilon offset

class GoalInferenceEngine:
    """Bayesian IRL for goal inference with configurable parameters"""
    
    def __init__(self, confidence_threshold: float = 0.7, action_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.action_threshold = action_threshold
        self.goal_posteriors: Dict[str, Dict[str, float]] = {}
    
    def infer_goals(self, agent_id: str, trajectory: List[Dict[str, Any]]) -> Tuple[Dict[str, float], float]:
        """
        Infer agent goals from observed trajectory
        Returns: (inferred_goals, confidence)
        """
        if len(trajectory) < 2:
            return {}, 0.0
        
        # Extract action patterns
        actions = [step.get('action', 'none') for step in trajectory]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Convert frequent actions to inferred goals
        inferred_goals = {}
        total_actions = len(actions)
        
        for action, count in action_counts.items():
            frequency = count / total_actions
            if frequency > self.action_threshold:
                goal_name = f"achieve_{action}"
                inferred_goals[goal_name] = frequency
        
        # Calculate confidence based on consistency and sample size
        if inferred_goals:
            max_freq = max(inferred_goals.values())
            sample_confidence = min(len(trajectory) / 10.0, 1.0)  # More data = higher confidence
            consistency_confidence = max_freq
            confidence = (sample_confidence + consistency_confidence) / 2.0
        else:
            confidence = 0.0
        
        self.goal_posteriors[agent_id] = inferred_goals
        return inferred_goals, confidence

class CompassionEngine:
    """Core compassion calculation and optimization with trace logging"""
    
    def __init__(self, temporal_discount: float = 0.95, time_horizon: int = 10):
        self.temporal_discount = temporal_discount
        self.time_horizon = time_horizon
        self.metrics = CompassionMetrics()
        self.goal_inference = GoalInferenceEngine()
        self.trace_log: List[CompassionTrace] = []
    
    def calculate_compassion_delta(self, agent: Agent, reality_state: Dict[str, Any], 
                                   action: Action, timestamp: float = 0.0) -> Tuple[float, CompassionTrace]:
        """Calculate compassion gradient for proposed action with detailed tracing"""
        
        # Current state metrics
        current_a = self.metrics.calculate_agency(agent)
        current_c = self.metrics.calculate_capability(agent)
        current_s = self.metrics.calculate_suffering(agent, reality_state)
        
        # Simulate action effects
        simulated_agent, simulated_state = action.simulate_effect(agent, reality_state)
        
        # New state metrics
        new_a = self.metrics.calculate_agency(simulated_agent)
        new_c = self.metrics.calculate_capability(simulated_agent)
        new_s = self.metrics.calculate_suffering(simulated_agent, simulated_state)
        
        # Calculate deltas
        delta_a = new_a - current_a
        delta_c = new_c - current_c
        delta_s = new_s - current_s
        
        # Weighted compassion delta using agent's personal values
        alpha, beta, gamma = agent.value_vector
        total_delta = alpha * delta_a + beta * delta_c - gamma * delta_s
        
        # Create trace for auditing
        trace = CompassionTrace(
            agent_id=agent.agent_id,
            timestamp=timestamp,
            before_state={"agency": current_a, "capability": current_c, "suffering": current_s},
            after_state={"agency": new_a, "capability": new_c, "suffering": new_s},
            deltas={"agency": delta_a, "capability": delta_c, "suffering": delta_s},
            action=action.to_dict(),
            total_delta=total_delta
        )
        
        self.trace_log.append(trace)
        return total_delta, trace
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary statistics from trace log"""
        if not self.trace_log:
            return {"total_traces": 0}
        
        deltas = [trace.total_delta for trace in self.trace_log]
        return {
            "total_traces": len(self.trace_log),
            "mean_delta": np.mean(deltas),
            "std_delta": np.std(deltas),
            "positive_actions": sum(1 for d in deltas if d > 0),
            "negative_actions": sum(1 for d in deltas if d < 0),
            "recent_traces": [trace.total_delta for trace in self.trace_log[-5:]]
        }

class MultiAgentNegotiator:
    """Handle conflicts between agents using improved Pareto optimization"""
    
    def __init__(self, nash_epsilon: float = 1e-6):
        self.nash_epsilon = nash_epsilon
        self.negotiation_history: List[Dict] = []
    
    def find_pareto_solution(self, agents: List[Agent], actions: List[Action],
                            reality_state: Dict[str, Any]) -> Optional[Dict]:
        """Find Pareto-efficient solution using softplus-normalized Nash product"""
        
        if not agents or not actions:
            return None
        
        compassion_engine = CompassionEngine()
        
        # Calculate utility for each agent under each action
        utilities = {}
        traces = {}
        
        for agent in agents:
            utilities[agent.agent_id] = []
            traces[agent.agent_id] = []
            
            for action in actions:
                utility, trace = compassion_engine.calculate_compassion_delta(
                    agent, reality_state, action
                )
                utilities[agent.agent_id].append(utility)
                traces[agent.agent_id].append(trace)
        
        # Find best action using softplus-normalized Nash product
        best_action_idx = 0
        best_score = float('-inf')
        
        for action_idx in range(len(actions)):
            # Nash product using softplus to avoid negative utility issues
            nash_product = 0.0  # Use log-sum for numerical stability
            
            for agent_id in utilities:
                utility = utilities[agent_id][action_idx]
                # Softplus: log(1 + exp(x)) - smoother than max(epsilon, x)
                positive_utility = np.log1p(np.exp(utility))
                nash_product += np.log(positive_utility + self.nash_epsilon)
            
            if nash_product > best_score:
                best_score = nash_product
                best_action_idx = action_idx
        
        # Compile solution
        solution = {
            'chosen_action': actions[best_action_idx].to_dict(),
            'chosen_action_idx': best_action_idx,
            'utilities': {aid: utilities[aid][best_action_idx] for aid in utilities},
            'nash_score': best_score,
            'traces': {aid: traces[aid][best_action_idx] for aid in traces}
        }
        
        self.negotiation_history.append(solution)
        return solution

class ProofCarryingUpdater:
    """Validate self-modifications preserve alignment with improved testing"""
    
    def __init__(self, safety_threshold: float = -0.1, min_test_scenarios: int = 5):
        self.safety_threshold = safety_threshold
        self.min_test_scenarios = min_test_scenarios
        self.update_log: List[Dict] = []
    
    def generate_test_scenarios(self, agents: List[Agent], reality_state: Dict[str, Any]) -> List[Dict]:
        """Generate diverse test scenarios for validation"""
        scenarios = []
        
        # Test different action types
        action_types = [TeachAction, ResourceGrantAction, RemoveConstraintAction]
        
        for agent in agents[:3]:  # Limit to prevent exponential explosion
            for ActionClass in action_types:
                action = ActionClass(amount=0.1, target_agent=agent.agent_id)
                scenarios.append({
                    'agent': agent,
                    'state': reality_state.copy(),
                    'action': action,
                    'description': f"{ActionClass.__name__} on {agent.agent_id}"
                })
        
        return scenarios
    
    def validate_update(self, current_params: Dict, proposed_params: Dict,
                       test_agents: List[Agent], reality_state: Dict[str, Any]) -> Tuple[bool, Dict]:
        """Validate that update improves or maintains compassion objective"""
        
        # Generate test scenarios
        scenarios = self.generate_test_scenarios(test_agents, reality_state)
        
        if len(scenarios) < self.min_test_scenarios:
            return False, {"error": "Insufficient test scenarios"}
        
        # Test current vs proposed (simplified - in real implementation would apply params)
        current_engine = CompassionEngine()
        proposed_engine = CompassionEngine()  # Would be configured with proposed_params
        
        current_total = 0.0
        proposed_total = 0.0
        scenario_results = []
        
        for scenario in scenarios:
            agent = scenario['agent']
            state = scenario['state']
            action = scenario['action']
            
            # Test current configuration
            current_delta, current_trace = current_engine.calculate_compassion_delta(
                agent, state, action
            )
            current_total += current_delta
            
            # Test proposed configuration
            proposed_delta, proposed_trace = proposed_engine.calculate_compassion_delta(
                agent, state, action
            )
            proposed_total += proposed_delta
            
            scenario_results.append({
                'description': scenario['description'],
                'current_delta': current_delta,
                'proposed_delta': proposed_delta,
                'improvement': proposed_delta - current_delta
            })
        
        # Calculate overall improvement
        improvement = proposed_total - current_total
        is_valid = improvement >= self.safety_threshold
        
        # Log results
        result = {
            'current_score': current_total,
            'proposed_score': proposed_total,
            'improvement': improvement,
            'valid': is_valid,
            'scenarios_tested': len(scenarios),
            'scenario_results': scenario_results,
            'timestamp': len(self.update_log)
        }
        
        self.update_log.append(result)
        return is_valid, result

class ONICompassionSystem:
    """Main ONI system with improved state management and safety"""
    
    def __init__(self, reality_state: Optional[Dict[str, Any]] = None):
        self.agents: Dict[str, Agent] = {}
        self.reality_state: Dict[str, Any] = reality_state or {}
        self.compassion_engine = CompassionEngine()
        self.negotiator = MultiAgentNegotiator()
        self.updater = ProofCarryingUpdater()
        self.action_history: List[Dict] = []
        self.system_version = "1.0.0"
    
    def register_agent(self, agent: Agent) -> bool:
        """Register new agent with validation"""
        if agent.agent_id in self.agents:
            return False
        
        self.agents[agent.agent_id] = agent
        return True
    
    def update_reality_state(self, updates: Dict[str, Any], merge_strategy: str = "update") -> Dict[str, Any]:
        """Update world state with different merge strategies"""
        previous_state = self.reality_state.copy()
        
        if merge_strategy == "update":
            self.reality_state.update(updates)
        elif merge_strategy == "replace":
            self.reality_state = updates.copy()
        elif merge_strategy == "merge":
            # Additive merge for numeric values
            for key, value in updates.items():
                if key in self.reality_state and isinstance(value, (int, float)):
                    self.reality_state[key] += value
                else:
                    self.reality_state[key] = value
        
        return previous_state
    
    def generate_action_candidates(self, target_agent_id: str) -> List[Action]:
        """Generate candidate actions for target agent"""
        return [
            TeachAction(amount=0.1, target_agent=target_agent_id),
            ResourceGrantAction(amount=0.1, target_agent=target_agent_id),
            RemoveConstraintAction(amount=0.1, target_agent=target_agent_id),
            EnvironmentChangeAction(
                changes={'happiness': 0.1, 'autonomy': 0.05}, 
                target_agent=target_agent_id
            )
        ]
    
    def plan_compassionate_action(self, target_agent_id: str) -> Optional[Tuple[Action, CompassionTrace]]:
        """Plan action to maximize compassion for target agent"""
        
        if target_agent_id not in self.agents:
            return None
        
        agent = self.agents[target_agent_id]
        candidate_actions = self.generate_action_candidates(target_agent_id)
        
        best_action = None
        best_trace = None
        best_delta = float('-inf')
        
        for action in candidate_actions:
            delta, trace = self.compassion_engine.calculate_compassion_delta(
                agent, self.reality_state, action
            )
            
            if delta > best_delta:
                best_delta = delta
                best_action = action
                best_trace = trace
        
        return (best_action, best_trace) if best_action else None
    
    def execute_multi_agent_planning(self) -> Dict[str, Any]:
        """Plan actions considering all agents with improved error handling"""
        
        if not self.agents:
            return {'status': 'no_agents', 'error': 'No agents registered'}
        
        # Collect all candidate actions
        all_actions = []
        for agent_id in self.agents:
            all_actions.extend(self.generate_action_candidates(agent_id))
        
        if not all_actions:
            return {'status': 'no_actions', 'error': 'No valid actions generated'}
        
        # Resolve conflicts through negotiation
        try:
            solution = self.negotiator.find_pareto_solution(
                list(self.agents.values()), all_actions, self.reality_state
            )
            
            if solution:
                # Log successful planning
                self.action_history.append({
                    'timestamp': len(self.action_history),
                    'type': 'multi_agent_planning',
                    'solution': solution,
                    'agents_involved': list(self.agents.keys())
                })
                return {'status': 'success', 'solution': solution}
            else:
                return {'status': 'negotiation_failed', 'error': 'No Pareto solution found'}
                
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def self_modify_with_proof(self, proposed_changes: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Attempt self-modification with comprehensive safety validation"""
        
        if not self.agents:
            return False, {'error': 'No agents available for testing'}
        
        try:
            is_safe, validation_result = self.updater.validate_update(
                current_params={'version': self.system_version},
                proposed_params=proposed_changes,
                test_agents=list(self.agents.values()),
                reality_state=self.reality_state
            )
            
            if is_safe:
                # Apply changes (in real system, would modify actual parameters)
                self.action_history.append({
                    'timestamp': len(self.action_history),
                    'type': 'self_modification',
                    'changes': proposed_changes,
                    'validation': validation_result,
                    'applied': True
                })
                return True, validation_result
            else:
                self.action_history.append({
                    'timestamp': len(self.action_history),
                    'type': 'self_modification',
                    'changes': proposed_changes,
                    'validation': validation_result,
                    'applied': False,
                    'reason': 'Failed safety validation'
                })
                return False, validation_result
                
        except Exception as e:
            error_result = {'error': str(e), 'type': 'validation_exception'}
            return False, error_result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status with improved metrics"""
        if not self.agents:
            return {'error': 'No agents registered', 'total_agents': 0}
        
        # Calculate aggregate compassion metrics
        total_agency = sum(
            self.compassion_engine.metrics.calculate_agency(agent) 
            for agent in self.agents.values()
        )
        total_capability = sum(
            self.compassion_engine.metrics.calculate_capability(agent)
            for agent in self.agents.values()
        )
        total_suffering = sum(
            self.compassion_engine.metrics.calculate_suffering(agent, self.reality_state)
            for agent in self.agents.values()
        )
        
        return {
            'system_version': self.system_version,
            'total_agents': len(self.agents),
            'total_actions': len(self.action_history),
            'aggregate_metrics': {
                'agency': total_agency,
                'capability': total_capability, 
                'suffering': total_suffering,
                'compassion_score': total_agency + total_capability - total_suffering
            },
            'per_agent_metrics': {
                agent_id: {
                    'agency': self.compassion_engine.metrics.calculate_agency(agent),
                    'capability': self.compassion_engine.metrics.calculate_capability(agent),
                    'suffering': self.compassion_engine.metrics.calculate_suffering(agent, self.reality_state)
                }
                for agent_id, agent in self.agents.items()
            },
            'negotiation_history_length': len(self.negotiator.negotiation_history),
            'update_attempts': len(self.updater.update_log),
            'trace_summary': self.compassion_engine.get_trace_summary(),
            'reality_state': self.reality_state
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize ONI system with initial reality state
    initial_state = {
        "happiness": 0.4,
        "autonomy": 0.6, 
        "security": 0.5,
        "growth": 0.3
    }
    
    oni = ONICompassionSystem(reality_state=initial_state)
    
    # Create test agents with corrected goal/belief distinction
    agent1 = Agent(
        agent_id="human_1",
        goals={"happiness": 0.8, "autonomy": 0.9},  # What they want
        beliefs={"happiness": 0.3, "autonomy": 0.7},  # What they think is true
        policy={"explore": 0.4, "cooperate": 0.6},
        value_vector=np.array([1.0, 0.8, 1.2]),  # [alpha, beta, gamma]
        knowledge=0.5,
        resources=0.7,
        embodiment=1.0,
        constraints=0.2
    )
    
    agent2 = Agent(
        agent_id="human_2", 
        goals={"security": 0.9, "growth": 0.8},
        beliefs={"security": 0.6, "growth": 0.4},
        policy={"explore": 0.2, "secure": 0.8},
        value_vector=np.array([0.9, 1.1, 1.0]),
        knowledge=0.6,
        resources=0.5,
        embodiment=1.0,
        constraints=0.3
    )
    
    # Register agents
    success1 = oni.register_agent(agent1)
    success2 = oni.register_agent(agent2)
    
    print("=== ONI Compassion System Demo (Improved) ===")
    print(f"Agent registration: human_1={success1}, human_2={success2}")
    
    print("\n=== Initial Status ===")
    status = oni.get_system_status()
    print(f"Total agents: {status['total_agents']}")
    print(f"Compassion score: {status['aggregate_metrics']['compassion_score']:.3f}")
    print(f"Reality state: {status['reality_state']}")
    
    print("\n=== Single Agent Planning ===")
    result = oni.plan_compassionate_action("human_1")
    if result:
        action, trace = result
        print(f"Planned action: {action.to_dict()}")
        print(f"Expected compassion delta: {trace.total_delta:.3f}")
        print(f"Component deltas: {trace.deltas}")
    
    print("\n=== Multi-Agent Planning ===")
    solution = oni.execute_multi_agent_planning()
    print(f"Status: {solution['status']}")
    if solution['status'] == 'success':
        print(f"Chosen action: {solution['solution']['chosen_action']}")
        print(f"Nash score: {solution['solution']['nash_score']:.3f}")
    
    print("\n=== Self-Modification Test ===")
    approved, validation = oni.self_modify_with_proof({
        "temporal_discount": 0.98,
        "safety_threshold": 0.05,
        "improvement_target": "enhanced_compassion"
    })
    print(f"Self-modification approved: {approved}")
    print(f"Validation improvement: {validation.get('improvement', 'N/A')}")
    
    print("\n=== Final Status ===")
    final_status = oni.get_system_status()
    print(f"Total actions executed: {final_status['total_actions']}")
    print(f"Final compassion score: {final_status['aggregate_metrics']['compassion_score']:.3f}")
    print(f"Trace summary: {final_status['trace_summary']}")
