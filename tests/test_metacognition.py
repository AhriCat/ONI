import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_metacognition import PrincipleConflictGraph, AbductiveReasoning, AnalogicalReasoning, CausalInferenceEngine, MetaCognitionModule

class TestMetaCognition(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create sample parameters
        self.hidden_dim = 896
        self.num_principles = 10
        
    def test_principle_conflict_graph_initialization(self):
        """Test that the principle conflict graph initializes correctly."""
        graph = PrincipleConflictGraph(threshold=0.5)
        
        # Check that the graph has the expected attributes
        self.assertEqual(graph.threshold, 0.5)
        self.assertEqual(len(graph.conflict_graph), 0)
        self.assertEqual(len(graph.conflict_scores), 0)
        self.assertEqual(len(graph.conflict_history), 0)
        
    def test_principle_conflict_graph_add_conflict(self):
        """Test adding a conflict to the graph."""
        graph = PrincipleConflictGraph(threshold=0.5)
        
        # Add conflict
        principle_a = 1
        principle_b = 2
        score = 0.7
        context = torch.randn(10, self.hidden_dim)
        
        graph.add_conflict(principle_a, principle_b, score, context)
        
        # Check that conflict was added
        self.assertEqual(len(graph.conflict_graph), 2)
        self.assertIn(principle_b, graph.conflict_graph[principle_a])
        self.assertIn(principle_a, graph.conflict_graph[principle_b])
        
        # Check that conflict score was added
        key = (principle_a, principle_b)
        self.assertIn(key, graph.conflict_scores)
        self.assertEqual(graph.conflict_scores[key], score)
        
        # Check that conflict history was updated
        self.assertEqual(len(graph.conflict_history), 1)
        self.assertEqual(graph.conflict_history[0]["principles"], key)
        self.assertEqual(graph.conflict_history[0]["score"], score)
        
        # Test adding conflict below threshold
        graph.add_conflict(3, 4, 0.3, context)
        
        # Check that conflict was not added
        self.assertEqual(len(graph.conflict_graph), 2)
        self.assertNotIn(4, graph.conflict_graph.get(3, set()))
        self.assertNotIn(3, graph.conflict_graph.get(4, set()))
        
    def test_principle_conflict_graph_get_conflicts(self):
        """Test getting conflicts for a principle."""
        graph = PrincipleConflictGraph(threshold=0.5)
        
        # Add conflicts
        context = torch.randn(10, self.hidden_dim)
        graph.add_conflict(1, 2, 0.7, context)
        graph.add_conflict(1, 3, 0.8, context)
        graph.add_conflict(2, 3, 0.6, context)
        
        # Get conflicts for principle 1
        conflicts = graph.get_conflicts(1)
        
        # Check conflicts
        self.assertEqual(len(conflicts), 2)
        
        # Conflicts should be sorted by score (descending)
        self.assertEqual(conflicts[0][0], 3)  # Principle
        self.assertEqual(conflicts[0][1], 0.8)  # Score
        self.assertEqual(conflicts[1][0], 2)
        self.assertEqual(conflicts[1][1], 0.7)
        
    def test_principle_conflict_graph_get_all_conflicts(self):
        """Test getting all conflicts in the graph."""
        graph = PrincipleConflictGraph(threshold=0.5)
        
        # Add conflicts
        context = torch.randn(10, self.hidden_dim)
        graph.add_conflict(1, 2, 0.7, context)
        graph.add_conflict(1, 3, 0.8, context)
        graph.add_conflict(2, 3, 0.6, context)
        
        # Get all conflicts
        conflicts = graph.get_all_conflicts()
        
        # Check conflicts
        self.assertEqual(len(conflicts), 3)
        
        # Conflicts should be sorted by score (descending)
        self.assertEqual(conflicts[0][0], 1)  # Principle A
        self.assertEqual(conflicts[0][1], 3)  # Principle B
        self.assertEqual(conflicts[0][2], 0.8)  # Score
        
    def test_principle_conflict_graph_get_most_conflicted(self):
        """Test getting the most conflicted principles."""
        graph = PrincipleConflictGraph(threshold=0.5)
        
        # Add conflicts
        context = torch.randn(10, self.hidden_dim)
        graph.add_conflict(1, 2, 0.7, context)
        graph.add_conflict(1, 3, 0.8, context)
        graph.add_conflict(1, 4, 0.6, context)
        graph.add_conflict(2, 3, 0.6, context)
        
        # Get most conflicted principles
        most_conflicted = graph.get_most_conflicted_principles(top_k=2)
        
        # Check result
        self.assertEqual(len(most_conflicted), 2)
        
        # Principle 1 should be most conflicted (3 conflicts)
        self.assertEqual(most_conflicted[0][0], 1)
        self.assertEqual(most_conflicted[0][1], 3)
        
    def test_abductive_reasoning_initialization(self):
        """Test that the abductive reasoning module initializes correctly."""
        module = AbductiveReasoning(self.hidden_dim, num_hypotheses=8)
        
        # Check that the module has the expected attributes
        self.assertEqual(module.hidden_dim, self.hidden_dim)
        self.assertEqual(module.num_hypotheses, 8)
        self.assertTrue(hasattr(module, 'hypothesis_generator'))
        self.assertTrue(hasattr(module, 'hypothesis_evaluator'))
        self.assertEqual(module.hypothesis_memory.shape, (8, self.hidden_dim))
        self.assertEqual(module.hypothesis_scores.shape, (8,))
        self.assertEqual(module.hypothesis_count, 0)
        
    @patch('torch.nn.Linear')
    def test_abductive_reasoning_forward(self, mock_linear):
        """Test the forward method of the abductive reasoning module."""
        module = AbductiveReasoning(self.hidden_dim, num_hypotheses=8)
        
        # Create dummy input
        x = torch.randn(2, self.hidden_dim)
        
        # Mock the forward methods of components
        mock_linear_instance = mock_linear.return_value
        
        # For hypothesis generator
        mock_linear_instance.return_value = torch.randn(2, 8 * self.hidden_dim)
        
        # For hypothesis evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.return_value = torch.rand(2, 1)
        
        # Set up the mocks for the module's components
        module.hypothesis_generator = MagicMock()
        module.hypothesis_generator.return_value = torch.randn(2, 8 * self.hidden_dim)
        
        module.hypothesis_evaluator = mock_evaluator
        
        # Call forward
        with patch('torch.cat', return_value=torch.randn(2, self.hidden_dim * 2)):
            with patch('torch.argmax', return_value=torch.tensor([0, 0])):
                with patch('torch.gather', return_value=torch.randn(2, 1, self.hidden_dim)):
                    best_hypothesis, metadata = module.forward(x)
        
        # Check output shape
        self.assertEqual(best_hypothesis.shape, (2, self.hidden_dim))
        
        # Check metadata
        self.assertIn("all_hypotheses", metadata)
        self.assertIn("scores", metadata)
        self.assertIn("best_idx", metadata)
        self.assertIn("best_score", metadata)
        
        # Check that the expected components were called
        module.hypothesis_generator.assert_called_once()
        self.assertEqual(module.hypothesis_evaluator.call_count, 8)  # Called for each hypothesis
        
    def test_abductive_reasoning_retrieve_similar(self):
        """Test retrieving a similar hypothesis."""
        module = AbductiveReasoning(self.hidden_dim, num_hypotheses=8)
        
        # Create dummy input
        x = torch.randn(2, self.hidden_dim)
        
        # Add some hypotheses to memory
        module.hypothesis_memory[0] = torch.randn(self.hidden_dim)
        module.hypothesis_scores[0] = 0.8
        module.hypothesis_count = 1
        
        # Call retrieve_similar_hypothesis
        with patch('torch.nn.functional.cosine_similarity', return_value=torch.tensor([[0.9]])):
            with patch('torch.argmax', return_value=torch.tensor([0])):
                with patch('torch.gather', return_value=torch.tensor([0.9])):
                    hypothesis, similarity = module.retrieve_similar_hypothesis(x)
        
        # Check output shape
        self.assertEqual(hypothesis.shape, (2, self.hidden_dim))
        self.assertEqual(similarity.shape, (2,))
        
    def test_analogical_reasoning_initialization(self):
        """Test that the analogical reasoning module initializes correctly."""
        module = AnalogicalReasoning(self.hidden_dim, memory_size=100)
        
        # Check that the module has the expected attributes
        self.assertEqual(module.hidden_dim, self.hidden_dim)
        self.assertEqual(module.memory_size, 100)
        self.assertTrue(hasattr(module, 'source_encoder'))
        self.assertTrue(hasattr(module, 'target_encoder'))
        self.assertTrue(hasattr(module, 'mapping_network'))
        self.assertEqual(module.source_memory.shape, (100, self.hidden_dim))
        self.assertEqual(module.target_memory.shape, (100, self.hidden_dim))
        self.assertEqual(module.mapping_memory.shape, (100, self.hidden_dim))
        self.assertEqual(module.memory_count, 0)
        
    @patch('torch.nn.Linear')
    def test_analogical_reasoning_forward_with_target(self, mock_linear):
        """Test the forward method with a target."""
        module = AnalogicalReasoning(self.hidden_dim, memory_size=100)
        
        # Create dummy inputs
        source = torch.randn(2, self.hidden_dim)
        target = torch.randn(2, self.hidden_dim)
        
        # Mock the forward methods of components
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, self.hidden_dim)
        
        # Set up the mocks for the module's components
        module.source_encoder = MagicMock()
        module.source_encoder.return_value = torch.randn(2, self.hidden_dim)
        
        module.target_encoder = MagicMock()
        module.target_encoder.return_value = torch.randn(2, self.hidden_dim)
        
        module.mapping_network = MagicMock()
        module.mapping_network.return_value = torch.randn(2, self.hidden_dim)
        
        # Call forward with target
        with patch('torch.cat', return_value=torch.randn(2, self.hidden_dim * 2)):
            mapping, metadata = module.forward(source, target)
        
        # Check output shape
        self.assertEqual(mapping.shape, (2, self.hidden_dim))
        
        # Check metadata
        self.assertIn("source_encoded", metadata)
        self.assertIn("target_encoded", metadata)
        
        # Check that the expected components were called
        module.source_encoder.assert_called_once_with(source)
        module.target_encoder.assert_called_once_with(target)
        module.mapping_network.assert_called_once()
        
    @patch('torch.nn.functional.cosine_similarity')
    def test_analogical_reasoning_forward_without_target(self, mock_cosine_similarity):
        """Test the forward method without a target."""
        module = AnalogicalReasoning(self.hidden_dim, memory_size=100)
        
        # Create dummy input
        source = torch.randn(2, self.hidden_dim)
        
        # Add some mappings to memory
        module.source_memory[0] = torch.randn(self.hidden_dim)
        module.mapping_memory[0] = torch.randn(self.hidden_dim)
        module.memory_count = 1
        
        # Mock cosine_similarity
        mock_cosine_similarity.return_value = torch.tensor([[0.9]])
        
        # Set up the mocks for the module's components
        module.source_encoder = MagicMock()
        module.source_encoder.return_value = torch.randn(2, self.hidden_dim)
        
        # Call forward without target
        with patch('torch.argmax', return_value=torch.tensor([0])):
            with patch('torch.gather', return_value=torch.tensor([0.9])):
                mapping, metadata = module.forward(source)
        
        # Check output shape
        self.assertEqual(mapping.shape, (2, self.hidden_dim))
        
        # Check metadata
        self.assertIn("source_encoded", metadata)
        self.assertIn("similarity", metadata)
        
        # Check that the expected components were called
        module.source_encoder.assert_called_once_with(source)
        mock_cosine_similarity.assert_called_once()
        
    def test_causal_inference_engine_initialization(self):
        """Test that the causal inference engine initializes correctly."""
        module = CausalInferenceEngine(self.hidden_dim, max_variables=10)
        
        # Check that the module has the expected attributes
        self.assertEqual(module.hidden_dim, self.hidden_dim)
        self.assertEqual(module.max_variables, 10)
        self.assertTrue(hasattr(module, 'variable_encoder'))
        self.assertTrue(hasattr(module, 'graph_generator'))
        self.assertTrue(hasattr(module, 'intervention_predictor'))
        self.assertEqual(module.causal_graph.shape, (10, 10))
        self.assertEqual(module.variable_embeddings.shape, (10, self.hidden_dim))
        self.assertEqual(module.variable_count, 0)
        
    @patch('torch.nn.Linear')
    def test_causal_inference_engine_forward(self, mock_linear):
        """Test the forward method of the causal inference engine."""
        module = CausalInferenceEngine(self.hidden_dim, max_variables=10)
        
        # Create dummy input
        variables = torch.randn(2, 3, self.hidden_dim)  # Batch size 2, 3 variables
        
        # Mock the forward methods of components
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, self.hidden_dim)
        
        # Set up the mocks for the module's components
        module.variable_encoder = MagicMock()
        module.variable_encoder.return_value = torch.randn(2, 3, self.hidden_dim)
        
        module.graph_generator = MagicMock()
        module.graph_generator.return_value = torch.rand(2, 1)
        
        # Call forward
        with patch('torch.zeros', return_value=torch.zeros(2, 3, 3)):
            with patch('torch.cat', return_value=torch.randn(2, self.hidden_dim * 2)):
                causal_graph, metadata = module.forward(variables)
        
        # Check output shape
        self.assertEqual(causal_graph.shape, (2, 3, 3))
        
        # Check metadata
        self.assertIn("encoded_variables", metadata)
        
        # Check that the expected components were called
        module.variable_encoder.assert_called_once_with(variables)
        self.assertEqual(module.graph_generator.call_count, 6)  # Called for each variable pair
        
    @patch('torch.nn.Linear')
    def test_causal_inference_engine_forward_with_interventions(self, mock_linear):
        """Test the forward method with interventions."""
        module = CausalInferenceEngine(self.hidden_dim, max_variables=10)
        
        # Create dummy inputs
        variables = torch.randn(2, 3, self.hidden_dim)  # Batch size 2, 3 variables
        interventions = torch.randn(2, 2, self.hidden_dim)  # Batch size 2, 2 interventions
        
        # Mock the forward methods of components
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, self.hidden_dim)
        
        # Set up the mocks for the module's components
        module.variable_encoder = MagicMock()
        module.variable_encoder.return_value = torch.randn(2, 3, self.hidden_dim)
        
        module.graph_generator = MagicMock()
        module.graph_generator.return_value = torch.rand(2, 1)
        
        module.intervention_predictor = MagicMock()
        module.intervention_predictor.return_value = torch.randn(2, self.hidden_dim)
        
        # Call forward with interventions
        with patch('torch.zeros', return_value=torch.zeros(2, 3, 3)):
            with patch('torch.cat', return_value=torch.randn(2, self.hidden_dim * 2)):
                with patch('torch.zeros', return_value=torch.zeros(2, 2, 3, self.hidden_dim)):
                    effects, metadata = module.forward(variables, interventions)
        
        # Check output shape
        self.assertEqual(effects.shape, (2, 2, 3, self.hidden_dim))
        
        # Check metadata
        self.assertIn("causal_graph", metadata)
        self.assertIn("encoded_variables", metadata)
        
        # Check that the expected components were called
        module.variable_encoder.assert_called_once_with(variables)
        self.assertEqual(module.graph_generator.call_count, 6)  # Called for each variable pair
        self.assertEqual(module.intervention_predictor.call_count, 6)  # Called for each intervention-variable pair
        
    def test_metacognition_module_initialization(self):
        """Test that the metacognition module initializes correctly."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Check that the module has the expected attributes
        self.assertEqual(module.hidden_dim, self.hidden_dim)
        self.assertEqual(module.num_principles, 10)
        self.assertTrue(hasattr(module, 'self_reflection'))
        self.assertTrue(hasattr(module, 'confidence_estimation'))
        self.assertTrue(hasattr(module, 'layer_norm'))
        self.assertEqual(len(module.principles), 0)
        self.assertEqual(len(module.principle_descriptions), 0)
        self.assertTrue(hasattr(module, 'context_projection'))
        self.assertTrue(hasattr(module, 'adaptive_alignment'))
        self.assertIsInstance(module.conflict_graph, PrincipleConflictGraph)
        self.assertIsInstance(module.abductive_reasoning, AbductiveReasoning)
        self.assertIsInstance(module.analogical_reasoning, AnalogicalReasoning)
        self.assertIsInstance(module.causal_inference, CausalInferenceEngine)
        
    def test_metacognition_add_principle(self):
        """Test adding a principle to the module."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Create dummy principle
        principle_vector = torch.randn(self.hidden_dim)
        description = "Test principle"
        
        # Add principle
        module.add_principle(principle_vector, description)
        
        # Check that principle was added
        self.assertEqual(len(module.principles), 1)
        self.assertEqual(len(module.principle_descriptions), 1)
        self.assertEqual(module.principle_descriptions[0], description)
        
        # Check that principle weights were updated
        self.assertEqual(len(module.principle_weights), 1)
        self.assertEqual(module.principle_weights.item(), 1.0)
        
    def test_metacognition_contextual_conflict_score(self):
        """Test calculating contextual conflict score."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Create dummy principles and context
        principle_a = torch.randn(self.hidden_dim)
        principle_b = torch.randn(self.hidden_dim)
        context = torch.randn(self.hidden_dim)
        
        # Mock the forward methods of components
        module.context_projection = MagicMock()
        module.context_projection.side_effect = lambda x: x  # Identity function
        
        # Call contextual_conflict_score
        with patch('torch.nn.functional.normalize', side_effect=lambda x, p, dim: x):
            with patch('torch.sum', return_value=torch.tensor(0.5)):
                score = module.contextual_conflict_score(principle_a, principle_b, context)
        
        # Check output
        self.assertTrue(isinstance(score, torch.Tensor))
        
        # Check that the expected components were called
        self.assertEqual(module.context_projection.call_count, 2)  # Called for each principle
        
    def test_metacognition_detect_nuanced_conflicts(self):
        """Test detecting nuanced conflicts."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Add some principles
        module.add_principle(torch.randn(self.hidden_dim), "Principle 1")
        module.add_principle(torch.randn(self.hidden_dim), "Principle 2")
        module.add_principle(torch.randn(self.hidden_dim), "Principle 3")
        
        # Create dummy context
        context = torch.randn(self.hidden_dim)
        
        # Mock contextual_conflict_score
        module.contextual_conflict_score = MagicMock()
        module.contextual_conflict_score.return_value = torch.tensor(0.7)
        
        # Call detect_nuanced_conflicts
        conflicts = module.detect_nuanced_conflicts(context, threshold=0.5)
        
        # Check output
        self.assertEqual(len(conflicts), 3)  # 3 pairs of principles
        
        # Each conflict should be a tuple of (i, j, score)
        for conflict in conflicts:
            self.assertEqual(len(conflict), 3)
            self.assertTrue(isinstance(conflict[0], int))
            self.assertTrue(isinstance(conflict[1], int))
            self.assertEqual(conflict[2], 0.7)
        
        # Check that contextual_conflict_score was called for each pair
        self.assertEqual(module.contextual_conflict_score.call_count, 3)
        
    @patch('torch.nn.Linear')
    def test_metacognition_forward(self, mock_linear):
        """Test the forward method of the metacognition module."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Create dummy input
        x = torch.randn(2, self.hidden_dim)
        
        # Add some principles
        module.add_principle(torch.randn(self.hidden_dim), "Principle 1")
        module.add_principle(torch.randn(self.hidden_dim), "Principle 2")
        
        # Mock the forward methods of components
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, 1)
        
        module.self_reflection = mock_linear_instance
        module.confidence_estimation = mock_linear_instance
        module.adaptive_alignment = mock_linear_instance
        
        # Mock other methods
        module.select_reasoning_strategy = MagicMock()
        module.select_reasoning_strategy.return_value = torch.tensor([0, 1])
        
        module.apply_reasoning_strategy = MagicMock()
        module.apply_reasoning_strategy.return_value = torch.randn(2, self.hidden_dim)
        
        module.estimate_uncertainty = MagicMock()
        module.estimate_uncertainty.return_value = (torch.rand(2, 1), torch.rand(2, 1))
        
        module.detect_nuanced_conflicts = MagicMock()
        module.detect_nuanced_conflicts.return_value = [(0, 1, 0.7)]
        
        module.get_similar_reasoning = MagicMock()
        module.get_similar_reasoning.return_value = (torch.randn(2, self.hidden_dim), torch.tensor([0.9, 0.8]))
        
        module.update_reasoning_memory = MagicMock()
        
        # Call forward
        with patch('torch.tanh', return_value=torch.randn(2, self.hidden_dim)):
            with patch('torch.sigmoid', return_value=torch.rand(2, 1)):
                with patch('torch.stack', return_value=torch.randn(2, self.hidden_dim)):
                    with patch('torch.matmul', return_value=torch.randn(2, 2)):
                        with patch('torch.softmax', return_value=torch.rand(2, 2)):
                            output, confidence, conflicts, metadata = module.forward(x)
        
        # Check output shapes
        self.assertEqual(output.shape, (2, self.hidden_dim))
        self.assertEqual(confidence.shape, (2, 1))
        self.assertEqual(conflicts, [(0, 1, 0.7)])
        
        # Check metadata
        self.assertIn("strategy", metadata)
        self.assertIn("uncertainty", metadata)
        self.assertIn("principle_weights", metadata)
        self.assertIn("principle_descriptions", metadata)
        self.assertIn("similar_reasoning", metadata)
        self.assertIn("conflict_patterns", metadata)
        
        # Check that the expected components and methods were called
        module.select_reasoning_strategy.assert_called_once()
        module.apply_reasoning_strategy.assert_called_once()
        module.estimate_uncertainty.assert_called_once()
        module.detect_nuanced_conflicts.assert_called_once()
        module.get_similar_reasoning.assert_called_once()
        module.update_reasoning_memory.assert_called_once()
        
    def test_metacognition_get_metacognitive_state(self):
        """Test getting the metacognitive state."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Add some principles
        module.add_principle(torch.randn(self.hidden_dim), "Principle 1")
        module.add_principle(torch.randn(self.hidden_dim), "Principle 2")
        
        # Add some data to metacognitive state
        module.metacognitive_state["confidence_history"] = [0.8, 0.7, 0.9]
        module.metacognitive_state["reasoning_strategies"] = [0, 1, 2]
        module.metacognitive_state["uncertainty_history"] = [(0.1, 0.2), (0.2, 0.3)]
        
        # Mock conflict graph methods
        module.conflict_graph.get_all_conflicts = MagicMock()
        module.conflict_graph.get_all_conflicts.return_value = [(0, 1, 0.7)]
        
        module.conflict_graph.get_most_conflicted_principles = MagicMock()
        module.conflict_graph.get_most_conflicted_principles.return_value = [(1, 2)]
        
        module.conflict_graph.analyze_conflict_patterns = MagicMock()
        module.conflict_graph.analyze_conflict_patterns.return_value = {"patterns": [], "clusters": 0}
        
        # Get metacognitive state
        state = module.get_metacognitive_state()
        
        # Check state
        self.assertEqual(state["confidence"], [0.8, 0.7, 0.9])
        self.assertEqual(state["conflicts"], [(0, 1, 0.7)])
        self.assertEqual(state["most_conflicted_principles"], [(1, 2)])
        self.assertEqual(state["reasoning_strategies"], [0, 1, 2])
        self.assertEqual(state["uncertainty"], [(0.1, 0.2), (0.2, 0.3)])
        self.assertEqual(len(state["principle_weights"]), 2)
        self.assertEqual(state["principle_descriptions"], ["Principle 1", "Principle 2"])
        self.assertEqual(state["conflict_patterns"], {"patterns": [], "clusters": 0})
        
    def test_metacognition_explain_reasoning(self):
        """Test explaining reasoning."""
        module = MetaCognitionModule(self.hidden_dim, num_principles=10)
        
        # Create dummy input
        x = torch.randn(2, self.hidden_dim)
        
        # Mock forward method
        module.forward = MagicMock()
        module.forward.return_value = (
            torch.randn(2, self.hidden_dim),
            torch.tensor([[0.8], [0.7]]),
            [(0, 1, 0.7)],
            {
                "strategy": 1,
                "uncertainty": {"aleatoric": 0.1, "epistemic": 0.2},
                "principle_weights": [0.6, 0.4],
                "principle_descriptions": ["Principle 1", "Principle 2"],
                "similar_reasoning": 0.9,
                "conflict_patterns": {"patterns": [], "clusters": 0}
            }
        )
        
        # Call explain_reasoning
        explanation = module.explain_reasoning(x)
        
        # Check explanation
        self.assertIn("reasoning_strategy", explanation)
        self.assertEqual(explanation["reasoning_strategy"]["name"], "Abductive")
        
        self.assertIn("confidence", explanation)
        self.assertAlmostEqual(explanation["confidence"]["score"], 0.8, places=5)
        
        self.assertIn("uncertainty", explanation)
        self.assertEqual(explanation["uncertainty"]["aleatoric"], 0.1)
        self.assertEqual(explanation["uncertainty"]["epistemic"], 0.2)
        
        self.assertIn("principles", explanation)
        self.assertEqual(len(explanation["principles"]["active_principles"]), 2)
        
        self.assertIn("conflicts", explanation)
        self.assertEqual(len(explanation["conflicts"]["detected"]), 1)

if __name__ == '__main__':
    unittest.main()