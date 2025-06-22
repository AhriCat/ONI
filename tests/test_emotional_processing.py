import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_emotions import EmotionalState, EmotionalLayer, EnergyModule, EmotionalFeedbackModule, EmotionalEnergyModel

class TestEmotionalProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create sample parameters
        self.hidden_dim = 896
        self.init_energy = 100
        
        # Create mock for classifier
        self.mock_classifier = MagicMock()
        self.mock_classifier.return_value = [
            {"label": "joy", "score": 0.8},
            {"label": "admiration", "score": 0.1},
            {"label": "approval", "score": 0.05},
            {"label": "excitement", "score": 0.05}
        ]
        
    def test_emotional_state_initialization(self):
        """Test that the emotional state initializes correctly."""
        state = EmotionalState()
        
        # Check that the state has the expected attributes
        self.assertIsNone(state.current_emotion)
        self.assertEqual(state.emotion_intensity, 0.0)
        self.assertEqual(state.emotion_decay, 0.95)
        self.assertEqual(len(state.emotion_memory), 0)
        self.assertEqual(state.emotion_memory.maxlen, 5)
        
    def test_emotional_state_update(self):
        """Test updating the emotional state."""
        state = EmotionalState()
        
        # Update state
        state.update_state("joy", 0.8)
        
        # Check that state was updated
        self.assertEqual(state.current_emotion, "joy")
        self.assertEqual(state.emotion_intensity, 0.8)
        self.assertEqual(len(state.emotion_memory), 1)
        self.assertIsNone(state.emotion_memory[0])  # Previous emotion was None
        
        # Update state again
        state.update_state("sadness", 0.6)
        
        # Check that state was updated
        self.assertEqual(state.current_emotion, "sadness")
        self.assertEqual(state.emotion_intensity, 0.6)
        self.assertEqual(len(state.emotion_memory), 2)
        self.assertEqual(state.emotion_memory[1], "joy")  # Previous emotion was joy
        
    def test_emotional_state_decay(self):
        """Test decaying the emotional state."""
        state = EmotionalState()
        
        # Update state
        state.update_state("joy", 0.8)
        
        # Decay emotion
        state.decay_emotion()
        
        # Check that intensity was decayed
        self.assertEqual(state.current_emotion, "joy")
        self.assertEqual(state.emotion_intensity, 0.8 * 0.95)
        
    def test_emotional_layer_initialization(self):
        """Test that the emotional layer initializes correctly."""
        layer = EmotionalLayer(self.hidden_dim)
        
        # Check that the layer has the expected attributes
        self.assertEqual(layer.valence.item(), 0.0)
        self.assertEqual(layer.arousal.item(), 0.0)
        self.assertIsInstance(layer.emotional_state, EmotionalState)
        self.assertEqual(layer.emotion_embedding.num_embeddings, len(layer.EMOTION_VA_MAP))
        self.assertEqual(layer.emotion_embedding.embedding_dim, self.hidden_dim)
        
    def test_emotional_layer_map_emotion(self):
        """Test mapping emotion to valence-arousal space."""
        layer = EmotionalLayer(self.hidden_dim)
        
        # Test mapping for a few emotions
        joy_va = layer.map_emotion_to_valence_arousal("joy")
        self.assertEqual(joy_va, (0.8, 0.6))
        
        sadness_va = layer.map_emotion_to_valence_arousal("sadness")
        self.assertEqual(sadness_va, (-0.7, -0.3))
        
        fear_va = layer.map_emotion_to_valence_arousal("fear")
        self.assertEqual(fear_va, (-0.7, 0.7))
        
        # Test mapping for unknown emotion
        unknown_va = layer.map_emotion_to_valence_arousal("unknown_emotion")
        self.assertEqual(unknown_va, (0.0, 0.0))
        
    def test_emotional_layer_compute_influence(self):
        """Test computing emotional influence."""
        layer = EmotionalLayer(self.hidden_dim)
        
        # Create dummy input
        x = torch.randn(2, 10, self.hidden_dim)
        
        # Create emotion dictionary
        emotion_dict = {
            "joy": 0.8,
            "admiration": 0.1,
            "approval": 0.05,
            "excitement": 0.05
        }
        
        # Compute influence
        influence = layer.compute_emotion_influence(x, emotion_dict)
        
        # Check output shape
        self.assertEqual(influence.shape, (len(layer.EMOTION_VA_MAP),))
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(influence.sum().item(), 1.0, places=5)
        
    @patch('torch.nn.Parameter')
    def test_emotional_layer_forward(self, mock_parameter):
        """Test the forward method of the emotional layer."""
        # Create emotional layer
        layer = EmotionalLayer(self.hidden_dim)
        
        # Create dummy input
        x = torch.randn(2, 10, self.hidden_dim)
        
        # Create emotion input
        emotion_input = [
            {"label": "joy", "score": 0.8},
            {"label": "admiration", "score": 0.1},
            {"label": "approval", "score": 0.05},
            {"label": "excitement", "score": 0.05}
        ]
        
        # Mock Parameter
        mock_parameter.side_effect = lambda x: x
        
        # Set up mocks for the layer's components
        layer.compute_emotion_influence = MagicMock()
        layer.compute_emotion_influence.return_value = torch.tensor([0.8, 0.1, 0.05, 0.05, 0.0] + [0.0] * (len(layer.EMOTION_VA_MAP) - 5))
        
        layer.emotion_embedding = MagicMock()
        layer.emotion_embedding.return_value = torch.randn(1, self.hidden_dim)
        
        # Call forward
        output = layer.forward(x, emotion_input)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that the expected components were called
        layer.compute_emotion_influence.assert_called_once()
        
    def test_energy_module_initialization(self):
        """Test that the energy module initializes correctly."""
        module = EnergyModule(self.init_energy)
        
        # Check that the module has the expected attributes
        self.assertEqual(module.energy.item(), self.init_energy)
        self.assertEqual(module.max_energy, self.init_energy)
        self.assertEqual(module.recovery_rate, 0.1)
        
    def test_energy_module_forward(self):
        """Test the forward method of the energy module."""
        module = EnergyModule(self.init_energy)
        
        # Create dummy input
        x = torch.randn(2, 10, self.hidden_dim)
        
        # Create dummy emotional state
        emotional_state = EmotionalState()
        emotional_state.update_state("joy", 0.8)
        
        # Call forward
        output = module.forward(x, 0.5, 0.7, emotional_state)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that energy was updated
        self.assertLess(module.energy.item(), self.init_energy)
        
    def test_emotional_feedback_module_initialization(self):
        """Test that the emotional feedback module initializes correctly."""
        module = EmotionalFeedbackModule()
        
        # Check that the module has the expected attributes
        self.assertEqual(module.base_lr, 0.001)
        self.assertEqual(module.emotion_sensitivity, 0.5)
        self.assertEqual(len(module.feedback_history), 0)
        
    def test_emotional_feedback_compute_weight(self):
        """Test computing feedback weight."""
        module = EmotionalFeedbackModule()
        
        # Create dummy emotional state
        emotional_state = {
            "valence": 0.7,
            "intensity": 0.8,
            "current_emotion": "joy"
        }
        
        # Compute feedback weight
        feedback = module.compute_feedback_weight(emotional_state)
        
        # Check that feedback is a number
        self.assertTrue(isinstance(feedback, (int, float)))
        
        # Check that feedback is positive (for positive valence)
        self.assertGreater(feedback, 0)
        
    def test_emotional_feedback_apply(self):
        """Test applying emotional feedback."""
        module = EmotionalFeedbackModule()
        
        # Create dummy module to apply feedback to
        dummy_module = torch.nn.Linear(10, 10)
        
        # Create dummy emotional state
        emotional_state = {
            "valence": 0.7,
            "intensity": 0.8,
            "current_emotion": "joy"
        }
        
        # Apply feedback
        module.apply_emotional_feedback(dummy_module, emotional_state)
        
        # Check that feedback history was updated
        self.assertEqual(len(module.feedback_history), 1)
        self.assertEqual(module.feedback_history[0]["emotion"], "joy")
        
    def test_emotional_feedback_get_stats(self):
        """Test getting feedback statistics."""
        module = EmotionalFeedbackModule()
        
        # Create some feedback history
        module.feedback_history = [
            {"emotion": "joy", "feedback_weight": 0.3, "intensity": 0.8},
            {"emotion": "sadness", "feedback_weight": -0.2, "intensity": 0.6},
            {"emotion": "joy", "feedback_weight": 0.4, "intensity": 0.9}
        ]
        
        # Get stats
        stats = module.get_feedback_stats()
        
        # Check stats
        self.assertAlmostEqual(stats["average_feedback"], (0.3 - 0.2 + 0.4) / 3, places=5)
        self.assertEqual(stats["strongest_positive"]["emotion"], "joy")
        self.assertEqual(stats["strongest_positive"]["feedback_weight"], 0.4)
        self.assertEqual(stats["strongest_negative"]["emotion"], "sadness")
        self.assertEqual(stats["strongest_negative"]["feedback_weight"], -0.2)
        
    def test_emotional_energy_model_initialization(self):
        """Test that the emotional energy model initializes correctly."""
        with patch('modules.oni_emotions.classifier', self.mock_classifier):
            model = EmotionalEnergyModel(self.hidden_dim, self.init_energy)
            
            # Check that the model has the expected attributes
            self.assertIsInstance(model.emotion_layer, EmotionalLayer)
            self.assertIsInstance(model.energy_module, EnergyModule)
            self.assertEqual(model.classifier, self.mock_classifier)
            self.assertIsInstance(model.feedback_module, EmotionalFeedbackModule)
            
    def test_emotional_energy_get_state(self):
        """Test getting the emotional state."""
        with patch('modules.oni_emotions.classifier', self.mock_classifier):
            model = EmotionalEnergyModel(self.hidden_dim, self.init_energy)
            
            # Set up the emotional state
            model.emotion_layer.emotional_state.current_emotion = 5  # Index for "joy"
            model.emotion_layer.emotional_state.emotion_intensity = 0.8
            model.emotion_layer.valence.data = torch.tensor(0.7)
            model.emotion_layer.arousal.data = torch.tensor(0.6)
            model.energy_module.energy.data = torch.tensor(90.0)
            
            # Get state
            state = model.get_emotional_state()
            
            # Check state
            self.assertEqual(state["current_emotion"], "joy")
            self.assertEqual(state["intensity"], 0.8)
            self.assertEqual(state["valence"], 0.7)
            self.assertEqual(state["arousal"], 0.6)
            self.assertEqual(state["energy"], 90.0)
            
    def test_emotional_energy_forward_text(self):
        """Test the forward method with text input."""
        with patch('modules.oni_emotions.classifier', self.mock_classifier):
            model = EmotionalEnergyModel(self.hidden_dim, self.init_energy)
            
            # Create dummy text input
            text_input = "This is a happy test message!"
            
            # Set up mocks
            model.classifier = self.mock_classifier
            model.emotion_layer = MagicMock()
            model.emotion_layer.return_value = torch.randn(1, self.hidden_dim)
            
            model.energy_module = MagicMock()
            model.energy_module.return_value = torch.randn(1, self.hidden_dim)
            
            model.get_emotional_state = MagicMock()
            model.get_emotional_state.return_value = {
                "current_emotion": "joy",
                "intensity": 0.8,
                "valence": 0.7,
                "arousal": 0.6,
                "energy": 90.0
            }
            
            model.feedback_module = MagicMock()
            model.feedback_module.get_feedback_stats.return_value = {
                "average_feedback": 0.3,
                "strongest_positive": {"emotion": "joy", "feedback_weight": 0.4},
                "strongest_negative": None
            }
            
            # Call forward with text
            output = model.forward(text_input)
            
            # Check output structure
            self.assertIn("classification", output)
            self.assertIn("emotional_state", output)
            self.assertIn("feedback_stats", output)
            
            # Check that the expected components were called
            self.mock_classifier.assert_called_once_with([text_input])
            model.emotion_layer.assert_called_once()
            model.feedback_module.apply_emotional_feedback.assert_called_once()
            model.get_emotional_state.assert_called_once()
            model.feedback_module.get_feedback_stats.assert_called_once()
            
    def test_emotional_energy_forward_tensor(self):
        """Test the forward method with tensor input."""
        with patch('modules.oni_emotions.classifier', self.mock_classifier):
            model = EmotionalEnergyModel(self.hidden_dim, self.init_energy)
            
            # Create dummy tensor input
            tensor_input = torch.randn(2, 10, self.hidden_dim)
            
            # Set up mocks
            model.emotion_layer = MagicMock()
            model.emotion_layer.return_value = torch.randn(2, 10, self.hidden_dim)
            
            model.energy_module = MagicMock()
            model.energy_module.return_value = torch.randn(2, 10, self.hidden_dim)
            
            # Call forward with tensor
            output = model.forward(tensor_input)
            
            # Check output shape
            self.assertEqual(output.shape, tensor_input.shape)
            
            # Check that the expected components were called
            model.emotion_layer.assert_called_once_with(tensor_input)
            model.energy_module.assert_called_once()

if __name__ == '__main__':
    unittest.main()