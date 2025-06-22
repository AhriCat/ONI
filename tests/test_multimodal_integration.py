import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_MM_attention import MultiModalAttention
from modules.oni_audio import MiniAudioModule
from modules.oni_vision import MiniVisionTransformerWithIO
from modules.oni_NLP import OptimizedNLPModule

class TestMultimodalIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create sample dimensions
        self.dim = 896
        self.num_heads = 8
        
        # Create mock modules
        self.mock_nlp_module = MagicMock(spec=OptimizedNLPModule)
        self.mock_vision_module = MagicMock(spec=MiniVisionTransformerWithIO)
        self.mock_audio_module = MagicMock(spec=MiniAudioModule)
        
        # Set up return values for mock modules
        self.mock_nlp_module.forward.return_value = (torch.randn(2, 10, self.dim), 0.5)
        self.mock_vision_module.forward.return_value = (torch.randn(2, self.dim), torch.tensor(0.6))
        self.mock_audio_module.forward.return_value = (torch.randn(2, self.dim), torch.tensor(0.7))
        
    def test_multimodal_attention_initialization(self):
        """Test that the multimodal attention module initializes correctly."""
        attention = MultiModalAttention(self.dim)
        
        # Check that the module has the expected attributes
        self.assertEqual(attention.q_proj.in_features, self.dim)
        self.assertEqual(attention.q_proj.out_features, self.dim)
        self.assertEqual(attention.k_proj.in_features, self.dim)
        self.assertEqual(attention.k_proj.out_features, self.dim)
        self.assertEqual(attention.v_proj.in_features, self.dim)
        self.assertEqual(attention.v_proj.out_features, self.dim)
        
    def test_multimodal_attention_forward(self):
        """Test the forward method of the multimodal attention module."""
        attention = MultiModalAttention(self.dim)
        
        # Create dummy inputs
        x_nlp = torch.randn(2, 10, self.dim)
        x_vision = torch.randn(2, 5, self.dim)
        x_audio = torch.randn(2, 3, self.dim)
        
        # Call forward
        output = attention.forward(x_nlp, x_vision, x_audio)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10, self.dim))
        
    def test_multimodal_integration(self):
        """Test the integration of NLP, vision, and audio modules."""
        # Create multimodal attention module
        attention = MultiModalAttention(self.dim)
        
        # Create dummy inputs
        text_input = "This is a test input"
        image_input = torch.randn(1, 3, 224, 224)
        audio_input = torch.randn(1, 16000)
        
        # Process inputs through respective modules
        with patch.object(self.mock_nlp_module, 'forward') as mock_nlp_forward:
            mock_nlp_forward.return_value = (torch.randn(1, 10, self.dim), 0.5)
            nlp_output, nlp_energy = self.mock_nlp_module.forward(text_input)
        
        with patch.object(self.mock_vision_module, 'forward') as mock_vision_forward:
            mock_vision_forward.return_value = (torch.randn(1, self.dim), torch.tensor(0.6))
            vision_output, vision_energy = self.mock_vision_module.forward(image_input)
        
        with patch.object(self.mock_audio_module, 'forward') as mock_audio_forward:
            mock_audio_forward.return_value = (torch.randn(1, self.dim), torch.tensor(0.7))
            audio_output, audio_energy = self.mock_audio_module.forward(audio_input)
        
        # Reshape outputs for attention
        vision_output = vision_output.unsqueeze(1)  # [1, 1, dim]
        audio_output = audio_output.unsqueeze(1)    # [1, 1, dim]
        
        # Apply multimodal attention
        combined_output = attention.forward(nlp_output, vision_output, audio_output)
        
        # Check output shape
        self.assertEqual(combined_output.shape, (1, 10, self.dim))
        
        # Check that the modules were called with the correct inputs
        self.mock_nlp_module.forward.assert_called_once_with(text_input)
        self.mock_vision_module.forward.assert_called_once_with(image_input)
        self.mock_audio_module.forward.assert_called_once_with(audio_input)
        
    def test_cross_modal_attention(self):
        """Test cross-modal attention between different modalities."""
        attention = MultiModalAttention(self.dim)
        
        # Create dummy inputs with different sequence lengths
        x_nlp = torch.randn(2, 10, self.dim)      # Text with 10 tokens
        x_vision = torch.randn(2, 5, self.dim)    # Vision with 5 regions
        x_audio = torch.randn(2, 3, self.dim)     # Audio with 3 segments
        
        # Mock the projection layers
        attention.q_proj = MagicMock()
        attention.q_proj.return_value = torch.randn(2, 10, self.dim)
        
        attention.k_proj = MagicMock()
        attention.k_proj.side_effect = [
            torch.randn(2, 5, self.dim),   # For vision
            torch.randn(2, 3, self.dim)    # For audio
        ]
        
        attention.v_proj = MagicMock()
        attention.v_proj.side_effect = [
            torch.randn(2, 5, self.dim),   # For vision
            torch.randn(2, 3, self.dim)    # For audio
        ]
        
        # Mock torch.bmm for attention calculation
        with patch('torch.bmm') as mock_bmm:
            # Mock for vision attention
            mock_bmm.side_effect = [
                torch.randn(2, 10, 5),     # Query-key product for vision
                torch.randn(2, 10, self.dim)  # Attention-value product for vision
            ]
            
            # Call forward
            with patch('torch.softmax', return_value=torch.randn(2, 10, 5)):
                output = attention.forward(x_nlp, x_vision, x_audio)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10, self.dim))
        
        # Check that the projection layers were called with the correct inputs
        attention.q_proj.assert_called_once_with(x_nlp)
        attention.k_proj.assert_any_call(x_vision)
        attention.k_proj.assert_any_call(x_audio)
        attention.v_proj.assert_any_call(x_vision)
        attention.v_proj.assert_any_call(x_audio)
        
    def test_multimodal_fusion(self):
        """Test fusion of multimodal information."""
        # Create multimodal attention module
        attention = MultiModalAttention(self.dim)
        
        # Create dummy features
        text_features = torch.randn(1, 10, self.dim)
        vision_features = torch.randn(1, 1, self.dim)
        audio_features = torch.randn(1, 1, self.dim)
        
        # Apply multimodal attention
        fused_features = attention.forward(text_features, vision_features, audio_features)
        
        # Check output shape
        self.assertEqual(fused_features.shape, (1, 10, self.dim))
        
        # Check that the output is different from the input
        # (indicating that fusion has occurred)
        self.assertFalse(torch.allclose(fused_features, text_features))
        
    def test_end_to_end_multimodal_processing(self):
        """Test end-to-end multimodal processing pipeline."""
        # Create multimodal attention module
        attention = MultiModalAttention(self.dim)
        
        # Create config for NLP module
        nlp_config = {
            "hidden_dim": self.dim,
            "num_heads": self.num_heads,
            "num_layers": 6,
            "vocab_size": 300000,
            "max_length": 4096,
            "dropout": 0.1,
            "pad_token_id": 0
        }
        
        # Create mock modules with more realistic behavior
        with patch('modules.oni_NLP.OptimizedNLPModule') as mock_nlp_class:
            mock_nlp = MagicMock()
            mock_nlp.forward.return_value = (torch.randn(1, 10, self.dim), 0.5)
            mock_nlp_class.return_value = mock_nlp
            
            with patch('modules.oni_vision.MiniVisionTransformerWithIO') as mock_vision_class:
                mock_vision = MagicMock()
                mock_vision.forward.return_value = (torch.randn(1, self.dim), torch.tensor(0.6))
                mock_vision_class.return_value = mock_vision
                
                with patch('modules.oni_audio.MiniAudioModule') as mock_audio_class:
                    mock_audio = MagicMock()
                    mock_audio.forward.return_value = (torch.randn(1, self.dim), torch.tensor(0.7))
                    mock_audio_class.return_value = mock_audio
                    
                    # Create modules
                    nlp_module = mock_nlp_class(nlp_config)
                    vision_module = mock_vision_class(3, 64, self.dim)
                    audio_module = mock_audio_class(3, self.dim)
                    
                    # Create dummy inputs
                    text_input = "This is a test input"
                    image_input = torch.randn(1, 3, 224, 224)
                    audio_input = torch.randn(1, 16000)
                    
                    # Process inputs
                    nlp_output, _ = nlp_module.forward(text_input)
                    vision_output, _ = vision_module.forward(image_input)
                    audio_output, _ = audio_module.forward(audio_input)
                    
                    # Reshape outputs for attention
                    vision_output = vision_output.unsqueeze(1)
                    audio_output = audio_output.unsqueeze(1)
                    
                    # Apply multimodal attention
                    fused_output = attention.forward(nlp_output, vision_output, audio_output)
                    
                    # Check output shape
                    self.assertEqual(fused_output.shape, (1, 10, self.dim))
                    
                    # Check that the modules were called with the correct inputs
                    nlp_module.forward.assert_called_once_with(text_input)
                    vision_module.forward.assert_called_once_with(image_input)
                    audio_module.forward.assert_called_once_with(audio_input)

if __name__ == '__main__':
    unittest.main()