import unittest
import torch
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_NLP import OptimizedNLPModule
from modules.oni_Tokenizer import MultitokenBPETokenizer
from modules.oni_nlp_embeddings import EmbeddingModule
from modules.oni_nlp_transformer import TransformerEncoder
from modules.oni_nlp_generation import TextGenerator

class TestTextProcessing(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create a mock config
        self.config = {
            "hidden_dim": 896,
            "num_heads": 8,
            "num_layers": 6,
            "vocab_size": 300000,
            "max_length": 4096,
            "dropout": 0.1,
            "pad_token_id": 0
        }
        
        # Create a tokenizer for testing
        self.tokenizer = MultitokenBPETokenizer(
            vocab_size=self.config["vocab_size"],
            max_merges=1000,
            n_future_tokens=4
        )
        
        # Create sample text
        self.sample_text = "Hello, this is a test of the ONI text processing system."
        
    def test_tokenizer_initialization(self):
        """Test that the tokenizer initializes correctly."""
        self.assertEqual(self.tokenizer.vocab_size, 300000)
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.unk_token_id, 1)
        
    def test_tokenizer_encode_decode(self):
        """Test tokenizer encoding and decoding."""
        # Encode text
        encoded = self.tokenizer.encode(self.sample_text)
        
        # Check that encoded is a tensor or list
        self.assertTrue(isinstance(encoded, (list, torch.Tensor)))
        
        # Decode text
        decoded = self.tokenizer.decode(encoded)
        
        # Check that decoded text contains the original text (may not be exact due to tokenization)
        self.assertTrue(self.sample_text.lower() in decoded.lower())
        
    def test_tokenizer_batch_processing(self):
        """Test tokenizer batch processing."""
        texts = [self.sample_text, "Another test sentence.", "A third test sentence for good measure."]
        
        # Encode batch
        encoded = self.tokenizer.batch_encode(texts, None, None, None)
        
        # Check that encoded is a tensor with the right shape
        self.assertTrue(isinstance(encoded, torch.Tensor))
        self.assertEqual(encoded.shape[0], len(texts))
        
        # Decode batch
        decoded = self.tokenizer.batch_decode(encoded)
        
        # Check that we get the right number of decoded texts
        self.assertEqual(len(decoded), len(texts))
        
    @patch('torch.nn.Embedding')
    @patch('torch.nn.LayerNorm')
    def test_embedding_module(self, mock_layer_norm, mock_embedding):
        """Test the embedding module."""
        # Create embedding module
        embedding_module = EmbeddingModule(self.config)
        
        # Create dummy input
        input_ids = torch.randint(0, self.config["vocab_size"], (2, 10))
        
        # Mock the forward methods
        mock_embedding_instance = mock_embedding.return_value
        mock_embedding_instance.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        mock_layer_norm_instance = mock_layer_norm.return_value
        mock_layer_norm_instance.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        # Call forward
        with patch.object(embedding_module, 'token_embedding', mock_embedding_instance):
            with patch.object(embedding_module, 'layer_norm', mock_layer_norm_instance):
                output = embedding_module.forward(input_ids)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10, self.config["hidden_dim"]))
        
    @patch('modules.oni_nlp_attention.SelfAttentionBlock')
    @patch('modules.oni_nlp_feedforward.FeedForwardBlock')
    def test_transformer_encoder(self, mock_ff_block, mock_attn_block):
        """Test the transformer encoder."""
        # Create transformer encoder
        encoder = TransformerEncoder(self.config)
        
        # Create dummy input
        x = torch.randn(2, 10, self.config["hidden_dim"])
        
        # Mock the forward methods
        mock_attn_instance = mock_attn_block.return_value
        mock_attn_instance.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        mock_ff_instance = mock_ff_block.return_value
        mock_ff_instance.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        # Replace layers with mocks
        encoder.layers = [MagicMock() for _ in range(self.config["num_layers"])]
        for layer in encoder.layers:
            layer.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        # Call forward
        output = encoder.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10, self.config["hidden_dim"]))
        
        # Check that each layer was called
        for layer in encoder.layers:
            layer.assert_called_once()
        
    @patch('torch.nn.Linear')
    def test_text_generator(self, mock_linear):
        """Test the text generator."""
        # Create text generator
        generator = TextGenerator(self.config)
        
        # Create dummy hidden states
        hidden_states = torch.randn(2, 10, self.config["hidden_dim"])
        
        # Mock the forward method of the projection layer
        mock_linear_instance = mock_linear.return_value
        mock_linear_instance.return_value = torch.randn(2, 10, self.config["vocab_size"])
        
        # Call forward
        with patch.object(generator, 'generation_head', mock_linear_instance):
            output = generator.forward(hidden_states)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 10, self.config["vocab_size"]))
        
    @patch('modules.oni_nlp_embeddings.EmbeddingModule')
    @patch('modules.oni_nlp_transformer.TransformerEncoder')
    @patch('modules.oni_nlp_generation.TextGenerator')
    def test_nlp_module(self, mock_generator, mock_encoder, mock_embedding):
        """Test the NLP module."""
        # Create NLP module
        nlp_module = OptimizedNLPModule(self.config)
        
        # Create dummy input
        input_ids = torch.randint(0, self.config["vocab_size"], (2, 10))
        attention_mask = torch.ones(2, 10)
        
        # Mock the forward methods
        mock_embedding_instance = mock_embedding.return_value
        mock_embedding_instance.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        mock_encoder_instance = mock_encoder.return_value
        mock_encoder_instance.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        mock_generator_instance = mock_generator.return_value
        mock_generator_instance.return_value = torch.randn(2, 10, self.config["vocab_size"])
        
        # Replace components with mocks
        nlp_module.embedding = mock_embedding_instance
        nlp_module.transformer = mock_encoder_instance
        nlp_module.generator = mock_generator_instance
        
        # Mock layer norm and output projection
        nlp_module.layer_norm = MagicMock()
        nlp_module.layer_norm.return_value = torch.randn(2, 10, self.config["hidden_dim"])
        
        nlp_module.output_projection = MagicMock()
        nlp_module.output_projection.return_value = torch.randn(2, 10, self.config["vocab_size"])
        
        # Call forward
        logits, energy = nlp_module.forward(input_ids, attention_mask)
        
        # Check output shapes
        self.assertEqual(logits.shape, (2, 10, self.config["vocab_size"]))
        self.assertTrue(isinstance(energy, float))
        
        # Check that components were called
        mock_embedding_instance.assert_called_once()
        mock_encoder_instance.assert_called_once()
        nlp_module.layer_norm.assert_called_once()
        nlp_module.output_projection.assert_called_once()

if __name__ == '__main__':
    unittest.main()