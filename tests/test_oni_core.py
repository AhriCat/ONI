"""
Unit tests for Oni core functionality
"""
import pytest
import torch
from oni_core import OniCore, create_oni
from config.settings import DEFAULT_MODEL_CONFIG

class TestOniCore:
    """Test cases for OniCore"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = DEFAULT_MODEL_CONFIG.copy()
        self.config["vocab_size"] = 1000  # Smaller for testing
        
    def test_oni_initialization(self):
        """Test Oni initialization"""
        oni = create_oni(self.config)
        assert oni.initialized
        assert oni.device is not None
        
    def test_forward_pass(self):
        """Test forward pass"""
        oni = create_oni(self.config)
        result = oni.forward("Hello world")
        assert result["success"]
        assert "nlp_output" in result
        
    def test_response_generation(self):
        """Test response generation"""
        oni = create_oni(self.config)
        response = oni.generate_response("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
        
    def test_module_status(self):
        """Test module status checking"""
        oni = create_oni(self.config)
        status = oni.get_module_status()
        assert isinstance(status, dict)
        assert "nlp" in status
        
    def test_cleanup(self):
        """Test cleanup functionality"""
        oni = create_oni(self.config)
        oni.cleanup()  # Should not raise exception

if __name__ == "__main__":
    pytest.main([__file__])