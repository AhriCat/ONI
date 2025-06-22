import unittest
import torch
import numpy as np
import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from modules.oni_memory import FadingMemorySystem, SnapshotMemorySystem, SpatialMemoryModule, Memory

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test method."""
        # Create sample parameters
        self.hidden_dim = 896
        self.decay_rate = 0.1
        self.memory_size = 100
        
        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
        
    def test_fading_memory_initialization(self):
        """Test that the fading memory system initializes correctly."""
        memory = FadingMemorySystem(self.hidden_dim, self.decay_rate)
        
        # Check that the memory has the expected attributes
        self.assertEqual(memory.hidden_dim, self.hidden_dim)
        self.assertEqual(memory.decay_rate, self.decay_rate)
        self.assertIsNone(memory.memory_state)
        
    def test_fading_memory_forward(self):
        """Test the forward method of the fading memory system."""
        memory = FadingMemorySystem(self.hidden_dim, self.decay_rate)
        
        # Create dummy input
        x = torch.randn(2, self.hidden_dim)
        
        # Call forward
        output = memory.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, x.shape)
        
        # Check that memory state was initialized
        self.assertIsNotNone(memory.memory_state)
        self.assertEqual(memory.memory_state.shape, x.shape)
        
        # Call forward again
        output2 = memory.forward(x)
        
        # Check that memory state was updated
        self.assertFalse(torch.allclose(output, output2))
        
    def test_snapshot_memory_initialization(self):
        """Test that the snapshot memory system initializes correctly."""
        memory = SnapshotMemorySystem(self.hidden_dim, self.memory_size)
        
        # Check that the memory has the expected attributes
        self.assertEqual(memory.hidden_dim, self.hidden_dim)
        self.assertEqual(memory.memory_size, self.memory_size)
        self.assertEqual(memory.current_index, 0)
        self.assertEqual(memory.memory_bank.shape, (self.memory_size, self.hidden_dim))
        
    def test_snapshot_memory_update(self):
        """Test updating the snapshot memory."""
        memory = SnapshotMemorySystem(self.hidden_dim, self.memory_size)
        
        # Create dummy snapshot
        snapshot = torch.randn(2, 5, self.hidden_dim)
        
        # Update memory
        memory.update(snapshot)
        
        # Check that memory was updated
        self.assertEqual(memory.current_index, 2)
        
        # Check that memory bank contains the snapshot mean
        expected_mean = snapshot.mean(dim=1)
        for i in range(2):
            self.assertTrue(torch.allclose(memory.memory_bank[i], expected_mean[i]))
            
    def test_snapshot_memory_get_snapshots(self):
        """Test getting snapshots from memory."""
        memory = SnapshotMemorySystem(self.hidden_dim, self.memory_size)
        
        # Create and add dummy snapshots
        for i in range(5):
            snapshot = torch.randn(2, 5, self.hidden_dim)
            memory.update(snapshot)
        
        # Get all snapshots
        snapshots = memory.get_snapshots()
        
        # Check output shape
        self.assertEqual(snapshots.shape, (memory.current_index, self.hidden_dim))
        
        # Get specific number of snapshots
        num_snapshots = 3
        snapshots = memory.get_snapshots(num_snapshots)
        
        # Check output shape
        self.assertEqual(snapshots.shape, (num_snapshots, self.hidden_dim))
        
    def test_spatial_memory_initialization(self):
        """Test that the spatial memory module initializes correctly."""
        room_size = (100, 100)
        overlap = 0.2
        max_memory = 10
        
        memory = SpatialMemoryModule(room_size, overlap, max_memory)
        
        # Check that the memory has the expected attributes
        self.assertEqual(memory.room_width, room_size[0])
        self.assertEqual(memory.room_height, room_size[1])
        self.assertEqual(memory.overlap, overlap)
        self.assertEqual(memory.current_position, (0, 0))
        self.assertEqual(len(memory.memory), 0)
        self.assertEqual(memory.max_memory, max_memory)
        
    def test_spatial_memory_get_room_key(self):
        """Test getting the current room key."""
        room_size = (100, 100)
        memory = SpatialMemoryModule(room_size)
        
        # Test with different positions
        memory.current_position = (0, 0)
        self.assertEqual(memory.get_current_room_key(), (0, 0))
        
        memory.current_position = (50, 50)
        self.assertEqual(memory.get_current_room_key(), (0, 0))
        
        memory.current_position = (150, 150)
        self.assertEqual(memory.get_current_room_key(), (1, 1))
        
    def test_spatial_memory_update_position(self):
        """Test updating position and detecting room changes."""
        room_size = (100, 100)
        memory = SpatialMemoryModule(room_size)
        
        # Test with position in same room
        memory.current_position = (0, 0)
        result = memory.update_position((50, 50))
        self.assertFalse(result)  # Should not detect room change
        self.assertEqual(memory.current_position, (50, 50))
        
        # Test with position in different room
        result = memory.update_position((150, 150))
        self.assertTrue(result)  # Should detect room change
        self.assertEqual(memory.current_position, (150, 150))
        
    def test_spatial_memory_load_get_room(self):
        """Test loading and getting room data."""
        room_size = (100, 100)
        memory = SpatialMemoryModule(room_size)
        
        # Create room data
        room_key = (0, 0)
        room_data = {"objects": ["chair", "table", "lamp"]}
        
        # Load room
        memory.load_room(room_key, room_data)
        
        # Check that room was loaded
        self.assertEqual(len(memory.memory), 1)
        self.assertIn(room_key, memory.memory)
        self.assertEqual(memory.memory[room_key], room_data)
        
        # Get room data
        memory.current_position = (50, 50)  # Position in room (0, 0)
        result = memory.get_current_room_data()
        
        # Check result
        self.assertEqual(result, room_data)
        
        # Test with position in non-existent room
        memory.current_position = (150, 150)  # Position in room (1, 1)
        result = memory.get_current_room_data()
        
        # Check result
        self.assertIsNone(result)
        
    def test_memory_initialization(self):
        """Test that the main Memory class initializes correctly."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Check that the memory has the expected attributes
        self.assertEqual(memory.tokenizer, tokenizer)
        self.assertEqual(memory.working_memory_capacity, 5)
        self.assertEqual(len(memory.working_memory), 0)
        self.assertEqual(len(memory.semantic_memory), 0)
        self.assertEqual(len(memory.context), 0)
        
    def test_memory_update_context(self):
        """Test updating the context."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Update context
        memory.update_context("location", "office")
        
        # Check that context was updated
        self.assertEqual(len(memory.context), 1)
        self.assertEqual(memory.context["location"], "office")
        
    def test_memory_get_context(self):
        """Test getting the context."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Update context
        memory.update_context("location", "office")
        memory.update_context("time", "morning")
        
        # Get context
        context = memory.get_context()
        
        # Check result
        self.assertEqual(len(context), 2)
        self.assertEqual(context["location"], "office")
        self.assertEqual(context["time"], "morning")
        
    @patch('os.path.exists')
    @patch('pickle.load')
    def test_memory_load_long_term_memory(self, mock_pickle_load, mock_path_exists):
        """Test loading long-term memory."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Mock os.path.exists
        mock_path_exists.return_value = True
        
        # Mock pickle.load
        mock_semantic_memory = {"word1": 1, "word2": 2}
        mock_ltm_summary = {"key1": {"data_type": "text", "path": "path1"}}
        mock_pickle_load.side_effect = [mock_semantic_memory, mock_ltm_summary]
        
        # Load long-term memory
        with patch('builtins.open', MagicMock()):
            memory.load_long_term_memory()
        
        # Check that memory was loaded
        self.assertEqual(memory.semantic_memory, mock_semantic_memory)
        self.assertEqual(memory.ltm_summary, mock_ltm_summary)
        
    @patch('os.makedirs')
    @patch('builtins.open')
    def test_memory_update_episodic_memory(self, mock_open, mock_makedirs):
        """Test updating episodic memory."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Create dummy data
        data = b"test data"
        data_type = "audio"
        key = "test_key"
        
        # Update episodic memory
        memory.update_episodic_memory(data, data_type, key)
        
        # Check that directory was created
        mock_makedirs.assert_called_once_with(memory.episodic_memory_path, exist_ok=True)
        
        # Check that file was written
        mock_open.assert_called_once()
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(data)
        
        # Check that LTM summary was updated
        self.assertEqual(len(memory.ltm_summary), 1)
        self.assertEqual(memory.ltm_summary[key]["data_type"], data_type)
        
    @patch('os.path.join')
    def test_memory_retrieve_from_episodic(self, mock_path_join):
        """Test retrieving from episodic memory."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Set up LTM summary
        key = "test_key"
        path = "test_path"
        memory.ltm_summary = {key: {"path": path}}
        
        # Mock os.path.join
        mock_path_join.return_value = path
        
        # Retrieve from episodic memory
        result = memory.retrieve_from_episodic(key)
        
        # Check result
        self.assertEqual(result, path)
        
    @patch('os.path.isdir')
    @patch('os.makedirs')
    @patch('builtins.open')
    def test_memory_update_semantic_memory(self, mock_open, mock_makedirs, mock_isdir):
        """Test updating semantic memory."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        
        # Create dummy data
        data = "test data"
        data_type = "text"
        key = "test_key"
        
        # Mock os.path.isdir
        mock_isdir.return_value = False
        
        # Update semantic memory
        memory.update_semantic_memory(data, data_type, key)
        
        # Check that directory was created
        mock_makedirs.assert_called_once_with(memory.semantic_memory_path, exist_ok=True)
        
        # Check that file was written
        mock_open.assert_called_once()
        mock_open.return_value.__enter__.return_value.write.assert_called_once_with(data)
        
        # Check that LTM summary was updated
        self.assertEqual(len(memory.ltm_summary), 1)
        self.assertEqual(memory.ltm_summary[key]["data_type"], data_type)
        
    def test_memory_update_memory(self):
        """Test updating all memory stores."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory
        memory = Memory(tokenizer)
        memory.save_long_term_memory = MagicMock()  # Mock to avoid file operations
        
        # Create dummy data
        stm_data = "short-term data"
        episodic_data = b"episodic data"
        episodic_key = "episodic_key"
        semantic_data = "semantic data"
        
        # Set up mocks
        memory.update_episodic_memory = MagicMock()
        memory.update_semantic_memory = MagicMock()
        
        # Update memory
        memory.update_memory(stm_data, episodic_data, episodic_key, semantic_data)
        
        # Check that working memory was updated
        self.assertEqual(len(memory.working_memory), 1)
        self.assertEqual(memory.working_memory[0], stm_data)
        
        # Check that episodic memory was updated
        memory.update_episodic_memory.assert_called_once_with(episodic_data, "audio", episodic_key)
        
        # Check that semantic memory was updated
        memory.update_semantic_memory.assert_called_once_with(semantic_data, "text", episodic_key)
        
        # Check that long-term memory was saved
        memory.save_long_term_memory.assert_called_once()
        
    def test_memory_working_memory_capacity(self):
        """Test that working memory respects capacity limits."""
        # Create mock tokenizer
        tokenizer = MagicMock()
        
        # Create memory with small capacity
        capacity = 3
        memory = Memory(tokenizer)
        memory.working_memory_capacity = capacity
        
        # Add items to working memory
        for i in range(5):
            memory.working_memory.append(f"item {i}")
        
        # Check that working memory respects capacity
        self.assertEqual(len(memory.working_memory), capacity)
        
        # Check that oldest items were removed
        self.assertEqual(memory.working_memory[0], "item 2")
        self.assertEqual(memory.working_memory[1], "item 3")
        self.assertEqual(memory.working_memory[2], "item 4")

if __name__ == '__main__':
    unittest.main()