# Example usage and testing
def test_quantum_memory_system():
    """Test the quantum-enhanced memory system."""
    # Initialize tokenizer (placeholder)
    class MockTokenizer:
        def encode(self, text):
            return [ord(c) for c in text]
    
    tokenizer = MockTokenizer()
    
    # Create quantum-enhanced memory system
    memory_system = QuantumEnhancedMemory(tokenizer)
    
    # Test quantum search
    print("Testing quantum memory search...")
    test_query = "hello world"
    results = memory_system.quantum_search_memories(test_query)
    print(f"Quantum search results: {results}")
    
    # Test memory update
    print("\nTesting memory update...")
    memory_system.update_memory(
        stm_data="test short term memory",
        semantic_data="test semantic information",
        episodic_key="test_episode_1"
    )
    
    # Test quantum stats
    print("\nQuantum memory statistics:")
    stats = memory_system.quantum_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test visualization
    print("\nTesting quantum memory visualization...")
    visualizer = QuantumMemoryVisualizer(memory_system)
    if memory_system.quantum_memory_index:
        sample_keys = list(memory_system.quantum_memory_index.keys())[:3]
        viz_data = visualizer.visualize_quantum_states(sample_keys)
        print(f"Visualization data: {viz_data}")
    
    print("\nQuantum memory system test completed!")

if __name__ == "__main__":
    test_quantum_memory_system()
