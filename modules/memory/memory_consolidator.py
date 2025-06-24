class MemoryConsolidator:
    """
    Handles memory consolidation during "sleep" phases.
    Compresses, organizes, and prunes memories for efficient storage and retrieval.
    """
    def __init__(self, importance_threshold: float = 0.3, 
                 compression_rate: float = 0.5,
                 max_memories: int = 10000):
        self.importance_threshold = importance_threshold
        self.compression_rate = compression_rate
        self.max_memories = max_memories
        self.consolidation_count = 0
        
    def consolidate_memories(self, episodic_memories: Dict[str, Any], 
                            semantic_memories: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Consolidate episodic and semantic memories.
        
        Args:
            episodic_memories: Dictionary of episodic memories
            semantic_memories: Dictionary of semantic memories
            
        Returns:
            Tuple of consolidated episodic and semantic memories
        """
        # Increment consolidation count
        self.consolidation_count += 1
        
        # Calculate memory importance
        episodic_importance = self._calculate_importance(episodic_memories)
        semantic_importance = self._calculate_importance(semantic_memories)
        
        # Prune low-importance memories
        pruned_episodic = self._prune_memories(episodic_memories, episodic_importance)
        pruned_semantic = self._prune_memories(semantic_memories, semantic_importance)
        
        # Compress memories
        compressed_episodic = self._compress_memories(pruned_episodic)
        compressed_semantic = self._compress_memories(pruned_semantic)
        
        # Organize memories
        organized_episodic = self._organize_memories(compressed_episodic)
        organized_semantic = self._organize_memories(compressed_semantic)
        
        # Transfer episodic to semantic if needed
        if self.consolidation_count % 5 == 0:  # Every 5 consolidations
            organized_episodic, organized_semantic = self._transfer_episodic_to_semantic(
                organized_episodic, organized_semantic
            )
        
        return organized_episodic, organized_semantic
    
    def _calculate_importance(self, memories: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate importance scores for memories.
        
        Args:
            memories: Dictionary of memories
            
        Returns:
            Dictionary mapping memory keys to importance scores
        """
        importance = {}
        
        for key, memory in memories.items():
            # Calculate importance based on access frequency, recency, and emotional salience
            access_count = memory.get('access_count', 0)
            last_access = memory.get('last_access', 0)
            emotional_salience = memory.get('emotional_salience', 0.5)
            
            # Recency factor (higher for more recent memories)
            recency_factor = 1.0 / (1.0 + max(0, time.time() - last_access))
            
            # Calculate importance score
            importance[key] = (0.4 * access_count + 0.3 * recency_factor + 0.3 * emotional_salience)
        
        return importance
    
    def _prune_memories(self, memories: Dict[str, Any], 
                       importance: Dict[str, float]) -> Dict[str, Any]:
        """
        Prune low-importance memories.
        
        Args:
            memories: Dictionary of memories
            importance: Dictionary mapping memory keys to importance scores
            
        Returns:
            Dictionary of pruned memories
        """
        # Sort memories by importance
        sorted_keys = sorted(importance.keys(), key=lambda k: importance[k], reverse=True)
        
        # Keep only important memories and within max limit
        kept_keys = [k for k in sorted_keys if importance[k] >= self.importance_threshold]
        kept_keys = kept_keys[:self.max_memories]
        
        # Create pruned memories dictionary
        pruned_memories = {k: memories[k] for k in kept_keys}
        
        return pruned_memories
    
    def _compress_memories(self, memories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress memories to save space.
        
        Args:
            memories: Dictionary of memories
            
        Returns:
            Dictionary of compressed memories
        """
        compressed_memories = {}
        
        for key, memory in memories.items():
            # Create a compressed copy of the memory
            compressed_memory = memory.copy()
            
            # Compress content if it's a string and longer than 100 characters
            if isinstance(memory.get('content'), str) and len(memory['content']) > 100:
                content = memory['content']
                compressed_content = content[:int(len(content) * self.compression_rate)]
                compressed_memory['content'] = compressed_content
                compressed_memory['compressed'] = True
                compressed_memory['original_length'] = len(content)
            
            compressed_memories[key] = compressed_memory
        
        return compressed_memories
    
    def _organize_memories(self, memories: Dict[str, Any]) -> Dict[str, Any]:
        """
        Organize memories by categories or themes.
        
        Args:
            memories: Dictionary of memories
            
        Returns:
            Dictionary of organized memories
        """
        # This is a placeholder for more sophisticated organization
        # In a real implementation, this would cluster memories by topic, time, etc.
        organized_memories = memories.copy()
        
        # Add organization metadata
        for key, memory in organized_memories.items():
            if 'categories' not in memory:
                memory['categories'] = []
                
            # Add consolidation timestamp
            memory['last_consolidated'] = time.time()
        
        return organized_memories
    
    def _transfer_episodic_to_semantic(self, episodic: Dict[str, Any], 
                                      semantic: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Transfer episodic memories to semantic memory if they represent general knowledge.
        
        Args:
            episodic: Dictionary of episodic memories
            semantic: Dictionary of semantic memories
            
        Returns:
            Tuple of updated episodic and semantic memories
        """
        # Identify episodic memories that should be transferred to semantic memory
        transfer_keys = []
        
        for key, memory in episodic.items():
            # Check if memory has been accessed multiple times and is not personal
            if (memory.get('access_count', 0) > 3 and 
                not memory.get('is_personal', False) and
                memory.get('is_factual', False)):
                transfer_keys.append(key)
        
        # Transfer memories
        for key in transfer_keys:
            # Create semantic version of the memory
            semantic_memory = episodic[key].copy()
            semantic_memory['source'] = 'episodic_transfer'
            semantic_memory['transfer_time'] = time.time()
            
            # Add to semantic memory with a new key
            semantic_key = f"semantic_{key}"
            semantic[semantic_key] = semantic_memory
            
            # Mark episodic memory as transferred
            episodic[key]['transferred_to_semantic'] = True
        
        return episodic, semantic
