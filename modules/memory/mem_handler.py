from memory.volatile_memory import VolatileMemory
class MemoryInterferenceHandler:
    """
    Handles interference between conflicting memories.
    Resolves contradictions and manages memory integration.
    """
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.cache = VolatileMemory()
        self.contradiction_log = []
    
    def get_context(self, query):
        cached = self.cache.get(query)
        if cached is not None:
            return cached

        result = self.semantic.retrieve(query)
        self.cache.set(query, result)
        return result  
        
    def check_contradiction(self, memory1: Dict[str, Any], 
                           memory2: Dict[str, Any]) -> Tuple[bool, float]:
        """
        Check if two memories contradict each other.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Tuple of (is_contradiction, contradiction_score)
        """
        # This is a simplified implementation
        # In a real system, this would use NLP or other techniques to detect contradictions
        
        # Check if memories have the same subject but different facts
        if (memory1.get('subject') == memory2.get('subject') and
            memory1.get('fact') != memory2.get('fact')):
            
            # Calculate confidence for each memory
            confidence1 = memory1.get('confidence', 0.5)
            confidence2 = memory2.get('confidence', 0.5)
            
            # Calculate contradiction score
            contradiction_score = abs(confidence1 - confidence2)
            
            return True, contradiction_score
        
        return False, 0.0
    
    def resolve_contradiction(self, memory1: Dict[str, Any], 
                             memory2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve contradiction between two memories.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Resolved memory
        """
        # Check confidence levels
        confidence1 = memory1.get('confidence', 0.5)
        confidence2 = memory2.get('confidence', 0.5)
        
        # Log the contradiction
        self.contradiction_log.append({
            'time': time.time(),
            'memory1': memory1,
            'memory2': memory2,
            'confidence1': confidence1,
            'confidence2': confidence2
        })
        
        # If one memory has significantly higher confidence, prefer it
        if abs(confidence1 - confidence2) > self.confidence_threshold:
            if confidence1 > confidence2:
                resolved = memory1.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory2.get('id', 'unknown')
                return resolved
            else:
                resolved = memory2.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory1.get('id', 'unknown')
                return resolved
        
        # If recency matters, prefer the more recent memory
        if memory1.get('recency_matters', False) or memory2.get('recency_matters', False):
            timestamp1 = memory1.get('timestamp', 0)
            timestamp2 = memory2.get('timestamp', 0)
            
            if timestamp1 > timestamp2:
                resolved = memory1.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory2.get('id', 'unknown')
                return resolved
            else:
                resolved = memory2.copy()
                resolved['contradiction_resolved'] = True
                resolved['contradicted_with'] = memory1.get('id', 'unknown')
                return resolved
        
        # If we can't determine which is correct, create a merged memory with uncertainty
        merged = {
            'subject': memory1.get('subject'),
            'fact': f"Uncertain: {memory1.get('fact')} OR {memory2.get('fact')}",
            'confidence': min(confidence1, confidence2),
            'is_uncertain': True,
            'contradiction_sources': [memory1.get('id', 'unknown'), memory2.get('id', 'unknown')],
            'timestamp': max(memory1.get('timestamp', 0), memory2.get('timestamp', 0))
        }
        
        return merged
    
    def integrate_new_memory(self, new_memory: Dict[str, Any], 
                            existing_memories: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Integrate a new memory into existing memories, handling contradictions.
        
        Args:
            new_memory: New memory to integrate
            existing_memories: Dictionary of existing memories
            
        Returns:
            Updated memories dictionary
        """
        # Check for contradictions with existing memories
        contradictions = []
        
        for key, memory in existing_memories.items():
            is_contradiction, score = self.check_contradiction(new_memory, memory)
            if is_contradiction:
                contradictions.append((key, memory, score))
        
        # If no contradictions, simply add the new memory
        if not contradictions:
            memory_id = new_memory.get('id', f"memory_{len(existing_memories)}")
            existing_memories[memory_id] = new_memory
            return existing_memories
        
        # Handle contradictions
        updated_memories = existing_memories.copy()
        
        # Sort contradictions by score (highest first)
        contradictions.sort(key=lambda x: x[2], reverse=True)
        
        # Resolve the most significant contradiction
        key, memory, _ = contradictions[0]
        resolved_memory = self.resolve_contradiction(new_memory, memory)
        
        # Update or add the resolved memory
        if resolved_memory.get('id', None):
            updated_memories[resolved_memory['id']] = resolved_memory
        else:
            memory_id = f"resolved_{len(existing_memories)}"
            resolved_memory['id'] = memory_id
            updated_memories[memory_id] = resolved_memory
        
        # Mark or remove the contradicted memory
        if resolved_memory != memory:
            if self.confidence_threshold > 0.9:
                # High threshold - remove contradicted memory
                del updated_memories[key]
            else:
                # Lower threshold - mark as contradicted
                updated_memories[key]['contradicted'] = True
                updated_memories[key]['contradicted_by'] = resolved_memory.get('id')
        
        return updated_memories
