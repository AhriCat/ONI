from modules.attention.concept_similarity_attention import ConceptSimilarityMemoryAttention

class EpisodicEmbeddingLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, compression_rate: float = 0.9):
        super(EpisodicEmbeddingLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.compression_rate = compression_rate

        # Encoder for different media types (multi-modal)
        self.media_encoder = nn.ModuleDict({
            'text': nn.Linear(input_dim, output_dim),
            'image': nn.Linear(input_dim, output_dim),
            'audio': nn.Linear(input_dim, output_dim),
            # Add other media types as needed
        })

        # Chain compression mechanism
        self.compression_layer = nn.Linear(output_dim, int(output_dim * self.compression_rate))

        # Infinite space handler
        self.embeddings = []

    def forward(self, x, media_type: str):
        if media_type not in self.media_encoder:
            raise ValueError(f"Unsupported media type: {media_type}")

        # Encode input
        x = self.media_encoder[media_type](x)

        # Chain compression
        x = self.compression_layer(x)

        # Store embedding (infinite space handler)
        self.embeddings.append(x)

        return x

class EpisodicBuffer(nn.Module):
    """
    Working memory buffer that combines episodic and semantic memory.
    Acts as a temporary storage for ongoing cognitive processes.
    """
    def __init__(self, hidden_dim: int, buffer_size: int = 10, decay_rate: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.buffer_size = buffer_size
        self.decay_rate = decay_rate
        
        # Initialize buffer as a learnable parameter
        self.register_parameter(
            'buffer',
            nn.Parameter(torch.zeros(buffer_size, hidden_dim), requires_grad=True)
        )
        
        # Importance scores for each buffer item
        self.register_buffer('importance', torch.zeros(buffer_size))
        
        # Recency scores for each buffer item (timestamp)
        self.register_buffer('recency', torch.zeros(buffer_size))
        
        # Current time step
        self.time_step = 0
        
        # Attention mechanism for buffer access
        self.attention = ConceptSimilarityMemoryAttention(hidden_dim, memory_slots=buffer_size)
        
        # Gating mechanism for writing to buffer
        self.write_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, query: torch.Tensor, episodic_input: Optional[torch.Tensor] = None, 
                semantic_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Access the buffer using attention mechanism and optionally update it.
        
        Args:
            query: Query tensor for retrieving from buffer
            episodic_input: Optional new episodic memory to add to buffer
            semantic_input: Optional new semantic memory to add to buffer
            
        Returns:
            Retrieved memory from buffer
        """
        batch_size = query.size(0)
        
        # Apply decay to importance based on recency
        self._apply_decay()
        
        # Expand buffer for batch processing
        expanded_buffer = self.buffer.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Retrieve from buffer using attention
        retrieved, attention_weights = self.attention(
            query=query,
            key=expanded_buffer,
            value=expanded_buffer
        )
        
        # Update buffer if new inputs are provided
        if episodic_input is not None or semantic_input is not None:
            self._update_buffer(episodic_input, semantic_input, query)
            
        # Increment time step
        self.time_step += 1
        
        return retrieved
    
    def _apply_decay(self):
        """Apply decay to importance scores based on recency"""
        if self.time_step > 0:
            time_since_update = self.time_step - self.recency
            decay_factor = torch.exp(-self.decay_rate * time_since_update)
            self.importance = self.importance * decay_factor
    
    def _update_buffer(self, episodic_input: Optional[torch.Tensor], 
                      semantic_input: Optional[torch.Tensor],
                      query: torch.Tensor):
        """
        Update buffer with new episodic and semantic information
        
        Args:
            episodic_input: New episodic memory
            semantic_input: New semantic memory
            query: Current query for context
        """
        # Combine inputs if both are provided
        if episodic_input is not None and semantic_input is not None:
            combined_input = episodic_input + semantic_input
        elif episodic_input is not None:
            combined_input = episodic_input
        elif semantic_input is not None:
            combined_input = semantic_input
        else:
            return
            
        # Calculate importance of new input
        with torch.no_grad():
            # Compute similarity with query as a measure of importance
            similarity = F.cosine_similarity(combined_input, query, dim=-1)
            new_importance = similarity.mean()
            
            # Find index to update (lowest importance or oldest)
            if torch.all(self.importance > 0):
                # Buffer is full, replace least important item
                update_idx = torch.argmin(self.importance)
            else:
                # Buffer has empty slots, use the first one
                update_idx = torch.where(self.importance == 0)[0][0]
            
            # Calculate write gate - determines how much to update
            concat_input = torch.cat([combined_input, self.buffer[update_idx].unsqueeze(0)], dim=-1)
            write_strength = self.write_gate(concat_input).item()
            
            # Update buffer
            new_value = write_strength * combined_input + (1 - write_strength) * self.buffer[update_idx]
            self.buffer.data[update_idx] = new_value.squeeze(0)
            
            # Update importance and recency
            self.importance[update_idx] = new_importance
            self.recency[update_idx] = self.time_step
