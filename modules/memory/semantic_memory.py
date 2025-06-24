
class SemanticMemoryLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(SemanticMemoryLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Text encoder
        self.text_encoder = nn.Linear(input_dim, output_dim)

        # Shared connection to episodic layer
        self.episodic_embedding_layer = None  # Set externally

    def forward(self, text_input: torch.Tensor, media_reference: Optional[torch.Tensor] = None):
        # Encode text
        text_embedding = self.text_encoder(text_input)

        # If media reference is provided, connect with episodic embedding
        if media_reference is not None and self.episodic_embedding_layer is not None:
            media_embedding = self.episodic_embedding_layer(media_reference, media_type='image')  # Example for image
            combined_embedding = torch.cat((text_embedding, media_embedding), dim=-1)
            return combined_embedding

        return text_embedding
