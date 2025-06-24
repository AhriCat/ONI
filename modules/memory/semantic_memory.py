
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

class TextPatternFinder:
    def __init__(self, tokenizer, min_pattern_length: int = 3, max_pattern_length: int = 10, min_occurrences: int = 2):
        self.tokenizer = tokenizer
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.min_occurrences = min_occurrences
        self.corpus_patterns = defaultdict(list)
        self.ltm_patterns = defaultdict(list)
        self.hopfield_network = None

    def find_patterns(self, corpus: List[str], ltm: List[str]) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        self._find_patterns_in_text(corpus, self.corpus_patterns)
        self._find_patterns_in_text(ltm, self.ltm_patterns)
        return self.corpus_patterns, self.ltm_patterns

    def _find_patterns_in_text(self, text: List[str], pattern_dict: Dict[str, List[int]]):
        tokens = self.tokenizer.tokenize(' '.join(text))
        num_tokens = len(tokens)

        for i in range(num_tokens):
            for length in range(self.min_pattern_length, self.max_pattern_length + 1):
                if i + length > num_tokens:
                    break
                pattern = ' '.join(tokens[i:i+length])
                pattern_dict[pattern].append(i)

        # Remove patterns that do not meet the minimum occurrence threshold
        for pattern, occurrences in list(pattern_dict.items()):
            if len(occurrences) < self.min_occurrences:
                del pattern_dict[pattern]

    def consolidate_patterns(self) -> Dict[str, List[int]]:
        combined_patterns = {**self.corpus_patterns, **self.ltm_patterns}
        unique_patterns = {p: combined_patterns[p] for p in combined_patterns if p in self.corpus_patterns and p in self.ltm_patterns}
        return unique_patterns

    def use_hopfield_network(self, unique_patterns: Dict[str, List[int]]):
        pattern_vectors = [self._pattern_to_vector(p) for p in unique_patterns.keys()]
        self.hopfield_network = SparseHopfieldNetwork(size=len(unique_patterns))
        self.hopfield_network.train(pattern_vectors)

    def _pattern_to_vector(self, pattern: str) -> List[int]:
        pattern_tokens = self.tokenizer.tokenize(pattern)
        pattern_vector = [0] * self.tokenizer.vocab_size
        for token in pattern_tokens:
            index = self.tokenizer.token_to_id(token)
            if 0 <= index < self.tokenizer.vocab_size:
                pattern_vector[index] = 1
        return pattern_vector

    def update_hopfield_network(self, new_patterns: Dict[str, List[int]]):
        new_pattern_vectors = [self._pattern_to_vector(p) for p in new_patterns.keys()]
        self.hopfield_network.train(new_pattern_vectors)
