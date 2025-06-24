# ===========================
# Memory Components
# ===========================




class SparseHopfieldNetwork:
    def __init__(self, size: int, sparsity: float = 0.1):
        self.size = size
        self.sparsity = sparsity
        self.weights = np.zeros((size, size))
        self.create_sparse_connections()

    def create_sparse_connections(self):
        num_connections = int(self.size * self.size * self.sparsity)
        for _ in range(num_connections):
            i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            while i == j or self.weights[i, j] != 0:
                i, j = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            self.weights[i, j] = 1
            self.weights[j, i] = 1  # Ensure symmetry

    def train(self, patterns: List[List[int]]):
        for p in patterns:
            p = np.array(p)
            outer_product = np.outer(p, p)
            self.weights += outer_product
        self.weights = self.weights / len(patterns)
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern: List[int], steps: int = 10) -> np.ndarray:
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
        return pattern



class ModernContinuousHopfieldNetwork(nn.Module):
    """
    Modern continuous Hopfield network with improved storage capacity and retrieval.
    Based on the paper "Hopfield Networks is All You Need" (Ramsauer et al., 2020).
    """
    def __init__(self, hidden_dim: int, beta: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta  # Inverse temperature parameter
        
        # Projection layers
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Stored patterns
        self.register_buffer('stored_keys', torch.zeros(0, hidden_dim))
        self.register_buffer('stored_values', torch.zeros(0, hidden_dim))
        
    def store(self, patterns: torch.Tensor, values: Optional[torch.Tensor] = None):
        """
        Store patterns in the Hopfield network.
        
        Args:
            patterns: Tensor of patterns to store [num_patterns, hidden_dim]
            values: Optional tensor of associated values [num_patterns, hidden_dim]
        """
        # Project patterns to key space
        keys = self.key_projection(patterns)
        
        # If values are not provided, use patterns as values
        if values is None:
            values = patterns
            
        # Project values
        values = self.value_projection(values)
        
        # Store patterns and values
        if self.stored_keys.size(0) == 0:
            self.stored_keys = keys
            self.stored_values = values
        else:
            self.stored_keys = torch.cat([self.stored_keys, keys], dim=0)
            self.stored_values = torch.cat([self.stored_values, values], dim=0)
    
    def retrieve(self, query: torch.Tensor, num_iterations: int = 1) -> torch.Tensor:
        """
        Retrieve patterns from the Hopfield network.
        
        Args:
            query: Query tensor [batch_size, hidden_dim]
            num_iterations: Number of retrieval iterations
            
        Returns:
            Retrieved patterns [batch_size, hidden_dim]
        """
        # Project query
        query = self.query_projection(query)
        
        # Initialize state with query
        state = query
        
        # Perform iterative retrieval
        for _ in range(num_iterations):
            # Compute attention weights
            similarities = torch.matmul(state, self.stored_keys.t()) * self.beta
            attention_weights = F.softmax(similarities, dim=-1)
            
            # Update state
            state = torch.matmul(attention_weights, self.stored_values)
        
        return state
    
    def forward(self, query: torch.Tensor, store_query: bool = False) -> torch.Tensor:
        """
        Forward pass: retrieve patterns and optionally store the query.
        
        Args:
            query: Query tensor [batch_size, hidden_dim]
            store_query: Whether to store the query in memory
            
        Returns:
            Retrieved patterns [batch_size, hidden_dim]
        """
        # Store query if requested
        if store_query:
            self.store(query)
            
        # Retrieve patterns
        return self.retrieve(query)
