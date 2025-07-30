
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import List, Tuple, Dict, Optional
import numpy as np
from collections import defaultdict
import math

# Sample vocabulary and corpus
vocab = ["<PAD>", "<UNK>", "<SOS>", "<EOS>", "the", "a", "an", "is", "are", "was", "were",
         "hello", "world", "this", "that", "simple", "example", "creating", "hypergraph",
         "based", "model", "can", "represent", "complex", "relationships", "node", "connect",
         "multiple", "nodes", "powerful", "tools", "nlp", "capture", "intricate", "dependencies"]

corpus = [
    "This is a simple example.",
    "Creating a hypergraph-based model.",
    "Hypergraphs can represent complex relationships.",
    "Each node in a hypergraph can connect to multiple nodes.",
    "Hypergraph-based models are powerful tools for NLP.",
    "These models can capture intricate dependencies.",
    "A hypergraph structure is more flexible than a simple graph.",
    "In NLP, hypergraphs can enhance the representation of text.",
    "They can improve the performance of language models.",
    "Advanced hypergraph models can capture higher-order relationships."
]

class Hypergraph:
    """Modern hypergraph representation with proper indexing and features."""

    def __init__(self, nodes: List[str], hyperedges: List[Tuple[str, ...]], word_to_index: Dict[str, int]):
        self.nodes = nodes
        self.hyperedges = hyperedges
        self.word_to_index = word_to_index
        self.num_nodes = len(nodes)
        self.num_hyperedges = len(hyperedges)

        # Create incidence matrix (nodes x hyperedges)
        self.incidence_matrix = self._create_incidence_matrix()
        self.node_features = self._create_node_features()

    def _create_incidence_matrix(self) -> torch.Tensor:
        """Create incidence matrix H where H[i,j] = 1 if node i is in hyperedge j."""
        matrix = torch.zeros(self.num_nodes, self.num_hyperedges)
        node_to_idx = {node: i for i, node in enumerate(self.nodes)}

        for j, hyperedge in enumerate(self.hyperedges):
            for node in hyperedge:
                if node in node_to_idx:
                    matrix[node_to_idx[node], j] = 1.0
        return matrix

    def _create_node_features(self) -> torch.Tensor:
        """Create node feature matrix."""
        features = torch.zeros(self.num_nodes, dtype=torch.long)
        for i, node in enumerate(self.nodes):
            features[i] = self.word_to_index.get(node, self.word_to_index.get("<UNK>", 1))
        return features

    def get_adjacency_matrix(self) -> torch.Tensor:
        """Compute node adjacency matrix from incidence matrix."""
        # A = H * H^T - D_v where D_v is degree matrix
        H = self.incidence_matrix
        A = torch.mm(H, H.t())
        # Remove self-loops (diagonal elements)
        A.fill_diagonal_(0)
        return A

class HypergraphConvLayer(nn.Module):
    """Hypergraph Convolution Layer based on HGNN formulation."""

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of hypergraph convolution.
        Args:
            x: Node features (num_nodes, in_features)
            incidence_matrix: Incidence matrix (num_nodes, num_hyperedges)
        """
        # Normalize incidence matrix
        H = incidence_matrix
        D_v = torch.diag(H.sum(dim=1))  # Node degree matrix
        D_e = torch.diag(H.sum(dim=0))  # Hyperedge degree matrix

        # Add small epsilon to avoid division by zero
        D_v_inv_sqrt = torch.diag(1.0 / (torch.sqrt(torch.diag(D_v)) + 1e-8))
        D_e_inv = torch.diag(1.0 / (torch.diag(D_e) + 1e-8))

        # Hypergraph Laplacian: L = D_v^(-1/2) * H * D_e^(-1) * H^T * D_v^(-1/2)
        L = torch.mm(torch.mm(torch.mm(D_v_inv_sqrt, H), D_e_inv),
                     torch.mm(H.t(), D_v_inv_sqrt))

        # Apply convolution
        x = torch.mm(L, x)
        x = self.linear(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x
class HypergraphAttention(nn.Module):
    """Multi-head attention for hypergraphs."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.size()

        # Project to Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim)

        return self.out_proj(attn_output)

class ModernMambaBlock(nn.Module):
    """Simplified Mamba-inspired state space block."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=3,
            bias=True,
            padding=1,
            groups=self.d_inner,
        )
        self.activation = nn.SiLU()
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # State space parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape

        # Input projection
        x_and_res = self.in_proj(x)  # (batch, length, 2 * d_inner)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)

        # Convolution
        x = x.transpose(-1, -2)  # (batch, d_inner, length)
        x = self.conv1d(x)
        x = x.transpose(-1, -2)  # (batch, length, d_inner)

        # Activation
        x = self.activation(x)

        # State space computation (simplified)
        A = -torch.exp(self.A_log.float())  # (d_state,)
        y = x * self.D  # Skip connection

        # Output projection
        y = y * self.activation(res)
        return self.out_proj(y)

class HypergraphDataset(Dataset):
    """Dataset for hypergraph-based text processing."""

    def __init__(self, texts: List[str], vocab: List[str], max_seq_len: int = 64, max_hyperedges: int = 128):
        self.texts = texts
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.max_hyperedges = max_hyperedges
        self.word_to_index = {word: i for i, word in enumerate(vocab)}
        self.hypergraphs = self._create_hypergraphs()

    def _create_hypergraphs(self) -> List[Hypergraph]:
        """Create hypergraphs from texts."""
        hypergraphs = []
        for text in self.texts:
            words = text.lower().split()
            # Pad or truncate to max_seq_len
            if len(words) > self.max_seq_len:
                words = words[:self.max_seq_len]
            else:
                words.extend(["<PAD>"] * (self.max_seq_len - len(words)))

            # Create hyperedges (sliding window + sentence-level connections)
            hyperedges = []
            # Bigram hyperedges
            for i in range(len(words) - 1):
                if words[i] != "<PAD>" and words[i+1] != "<PAD>":
                    hyperedges.append((words[i], words[i+1]))

            # Trigram hyperedges
            for i in range(len(words) - 2):
                if all(w != "<PAD>" for w in words[i:i+3]):
                    hyperedges.append(tuple(words[i:i+3]))

            # Limit number of hyperedges
            if len(hyperedges) > self.max_hyperedges:
                hyperedges = hyperedges[:self.max_hyperedges]

            hypergraph = Hypergraph(words, hyperedges, self.word_to_index)
            hypergraphs.append(hypergraph)

        return hypergraphs

    def __len__(self) -> int:
        return len(self.hypergraphs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        hypergraph = self.hypergraphs[idx]

        # Pad incidence matrix to max_hyperedges
        incidence_matrix = hypergraph.incidence_matrix
        current_hyperedges = incidence_matrix.size(1)

        if current_hyperedges < self.max_hyperedges:
            # Pad with zeros
            padding = torch.zeros(self.max_seq_len, self.max_hyperedges - current_hyperedges)
            incidence_matrix = torch.cat([incidence_matrix, padding], dim=1)

        return {
            'node_features': hypergraph.node_features,
            'incidence_matrix': incidence_matrix,
            'adjacency_matrix': hypergraph.get_adjacency_matrix(),
            'num_hyperedges': torch.tensor(min(current_hyperedges, self.max_hyperedges), dtype=torch.long)
        }
class ModernHypergraphNLP(nn.Module):
    """Modern hypergraph-based NLP model."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Hypergraph layers
        self.hypergraph_layers = nn.ModuleList([
            HypergraphConvLayer(embed_dim if i == 0 else hidden_dim, hidden_dim, dropout)
            for i in range(num_layers)
        ])

        # Attention and Mamba blocks
        self.attention_layers = nn.ModuleList([
            HypergraphAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.mamba_blocks = nn.ModuleList([
            ModernMambaBlock(hidden_dim)
            for _ in range(num_layers)
        ])

        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self,
        node_features: torch.Tensor,
        incidence_matrix: torch.Tensor,
        num_hyperedges: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        Args:
            node_features: Node feature indices (batch_size, num_nodes)
            incidence_matrix: Incidence matrix (batch_size, num_nodes, max_hyperedges)
            num_hyperedges: Actual number of hyperedges per sample (batch_size,)
            attention_mask: Attention mask (batch_size, num_nodes)
        """
        batch_size, num_nodes = node_features.shape

        # Embedding
        x = self.token_embedding(node_features)  # (batch_size, num_nodes, embed_dim)
        x = x + self.pos_embedding[:, :num_nodes, :]
        x = self.dropout(x)

        # Process each item in batch
        outputs = []
        for i in range(batch_size):
            node_emb = x[i]  # (num_nodes, embed_dim)
            inc_mat = incidence_matrix[i]  # (num_nodes, max_hyperedges)

            # Mask out padded hyperedges
            valid_hyperedges = num_hyperedges[i].item()
            if valid_hyperedges < inc_mat.size(1):
                inc_mat = inc_mat[:, :valid_hyperedges]

            # Skip if no hyperedges
            if inc_mat.size(1) == 0:
                outputs.append(node_emb)
                continue

            # Hypergraph convolution layers
            for j, (hg_layer, attn_layer, mamba_block) in enumerate(
                zip(self.hypergraph_layers, self.attention_layers, self.mamba_blocks)
            ):
                # Hypergraph convolution
                node_emb = hg_layer(node_emb, inc_mat)

                # Self-attention (add batch dimension)
                node_emb_batched = node_emb.unsqueeze(0)  # (1, num_nodes, hidden_dim)
                attn_out = attn_layer(node_emb_batched)
                node_emb = attn_out.squeeze(0)  # (num_nodes, hidden_dim)

                # Mamba block
                node_emb_batched = node_emb.unsqueeze(0)  # (1, num_nodes, hidden_dim)
                mamba_out = mamba_block(node_emb_batched)
                node_emb = mamba_out.squeeze(0)  # (num_nodes, hidden_dim)

            outputs.append(node_emb)

        # Stack outputs
        x = torch.stack(outputs, dim=0)  # (batch_size, num_nodes, hidden_dim)

        # Final processing
        x = self.layer_norm(x)
        logits = self.output_projection(x)  # (batch_size, num_nodes, vocab_size)

        return logits

    def generate(self, input_text: str, word_to_index: Dict[str, int],
                 index_to_word: Dict[int, str], max_length: int = 20) -> str:
        """Generate text given input."""
        self.eval()
        device = next(self.parameters()).device

        words = input_text.lower().split()
        original_len = len(words)

        # Pad to max_seq_len
        if len(words) > self.max_seq_len:
            words = words[:self.max_seq_len]
        else:
            words.extend(["<PAD>"] * (self.max_seq_len - len(words)))

        # Create hyperedges
        hyperedges = []
        # Bigram hyperedges
        for i in range(original_len - 1):
            hyperedges.append((words[i], words[i+1]))
        # Trigram hyperedges
        for i in range(original_len - 2):
            hyperedges.append(tuple(words[i:i+3]))

        hypergraph = Hypergraph(words, hyperedges, word_to_index)

        with torch.no_grad():
            node_features = hypergraph.node_features.unsqueeze(0).to(device)
            incidence_matrix = hypergraph.incidence_matrix.unsqueeze(0).to(device)

            # Pad incidence matrix to max_hyperedges (64)
            current_hyperedges = incidence_matrix.size(2)
            max_hyperedges = 64  # Should match dataset max_hyperedges

            if current_hyperedges < max_hyperedges:
                padding = torch.zeros(1, self.max_seq_len, max_hyperedges - current_hyperedges, device=device)
                incidence_matrix = torch.cat([incidence_matrix, padding], dim=2)
            elif current_hyperedges > max_hyperedges:
                incidence_matrix = incidence_matrix[:, :, :max_hyperedges]
                current_hyperedges = max_hyperedges

            num_hyperedges = torch.tensor([current_hyperedges], dtype=torch.long, device=device)

            logits = self.forward(node_features, incidence_matrix, num_hyperedges)

            # Get the last non-padding token for generation
            last_valid_idx = original_len - 1
            probs = F.softmax(logits[0, last_valid_idx, :], dim=-1)
            next_token_id = torch.multinomial(probs, 1).item()

            if next_token_id in index_to_word and index_to_word[next_token_id] not in ["<PAD>", "<UNK>"]:
                return input_text + " " + index_to_word[next_token_id]
            else:
                return input_text + " <UNK>"


def train_model(model: ModernHypergraphNLP, dataset: HypergraphDataset,
                num_epochs: int = 10, batch_size: int = 8, lr: float = 1e-4):
    """Training function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            node_features = batch['node_features'].to(device)
            incidence_matrix = batch['incidence_matrix'].to(device)
            num_hyperedges = batch['num_hyperedges'].to(device)

            # Shift for language modeling
            input_features = node_features[:, :-1]
            target_features = node_features[:, 1:]
            input_incidence = incidence_matrix[:, :-1, :]

            optimizer.zero_grad()

            logits = model(input_features, input_incidence, num_hyperedges)
            loss = criterion(logits.reshape(-1, model.vocab_size), target_features.reshape(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Create dataset with max_hyperedges parameter
    dataset = HypergraphDataset(corpus, vocab, max_seq_len=512, max_hyperedges=64)

    # Create model
    model = ModernHypergraphNLP(
        vocab_size=len(vocab),
        embed_dim=512,
        hidden_dim=1024,
        num_layers=8,
        num_heads=16,
        dropout=0.1,
        max_seq_len=512
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    train_model(model, dataset, num_epochs=5, batch_size=4)

    # Generate text
    word_to_index = dataset.word_to_index
    index_to_word = {i: word for word, i in word_to_index.items()}

    input_text = "this is a simple"
    generated = model.generate(input_text, word_to_index, index_to_word)
    print(f"Input: {input_text}")
    print(f"Generated: {generated}")
