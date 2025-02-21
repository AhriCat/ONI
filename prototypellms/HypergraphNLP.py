import torch
import torch.nn as nn
import torch.nn.functional as F

words = vocab
nodes = words

# Create the word_to_index dictionary
word_to_index = {word: i for i, word in enumerate(words)}

# Create the hyperedges
hyperedges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


class Hypergraph:
    def __init__(self, nodes, hyperedges):
        self.nodes = nodes
        self.hyperedges = hyperedges
        self.addnodes = self.nodes.extend(nodes)
        self.addedges = self.hyperedges.extend(hyperedges)
        self.num_nodes = len(nodes)
        self.num_hyperedges = len(hyperedges)
        self.num_edges = self.num_nodes * (self.num_nodes - 1) // 2
        self.num_vertices = self.num_nodes + self.num_hyperedges
        self.node_features = Hypergraph.create_node_features(nodes)



    def __len__(self):
        return self.num_nodes + self.num_hyperedges

    def __getitem__(self, idx):
        if idx < self.num_nodes:
            return self.nodes[idx]
        else:
            return self.hyperedges[idx - self.num_nodes]

    @staticmethod
    def create_node_features(nodes):
        node_features = torch.zeros(len(nodes))
        for i, node in enumerate(nodes):
            node_features[i] = word_to_index[node]
            if node not in word_to_index:
              node_features[i] = len(word_to_index)
        return node_features

    def set_adjacency_matrix(self, adjacency_matrix):
        self.adjacency_matrix = adjacency_matrix
        self.num_edges = adjacency_matrix.sum()
        self.num_vertices = self.num_nodes + self.num_hyperedges
        self.node_features = Hypergraph.create_node_features(self.nodes)
        self.adjacency_matrix = adjacency_matrix

class HypergraphDataset:
    def __init__(self, texts):
        self.texts = texts
        self.hypergraphs = self.create_hypergraphs(texts)
        hypergraph = self.text_to_hypergraph(texts[0])
        self.node_features = hypergraph.node_features
        self.adjacency_matrix = hypergraph.set_adjacency_matrix

    def __getitem__(self, idx):
        return self.hypergraphs[idx]

    def __len__(self):
        return len(self.texts)

    def text_to_hypergraph(self, text):
            # Build the word_to_index dictionary dynamically
            words = text.split()
            sentences = text.split('.')
            word_to_index = {word: i for i, word in enumerate(set(words))}

            # Convert words to indices
            node_features = [word_to_index[word] for word in words]

            # Create the adjacency matrix
            num_nodes = len(words)
            adjacency_matrix = torch.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(i+1, num_nodes):
                    adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
# Create the hypergraph
            hypergraph = Hypergraph(nodes, [(words[i], words[j]) for i in range(num_nodes) for j in range(i+1, num_nodes)])

            # Set the node features
            hypergraph.node_features = torch.tensor(node_features)

            return hypergraph

    def create_hypergraphs(self, texts):
        hypergraphs = []
        for text in texts:
            nodes = text.split()
            hyperedges = self.create_hyperedges(nodes)
            hypergraphs.append(Hypergraph(nodes, hyperedges))
        return hypergraphs

    def create_hyperedges(self, nodes):
        hyperedges = []
        for i in range(len(nodes) - 1):
            hyperedges.append((nodes[i], nodes[i + 1]))
        return hyperedges
    def get_relationships(self):
        relationships = []
        for hypergraph in self.hypergraphs:
            for node in hypergraph.nodes:
                for hyperedge in hypergraph.hyperedges:
                    relationships.append((node, hyperedge[0], hyperedge[1]))
            return relationships


texts = [
    "This is a simple example.",
    "Creating a hypergraph-based model.",
    "Hypergraphs can represent complex relationships.",
    "Each node in a hypergraph can connect to multiple nodes.",
    "Hypergraph-based models are powerful tools for NLP.",
    "These models can capture intricate dependencies.",
    "A hypergraph structure is more flexible than a simple graph.",
    "In NLP, hypergraphs can enhance the representation of text.",
    "They can improve the performance of language models.",
    "Advanced hypergraph models can capture higher-order relationships.",
    "Natural language processing benefits from complex models.",
    "Graph structures are essential for many machine learning tasks.",
    "Text data can be represented as graphs or hypergraphs.",
    "Hypergraphs offer a more expressive representation for text.",
    "Understanding hypergraphs is key to leveraging their power.",
    "Complex models can better capture the nuances of language.",
    "Language models require robust and flexible architectures.",
    "Hypergraph-based models can handle complex data scenarios.",

]




    # Add more texts as needed
dataset = HypergraphDataset(corpus + vocab)
hypergraphs = dataset.hypergraphs
hyper_vocab = []

for hypergraph in hypergraphs:
    print("Number of nodes:", hypergraph.num_nodes)
    print("Number of hyperedges:", hypergraph.num_hyperedges)
    print("Number of edges:", hypergraph.num_edges)
    print("Number of vertices:", hypergraph.num_vertices)
    print("Node features:", hypergraph.node_features)
    # print("Adjacency matrix:", hypergraph.adjacency_matrix)  # Uncomment if needed
    print("Nodes:", hypergraph.nodes)
    print("Hyperedges:", hypergraph.hyperedges)

    hyper_vocab.append(hypergraph.nodes)
    for hyperedge in hypergraph.hyperedges:
        hyper_vocab.append(hyperedge)

print("Hyper vocab:", hyper_vocab)

input_text = "test"

class HypergraphLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, bidirectional=False, dropout=0.):
        super(HypergraphLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.normalize = nn.SyncBatchNorm(input_dim)
        self.dropout = dropout
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.fc1 = nn.Linear(784, 128)  # input layer (28x28 images) -> hidden layer (128 units)
        self.fc2 = nn.Linear(128, 64)  # hidden layer (128 units) -> hidden layer (64 units)
        self.fc3 = nn.Linear(64, 32)  # hidden layer (64 units) -> hidden layer (32 units)
        self.fc4 = nn.Linear(32, 10)
        

    def forward(self, x, lengths):
        x = self.normalize(x)
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = torch.relu(self.fc2(x))  # activation function for hidden layer
        x = torch.relu(self.fc3(x))  # activation function for hidden layer
        x = self.fc4(x)
        # Embed the input
        x = self.embedding(x)

        # Pack the input sequence
        packed_input = pack_padded_sequence(x, lengths, batch_first=True)

        # Run the input through the LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_input)

        # Unpack the output sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # normalize output
        output = self.normalize(output)

        # Apply dropout
        output = F.dropout(output, p=self.dropout, training=self.training)

        # Apply the linear layer
        output = self.fc(output)

        # Reshape the output to match the expected shape
        output = output.view(-1, self.output_dim)

        return output

class MambaNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MambaNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.hypergraph_lstm = HypergraphLSTM(input_dim, hidden_dim, hidden_dim)
        self.mamba_layer = MambaLayer(hidden_dim, hidden_dim, output_dim)

    def forward(self, hypergraph, lengths):
        # Run the hypergraph through the Hypergraph LSTM
        output = self.hypergraph_lstm(hypergraph, lengths)

        # Run the output through the Mamba layer
        output = self.mamba_layer(output)

        return output

class MambaLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MambaLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
  
    def forward(self, x):
        # Apply the first linear layer
        x = F.relu(self.fc1(x))

        # Apply the second linear layer
        x = self.fc2(x)

        return x

class KANSpikingNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(KANSpikingNeuron, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, x):
        # Apply the first linear layer
        x = F.relu(self.fc1(x))

        # Apply the second linear layer
        x = self.fc2(x)

        # Convert the output to spikes using the KAN model
        spikes = torch.zeros_like(x)
        for t in range(x.shape[1]):
            spikes[:, t] = (x[:, t] > 0).float()

        return spikes
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.depth = embed_dim // num_heads

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        matmul_qk = torch.matmul(q, k.transpose(-2, -1))
        depth = k.size(-1)
        logits = matmul_qk / torch.sqrt(torch.tensor(depth, dtype=torch.float32))

        if mask is not None:
            logits += (mask * -1e9)

        attention_weights = F.softmax(logits, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.permute(0, 2, 1, 3).contiguous()
        output = output.view(batch_size, -1, self.embed_dim)

        output = self.dropout(output)
        return self.dense(output)
    
context_size = len(corpus)
embedding_dim = 128

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, input_dim, hidden_dim, output_dim, context_size):
        super(Transformer, self).__init__()
        self.context_size = context_size
        self.attention = MultiHeadSelfAttention(input_dim, hidden_dim)
        self.mamba = MambaLayer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim                                               
        self.embedding = HypergraphLSTM(embed_dim, context_size, output_dim)  # Initialize HypergraphLSTM with embed_dim and context_size
        self.encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.decoder = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.encoder_layer = nn.TransformerEncoder(self.encoder, num_layers=1)
        self.decoder_layer = nn.TransformerDecoder(self.decoder, num_layers=1)
        self.fc = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x):
        out = self.fc(out.squeeze(0))
        out = self.fc1(out.reshape(-1, 4096))
        out = self.decoder_layer(x)
        out = self.fc(out)
        return out
       # reshape out to (batch_size, 4096)

trans_model = Transformer(embed_dim=128, num_heads=8, input_dim=512, hidden_dim=256, output_dim=10, context_size=100)

class NLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NLPModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)  # example layer
        self.fc2 = nn.Linear(64, 10)  # example layer
        self.lstm1 = nn.LSTM(input_dim, hidden_dim)
        self.transformer = trans_model
        self.relu = torch.nn.LeakyReLU
        self.dataset = dataset
        self.hypergraph = hyper_vocab
        self.mamba_network = MambaNetwork(input_dim, hidden_dim, output_dim)
        self.device = torch.device("cuda:GPU 0" if torch.cuda.is_available() else "cpu")
        self.lookup_layer = lookup.create_lookup_layer
        self.vocab = vocab
        self.selfattention = MultiHeadSelfAttention(input_dim, hidden_dim)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.conversation_history = []
        self.input_dim = input_dim
        self.text_to_hypergraph = dataset.text_to_hypergraph(text=input_text)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
       # self.predict_next_statement = predict_next_statement
     #   self.get_sequences_from_dataset = get_sequences_from_dataset
        self.vocab_size = len(self.vocab)
        self.kanskp = KANSpikingNeuron(input_dim, hidden_dim, output_dim)
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.summary()

    def call(self, inputs):
        if self.lookup_layer is None:
            raise ValueError("Lookup layer not created. Call create_lookup_layer first.")
        return self.lookup_layer(inputs)

    def params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        print("NLP Model Summary:")
        print(f"Input Dimension: {self.input_dim}")
        print(f"Hidden Dimension: {self.hidden_dim}")
        print(f"Output Dimension: {self.output_dim}")
        print(f"Vocabulary Size: {self.vocab_size}")
        print(f"Embedding Size: {self.embedding.embedding_dim}")
        print(f"Mamba Network Input Dimension: {self.mamba_network.input_dim}")
        print(f"Mamba Network Hidden Dimension: {self.mamba_network.hidden_dim}")
        print(f"Mamba Network Output Dimension: {self.mamba_network.output_dim}")
        print(f"Params: {self.params}")
        print(f"Device: {self.device}")
        print(f"LSTM Input Dimension: {self.lstm1.input_size}")
        print(f"LSTM Hidden Dimension: {self.lstm1.hidden_size}")
        #print(f"LSTM Output Dimension: {self.lstm.output_dim}")
        print(f"LSTM Batch First: {self.lstm1.batch_first}")
        print(f"LSTM Bidirectional: {self.lstm1.bidirectional}")
        print(f"LSTM Num Layers: {self.lstm1.num_layers}")
        print(f"LSTM Input Dimension: {self.lstm2.input_size}")
        print(f"LSTM Hidden Dimension: {self.lstm2.hidden_size}")
        #print(f"LSTM Output Dimension: {self.lstm.output_dim}")
        print(f"LSTM Batch First: {self.lstm2.batch_first}")
        print(f"LSTM Bidirectional: {self.lstm2.bidirectional}")
        print(f"LSTM Num Layers: {self.lstm2.num_layers}")
        print(f"Transformer Input Dim: {self.transformer.input_dim}")
        print(f"Transformer Hidden Dim: {self.transformer.hidden_dim}")
        print(f"Transformer Output Dim: {self.transformer.output_dim}")
        print(f"KANSKP Input Dimension: {self.kanskp.input_dim}")
        print(f"KANSKP Hidden Dimension: {self.kanskp.hidden_dim}")
        print(f"KANSKP Output Dimension: {self.kanskp.output_dim}")
        print(f"KANSKP Params: {self.kanskp.params}")

    def forward(self, hypergraph, lengths):
        # Convert the hypergraph to a text
        for hypergraphs in dataset.hypergraphs:
            text = ''
            hypergraph = hypergraphs[0] + ' ' + hypergraphs[1] + ' '
            for node in hypergraphs.nodes:
                text += node + ' '
                for hyperedge in hypergraphs.hyperedges:
                      text += hyperedge[0] + ' ' + hyperedge[1] + ' '
                      x = self.text_to_hypergraph(text)
                      node_features = hypergraphs.create_node_features(nodes)
                      adjacency_matrix = hypergraphs.set_adjacency_matrix(hypergraphs.adjacency_matrix)
                      x = torch.cat((node_features, adjacency_matrix), dim=1)
                      x = x.view(1, -1)
                      h0 = torch.zeros(1, self.hidden_dim).to(self.device)
                      c0 = torch.zeros(1, self.hidden_dim).to(self.device)
                      output = self.mamba_network(x, lengths)
                      self.selfattention(output)
                      return output

    def get_hypergraph(self, text):
        return dataset.text_to_hypergraph(text)

    def get_sequences_from_dataset(self, dataset, batch_size):
        sequences = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            sequences.append(batch)
        return sequences

    def predict_next_statement(sequences):
        # Initialize the LSTM model
        sequences = text.split(".")

        lstm = self.lstm1(input_dim=len(sequences[0]), hidden_dim=128, num_layers=1)

        # Convert the sequences to tensors
        sequences_tensor = torch.tensor(sequences)

        # Initialize the hidden and cell states
        h0 = torch.zeros(1, 1, 128)
        c0 = torch.zeros(1, 1, 128)

        # Run the sequences through the LSTM
        out, _ = lstm(sequences_tensor, (h0, c0))

        # Get the output for the last time step
        last_output = out[-1, :, :]

        # Apply a linear layer to the output
        fc = nn.Linear(128, len(vocab))
        next_statement_logits = fc(last_output)

        # Predict the next statement using the softmax function
        next_statement_probs = torch.softmax(next_statement_logits, dim=1)
        next_statement_idx = torch.argmax(next_statement_probs, dim=1).item()

    # Return the predicted next statement
        return vocab[next_statement_idx]

    def memorize(self, text):
        self.conversation_history.append(text)
        self.ltm.append(self.conversation_history)
        print(self.ltm)

    def respond(self, input_text):
        # Ensure input_text is a string
        if not isinstance(input_text, str):
            raise TypeError("Input text must be a string.")
    
        # Ensure dataset has a text_to_hypergraph attribute
        if not hasattr(self.dataset, 'text_to_hypergraph'):
            raise AttributeError("Dataset object does not have a text_to_hypergraph attribute.")
    
        # Convert input text to a hypergraph
        input_hypergraph = self.dataset.text_to_hypergraph(input_text)
    
        # Initialize the hidden state and cell state for the LSTM
        h0 = torch.zeros(1, 1, self.hidden_dim).to(self.device)
        c0 = torch.zeros(1, 1, self.hidden_dim).to(self.device)
    
        # Initialize the output sequence
        outputs = []
    
        # Run the input hypergraph through the model
        x = self.embedding(torch.tensor([self.vocab_size - 1]).to(self.device))  # Start with a special "start" token
        x = x.view(1, 1, -1)
    
        for i in range(20):  # Generate a sequence of 20 words
            out, (h0, c0) = self.lstm2(x, (h0, c0))
            out = self.fc1(out.squeeze(0))
            out = self.softmax(out)
            out = self.selfattention(out)
            word_idx = torch.argmax(out, dim=1).item()
            if word_idx == self.vocab_size - 1:  # End with a special "end" token
                break
            outputs.append(self.vocab[word_idx])
            x = self.embedding(torch.tensor([word_idx]).to(self.device)).view(1, 1, -1)
            next_statement = self.predict_next_statement(x)
        # Generate a response based on the output sequence and the predicted next statement
        response = ' '.join(outputs) + ' ' + next_statement
        return response

    def train(rank, world_size, dataset, input_dim, hidden_dim, output_dim):
        rank = int(rank)
        world_size = int(world_size)

        torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        model = nlp_model.to(rank)
        init_process_group("nccl", rank=rank, world_size=world_size)

        torch.cuda.set_device(rank)
        ddp_model = DDP(model, device_ids=[rank])

        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = Adam(ddp_model.parameters(), lr=0.001)


        for epoch in range(128):  # Dummy epoch count
            for node_features, adjacency_matrix in dataloader:
                node_features = node_features[0].to(rank)
                adjacency_matrix = adjacency_matrix[0].to(rank)
   
    def spawn_train(world_size):
        mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    
input_dim = 4096
hidden_dim = 4096
input_hidden_dim = 1024
output_dim = 4096
nlp_model = NLPModel(input_dim, hidden_dim, output_dim)
spawn_train(4)
