import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDoubleEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, output_vocab_size, max_seq_length):
        super(TransformerDoubleEncoder, self).__init__()
        
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)
        ])
        
        self.linear = nn.Linear(d_model, output_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, src, tgt):
        src = self.input_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.output_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]
        
        for layer in self.encoder_layers:
            src = layer(src)
        
        for layer in self.decoder_layers:
            tgt = layer(tgt, src)
        
        output = self.linear(tgt)
        return self.softmax(output)

# Parámetros del modelo
num_layers = 48  # Número de capas
d_model = 2048  # Dimensionalidad de los embeddings
num_heads = 32  # Número de cabezas de atención
d_ff = 8192  # Dimensionalidad de la capa feed-forward
input_vocab_size = 50000  # Tamaño del vocabulario de entrada
output_vocab_size = 50000  # Tamaño del vocabulario de salida
max_seq_length = 512  # Longitud máxima de la secuencia

# Crear el modelo
trans_model = TransformerDoubleEncoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, output_vocab_size, max_seq_length)

# Contar el número de parámetros
num_params = sum(p.numel() for p in trans_model.parameters() if p.requires_grad)
print(f"Number of parameters: {num_params}")
