import math
import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:  # (d_model, vocab_size)
        super().__init__()  # Initialize the parent class
        self.d_model = d_model  # Dimension of the model
        self.vocab_size = vocab_size  # Size of the vocabulary
        self.embedding = nn.Embedding(vocab_size, d_model)  # Embedding layer

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
