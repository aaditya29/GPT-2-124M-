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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class
        self.d_model = d_model  # Dimension of the model
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        """Create a positional encoding matrix of shape (seq_len, d_model)
        We are using positional encoding to give the model some information about the relative position of the words in the sentence.
        The positional encoding is taken from the paper "Attention is All You Need" where we use sine and cosine functions of different frequencies to encode the positions. The sine function is used for even indices and the cosine function is used for odd indices so that each dimension of the positional encoding corresponds to a sinusoid of different wavelength.
        """
        pe = torch.zeros(
            seq_len, d_model)  # Initialize the positional encoding matrix
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1)  # Create a column vector of positions (seq_len, 1)
        # Create the div_term for the sine and cosine functions
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add a batch dimension (1, seq_len, d_model)
        # Register pe as a buffer so it is not considered a model parameter
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        # Apply dropout to the sum of embeddings and positional encodings
        return self.dropout(x)


class LayerNormalisation(nn.Module):

    # where eps is epsilon a small value to avoid division by zero
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()  # Initialize the parent class
        self.eps = eps  # Epsilon value to avoid division by zero
        self.alpha = nn.Parameter(torch.ones(1))  # Scale parameter
        self.bias = nn.Parameter(torch.zeros(1))  # Shift parameter

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Mean of the last dimension
        # Standard deviation of the last dimension
        std = x.std(dim=-1, keepdim=True)
        # Normalize and scaling
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class
        self.linear1 = nn.Linear(d_model, d_ff)  # First linear layer
        self.linear2 = nn.Linear(d_ff, d_model)  # Second linear layer
        self.dropout = nn.Dropout(dropout)  # Dropout layer
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    """
    d_model: Dimension of the model
    h: Number of heads
    dropout: Dropout rate
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()  # Initialize the parent class
        self.d_model = d_model  # Dimension of the model
        self.h = h  # Number of heads
        # d_model must be divisible by h because we will split the d_model into h heads
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h  # Dimension of each head
        # Linear layer for query
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        # Linear layer for key
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        # Linear layer for value
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        # Linear layer for output
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)  # Dropout layer
