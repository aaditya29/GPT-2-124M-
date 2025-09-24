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
