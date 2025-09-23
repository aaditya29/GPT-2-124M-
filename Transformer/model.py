import math
import torch
import torch.nn as nn


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps: float = 10**-6 -> None):
        super().__init__()  # Initialize the LayerNormalization module
