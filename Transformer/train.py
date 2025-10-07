import torch
import torch.nn as nn

from tokenizers import Tokenizer  # for custom tokenizer
from datasets import load_dataset  # for loading datasets
from tokenizers.models import WordLevel  # for word-level tokenization
# for training word-level tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace  # for whitespace tokenization
