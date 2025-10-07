import torch
import torch.nn as nn
from typing import Any  # for type hinting
from torch.utils.data import Dataset


class BilingualDataset(Dataset):

    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # here we are getting the token ids for special tokens like SOS, EOS, PAD and converting them to tensors using torch.tensor
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):  # returns the length of the dataset
        return len(self.ds)

    def __getitem__(self, idx):
        # starting from the index idx we are getting the source and target language pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]  # source text
        tgt_text = src_target_pair['translation'][self.tgt_lang]  # target text

        # converting each text to token then input ids
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # adding padding, start of sentence and end of sentence tokens
        enc_num_padding_tokens = self.seq_len - \
            len(enc_input_tokens) - 2  # -2 for SOS and EOS tokens
        dec_num_padding_tokens = self.seq_len - \
            len(dec_input_tokens) - 1  # -1 for EOS token

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            # making sure sentence is not too long
            raise ValueError("Sentence is too long")
