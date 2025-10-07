import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# for creating causal mask and dataset class
from dataset import BilingualDataset, causal_mask
from model import build_transformer


from tokenizers import Tokenizer  # for custom tokenizer
from datasets import load_dataset  # for loading datasets
from tokenizers.models import WordLevel  # for word-level tokenization
# for training word-level tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace  # for whitespace tokenization

from pathlib import Path


def get_all_sentences(ds, lang):
    for item in ds:  # iterate through all items in dataset
        yield item[lang]


def get_or_build_tokenizer(config, ds, lang):
    # Path to save/load tokenizer
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):  # if path does not exist then build tokenizer
        print(f"Building {lang} tokenizer...")
        # initialize word-level tokenizer and set unk token for unknown words
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()  # set pre-tokenizer to whitespace
        # splitting words by whitespace and special tokens called UNK, PAD, SOS, EOS which are unknown, padding, start of sentence and end of sentence tokens respectively
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(
            get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))  # save tokenizer to path
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset(
        f"{config['datasource']}", f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # building tokenizers for source and target languages
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # 90% train, 10% val split
    # where ds_raw is the entire dataset
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(
        ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt,
                                config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt,
                              config['lang_src'], config['lang_tgt'], config['seq_len'])

    # finding the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(
            item['translation'][config['lang_src']]).ids  # getting the token ids for each source and target sentence
        tgt_ids = tokenizer_tgt.encode(
            item['translation'][config['lang_tgt']]).ids  # getting the token ids for each source and target sentence
        # getting the maximum length of source sentence
        max_len_src = max(max_len_src, len(src_ids))
        # getting the maximum length of target sentence
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # creating dataloaders for train and validation datasets
    train_dataloader = DataLoader(
        train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    # here we are building the transformer model using the build_transformer function from model.py
    # we are passing the vocab size of source and target languages, sequence length and d_model as parameters to the function
    # and returning the model
    model = build_transformer(vocab_src_len, vocab_tgt_len,
                              config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model
