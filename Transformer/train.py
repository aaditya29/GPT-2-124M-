import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


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
