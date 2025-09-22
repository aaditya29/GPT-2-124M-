# GPT-2 (124M) Reproduction and Insights

## Introduction

Reproducing the GPT-2 (124M) Transformer model from scratch using PyTorch. This project is built for learning, experimentation, and extending transformer models. We are taking reference for learning from [Andrej Karpathy's Zero-To-Hero Series.](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

## Highlights

- Full Transformer architecture (multi-head attention, layer norm, residuals, position embeddings).
- BPE tokenizer implementation.
- Training pipeline with configs, logging, and checkpointing.
- Experiments on small datasets (Shakespeare, WikiText) and scaling toward OpenWebText.

## References

- [OpenAI GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

- [Karpathy’s “nanoGPT” and “minGPT”](https://github.com/karpathy/nanoGPT)

- [HuggingFace Transformers library](https://github.com/huggingface/transformers)
