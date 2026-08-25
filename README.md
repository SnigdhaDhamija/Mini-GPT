# Mini-GPT — Decoder-Only Transformer from Scratch

## Overview

Mini-GPT is a **GPT-style Decoder-Only Transformer implemented from scratch using TensorFlow and Keras**.

The project is designed to build a practical understanding of the architecture and training pipeline behind modern Large Language Models (LLMs). Rather than using a pre-trained GPT implementation, the core components of the Transformer architecture are implemented and assembled step by step.

The project covers the complete workflow from raw text preprocessing and subword tokenization to Transformer-based language modeling and autoregressive text generation.

### Core Pipeline

**Text Corpus → Tokenization → Sequence Preparation → Embeddings → Positional Information → Causal Multi-Head Self-Attention → Feed-Forward Network → Transformer Blocks → Language Modeling Head → Next-Token Prediction**

---

## Objectives

The primary objectives of this project are to:

- Understand the architecture of decoder-only Transformers.
- Implement the major Transformer components using TensorFlow/Keras.
- Understand the Query, Key, and Value attention mechanism.
- Implement causal self-attention for autoregressive language modeling.
- Understand the role of token embeddings and positional information.
- Build Transformer decoder blocks using residual connections and Layer Normalization.
- Prepare sequential training data for next-token prediction.
- Train a small language model from scratch.
- Understand the complete workflow behind GPT-style text generation.

---

## Architecture

The model follows the fundamental architecture of a decoder-only Transformer.

```text
                         Input Text
                             │
                             ▼
                    Text Preprocessing
                             │
                             ▼
                  SentencePiece Tokenizer
                             │
                             ▼
                         Token IDs
                             │
                             ▼
                  Input / Target Sequences
                             │
                             ▼
                  Token + Position Embeddings
                             │
                             ▼
               ┌─────────────────────────────┐
               │     Transformer Block       │
               │                             │
               │  Causal Multi-Head          │
               │  Self-Attention             │
               │             │                │
               │     Residual Connection     │
               │             │                │
               │       Layer Normalization   │
               │             │                │
               │     Feed-Forward Network    │
               │             │                │
               │     Residual Connection     │
               │             │                │
               │       Layer Normalization   │
               └──────────────┬──────────────┘
                              │
                              ▼
                    Multiple Transformer
                           Blocks
                              │
                              ▼
                     Output Representation
                              │
                              ▼
                       Language Model Head
                              │
                              ▼
                       Vocabulary Logits
                              │
                              ▼
                     Next-Token Prediction
                              │
                              ▼
                       Text Generation
