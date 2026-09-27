# SDM Narrative Classification

Fine-tuned large language models for classifying **Specificity** and **Integration** in self-defining memory (SDM) narratives, developed as part of an ongoing research collaboration between Rishi Gnanasekar and Prof. Caleb Siefert (University of Michigan–Dearborn).

## Overview

This repository contains LoRA adapter weights and fine-tuning code for five pretrained language model architectures, each independently fine-tuned on two binary classification tasks:

- **Specificity** — whether a self-defining memory narrative contains concrete, particular detail
- **Integration** — whether a narrative demonstrates explicit meaning-making or self-insight

Each architecture was fine-tuned separately for each task, using an identical training pipeline and hyperparameter configuration to enable direct comparison across models.

## Base Models

| Architecture | Base Model (Hugging Face) |
|---|---|
| Llama 3.2 (3B) | `unsloth/Llama-3.2-3B-Instruct` |
| Gemma 3 (4B) | `unsloth/gemma-3-4b-it` |
| Qwen 2.5 (3B) | `Qwen/Qwen2.5-3B-Instruct` |
| Mistral (7B) | `unsloth/mistral-7b-bnb-4bit` |
| OpenChat 3.5 | `openchat/openchat-3.5-0106` |

This repository contains only the fine-tuned LoRA adapter weights, not the base model weights. Base models must be downloaded separately from their respective Hugging Face repositories listed above.
Each model folder contains the fine-tuned LoRA adapter weights and tokenizer configuration for that architecture and task.

## Methodology

- **Fine-tuning approach:** Low-Rank Adaptation (LoRA), rank 8, alpha 16, applied to query and value attention projections
- **Quantization:** 4-bit (NF4, double quantization)
- **Class imbalance handling:** Focal Loss (gamma = 2.0), with class weights computed via inverse-frequency weighting
- **Validation:** Five-fold stratified cross-validation, with early stopping on validation F1
- **Optimizer:** AdamW (8-bit), linear learning rate schedule, learning rate 2e-4

Full methodology and results are documented in the accompanying manuscript (in preparation).

## Status

This work is part of an ongoing research collaboration and is currently under manuscript review. Models are provided for research purposes only.
