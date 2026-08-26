# Project Limitations & Scope Boundary

This document outlines known hardware, environmental, data, and methodology limitations for the reproduction project.

## 1. Hardware Limitations
- **GPU VRAM:** Target hardware is NVIDIA GeForce RTX 2050 with 4 GB VRAM. Original paper trained `bert-base` and `bert-large` on NVIDIA Tesla V100 (16-32 GB VRAM).
- **Mitigation:** Large batch size (16/32) simulation uses FP16 mixed precision (`torch.cuda.amp.autocast`) and gradient accumulation steps. Sequence length is capped at 128 tokens for baseline experiments.

## 2. Dataset Availability & License Constraints
- **Public STS & NLI:** STSb and AllNLI datasets are publicly accessible.
- **Wikipedia Section Triplets:** The 1.8M triplet dataset (Dor et al., 2018) requires fetching from original external mirrors.

## 3. Pretrained Weights Scope
- Base transformer backbones (`bert-base-uncased`, `roberta-base`) are loaded from HuggingFace Hub. No pre-trained SBERT models (e.g. `sentence-transformers/bert-base-nli-mean-tokens`) are used for training — all SBERT weights are fine-tuned from standard BERT checkpoints.
