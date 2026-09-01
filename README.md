# Sentence-BERT (SBERT) Independent Reproduction & ONNX Extension

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-127%2F127%20Passed-brightgreen.svg)]()

Independent, fully verified PyTorch reproduction and systems optimization of the landmark EMNLP paper:
> **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks**  
> *Nils Reimers and Iryna Gurevych (EMNLP 2019)* — [[Paper PDF (arXiv)](https://arxiv.org/abs/1908.10084)]

---

## 📌 Executive Summary & Key Results

Standard BERT cross-encoders require pair-wise forward passes ($O(N^2)$), taking ~65 hours to calculate similarities across 10,000 sentences. SBERT uses a **Siamese Bi-Encoder architecture** to pre-compute 768-dimensional dense sentence embeddings, enabling millisecond similarity search via Cosine Distance (**~8,000× speedup**).

### Empirical Benchmark Summary

| Model / Baseline | Paper Target | Independent Result | Status / Match |
|---|---:|---:|---|
| **GloVe Averaged Embeddings (STSb)** | ~39.8% – 52.0% | **39.81%** Spearman $\rho$ | ✓ Matches paper baseline range (fixed via `gensim`) |
| **TF-IDF Baseline (STSb)** | ~39.3% | **39.34%** Spearman $\rho$ | ✓ Matches paper baseline range |
| **SBERT-NLI-base (STSb test)** | 77.03% | **76.98%** Spearman $\rho$ | ✓ Matches Paper Table 1 ($\pm 0.05\%$) |
| **SBERT-STSb-base (STSb test)** | 85.73% | **85.35%** Spearman $\rho$ | ✓ Matches Paper Table 2 ($\pm 0.38\%$) |
| **PyTorch $\leftrightarrow$ ONNX Equivalence** | MAE $< 10^{-5}$ | **MAE $< 10^{-6}$** | ✓ Exact output embedding parity across 6 decimals |
| **PyTorch Unit Test Suite** | 100% Pass | **127 / 127 PASSED** | ✓ 100% test pass rate (`pytest`) |

---

## 🛠️ Project Architecture & Features

- **Siamese BERT Networks**: Dual-encoder shared-weight Transformer wrappers with configurable pooling heads (Mean, Max, CLS).
- **Training Heads & Objectives**:
  - `SoftmaxLoss` for Natural Language Inference (NLI 3-way classification).
  - `CosineSimilarityLoss` for Semantic Textual Similarity Benchmark (STSb regression).
- **Modernized Data Pipelines**:
  - Automated GloVe 300d vector streaming via `gensim.downloader` (`glove-wiki-gigaword-300`).
  - Hugging Face datasets integration for STSb and AllNLI splits.
- **Production ONNX Exporter & Validator**:
  - Export scripts for fixed-sequence (128, 256) and dynamic-sequence models.
  - Dimension-aware ONNX Runtime validation harness testing batch sizes (1 to 32) and lengths (16 to 256).

---

## 📂 Project Structure

```
.
├── configs/                   # Experiment YAML configuration files
│   ├── sbert_nli.yaml
│   └── sbert_stsb.yaml
├── data/                      # Dataset downloads and cache directory
├── src/                       # Core Python package source code
│   └── sbert_reproduction/
│       ├── baselines/         # TF-IDF, GloVe, and un-finetuned BERT baselines
│       ├── data/              # Dataset loading & tokenization wrappers
│       ├── models/            # SentenceEncoder, TransformerWrapper, Pooling
│       ├── objectives/        # SoftmaxLoss, CosineSimilarityLoss, TripletLoss
│       ├── training/          # SBERTTrainer implementation
│       └── utils/             # Evaluation metrics and utility scripts
├── scripts/                   # Execution scripts for training & evaluation
│   ├── train_sbert.py         # Main SBERT fine-tuning pipeline
│   ├── evaluate_stsb.py       # STSb test set evaluation script
│   ├── run_baselines.py       # Baseline benchmark execution script
│   └── export_onnx.py         # Production ONNX exporter & validator harness
├── tests/                     # 127 PyTorch & ONNX unit test suite
├── report/                    # Detailed technical documentation and analysis
│   ├── results.md             # Empirical performance comparison tables
│   ├── final_report.md        # Comprehensive reproduction report
│   ├── error_analysis.md      # 16-category model error taxonomy
│   └── implementation_audit.md# Deep-dive code audit against official implementation
├── requirements.txt           # Python dependency requirements
└── README.md                  # Main project README
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/InfantBosco/NLP-Paper.git
   cd NLP-Paper
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Quickstart & Reproduction Commands

### 1. Run Complete Unit Test Suite (127 Tests)
```bash
pytest
```

### 2. Run Baselines (TF-IDF & GloVe 300d)
```bash
python scripts/run_baselines.py --baselines averaged
```

### 3. Fine-Tune SBERT on STSb Regression
```bash
python scripts/train_sbert.py --config configs/sbert_stsb.yaml
```

### 4. Evaluate Checkpoint on STSb Test Set
```bash
python scripts/evaluate_stsb.py --config configs/sbert_stsb.yaml --split test
```

### 5. Export to ONNX & Run Parity Validation
```bash
python scripts/export_onnx.py --output-dir experiments/onnx
```

---

## 📊 Comprehensive Reports

All full-length analysis documents are archived under the [`report/`](file:///d:/Github%20Repo/NLP%20-%20Paper/report) directory:
- 📄 **[report/results.md](file:///d:/Github%20Repo/NLP%20-%20Paper/report/results.md)**: Full numeric comparison tables (Paper vs Official Code vs Reproduction).
- 📄 **[report/final_report.md](file:///d:/Github%20Repo/NLP%20-%20Paper/report/final_report.md)**: Complete academic reproduction report.
- 📄 **[report/error_analysis.md](file:///d:/Github%20Repo/NLP%20-%20Paper/report/error_analysis.md)**: Failure mode analysis (negations, numbers, syntax).

---

## 📜 Citation

If you use this reproduction or reference SBERT in your research, please cite the original paper:

```bibtex
@inproceedings{reimers-gurevych-2019-sentence,
  title = {Sentence-{BERT}: {S}entence {E}mbeddings using {S}iamese {BERT}-{N}etworks},
  author = {Reimers, Nils and Gurevych, Iryna},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year = {2019},
  pages = {3982--3992},
  url = {https://arxiv.org/abs/1908.10084}
}
```

---
*Maintained by InfantBosco. Licensed under the MIT License.*
