# Sentence-BERT (SBERT) Reproduction Project

Independent, modular, reproducible implementation of **Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks** ([Reimers & Gurevych, EMNLP 2019](https://arxiv.org/abs/1908.10084)).

## Project Summary
This repository reproduces the core methods, baselines, evaluation benchmarks, computational efficiency gains, and ablation studies presented in the original SBERT research paper. The codebase is implemented independently in `src/sbert_reproduction` and audited against the official historical repository (`official_reference/sentence-transformers-v0.3.9`).

---

## Key Results Comparison Summary

| Metric / Experiment | Paper Reported | Official Reference (`v0.3.9`) | Independent Reproduction |
|---|---:|---:|---:|
| **Avg. GloVe Embeddings (STSb)** | 58.02 | 58.02 | *Pending Stage 6* |
| **Un-finetuned BERT `[CLS]` (STSb)** | 16.50 | 16.50 | *Pending Stage 6* |
| **Un-finetuned BERT `MEAN` (STSb)** | 46.35 | 46.35 | *Pending Stage 6* |
| **SBERT-NLI-base (Unsupervised STSb)** | 77.03 | 77.03 | *Pending Stage 7* |
| **SBERT-STSb-base (Supervised STSb)** | 84.67 $\pm$ 0.19 | 84.67 | *Pending Stage 7* |
| **SBERT-NLI-STSb-base (Two-Stage)** | 85.35 $\pm$ 0.17 | 85.35 | *Pending Stage 7* |

---

## Directory Structure
```
sbert-reproduction/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── Makefile
├── official_reference/         # Isolated official codebase (tag v0.3.9)
├── paper_notes/                # Audit, paper analysis, & reproduction spec
├── data/                       # Datasets & manifests
├── configs/                    # YAML experiment configurations
├── src/sbert_reproduction/     # Independent modular reproduction package
├── scripts/                    # Command-line entry points
├── tests/                      # Pytest unit tests suite
├── experiments/                # Outputs, metrics, checkpoints, & logs
└── report/                     # Markdown report artifacts
```

---

## Quick Start
```bash
# 1. Install dependencies
pip install -e .

# 2. Inspect environment
python scripts/inspect_environment.py

# 3. Run unit test suite
pytest -v tests/
```

---

## Reproduction & Verification Commands
```bash
# Run Data Preparation
python scripts/prepare_data.py --config configs/sbert_stsb.yaml

# Run Baseline Experiments
python scripts/run_baselines.py --config configs/baseline_tfidf.yaml

# Train SBERT on AllNLI
python scripts/train_sbert.py --config configs/sbert_nli.yaml

# Evaluate SBERT on STS Benchmark
python scripts/evaluate_stsb.py --config configs/sbert_stsb.yaml

# Run Computational Efficiency Benchmark
python scripts/benchmark_similarity.py --config configs/benchmark.yaml
```

---

## License
MIT License. See [LICENSE](LICENSE) for details.
