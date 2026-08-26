# Data Directory & Datasets Documentation

This directory stores datasets, preprocessed split files, and metadata manifests for SBERT reproduction experiments.

## Supported Datasets

### 1. STS Benchmark (STSb)
- **Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz`
- **License:** CC BY-SA 4.0
- **Official Splits:** Train (5,749), Dev (1,500), Test (1,379)
- **Columns:** `split`, `genre`, `dataset`, `year`, `sid`, `score`, `sentence1`, `sentence2`
- **Label Normalization:** Score divided by 5.0 to map range $[0.0, 5.0] \rightarrow [0.0, 1.0]$.

### 2. AllNLI (SNLI + MultiNLI)
- **Source:** `https://sbert.net/datasets/AllNLI.tsv.gz`
- **License:** Open Data / CC BY-SA 4.0
- **Official Splits:** Train (~1,000,000 pairs), Dev, Test
- **Columns:** `split`, `dataset`, `label`, `sentence1`, `sentence2`
- **Label Encoding:** `{"contradiction": 0, "entailment": 1, "neutral": 2}`

> **Note:** Raw dataset files (`*.tsv`, `*.gz`) are ignored by `.gitignore` to prevent committing large binary assets to Git. Run `python scripts/prepare_data.py` to download and generate local dataset manifests.
