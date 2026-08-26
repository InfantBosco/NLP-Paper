# Data Directory & Datasets Documentation

This directory stores raw datasets, local debug datasets, preprocessed split manifests, and SHA-256 checksums for SBERT reproduction experiments.

---

## Validated Dataset Manifest

### 1. STS Benchmark (STSb)
- **File:** `data/stsbenchmark.tsv.gz`
- **SHA-256 Checksum:** `aa3ae9b0060fdab8a7239df4bed7668086b4181e7cf6e921d086a70f9e42c17e`
- **Original Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz`
- **License:** CC BY-SA 4.0
- **Total Records:** 8,628
- **Official Splits:**
  - `train`: 5,749 sentence pairs
  - `dev`: 1,500 sentence pairs
  - `test`: 1,379 sentence pairs
- **Expected Columns:** `split`, `genre`, `dataset`, `year`, `sid`, `score`, `sentence1`, `sentence2`
- **Similarity Score Range:** $[0.0, 5.0]$
- **Preprocessing:** Normalized target score $y = score / 5.0 \in [0.0, 1.0]$.
- **Duplicate Pairs Detected:** 83

### 2. AllNLI (SNLI + MultiNLI Combined)
- **File:** `data/AllNLI.tsv.gz`
- **SHA-256 Checksum:** `60abeaa8f94820954297520a759e677cce6931f524cf535408dee0aab5ae7b54`
- **Original Source:** `https://sbert.net/datasets/AllNLI.tsv.gz`
- **License:** Open Data / CC BY-SA 4.0
- **Total Records:** 981,382
- **Official Splits:**
  - `train`: 942,069 sentence pairs
  - `dev`: 19,657 sentence pairs
  - `test`: 19,656 sentence pairs
- **Class Label Counts (3-way balance):**
  - `entailment`: 327,954
  - `contradiction`: 327,058
  - `neutral`: 326,370
- **Expected Columns:** `split`, `dataset`, `label`, `sentence1`, `sentence2`
- **Label Encoding:** `{"contradiction": 0, "entailment": 1, "neutral": 2}`
- **Duplicate Pairs Detected:** 762

### 3. Local Debug Subset
- **File:** `data/debug_dataset.json`
- **Purpose:** Fast offline testing of training and evaluation loops without loading full 1M NLI pairs.
- **Sample Count:** 3 STSb pairs, 3 AllNLI pairs.

---

## Data Leakage & Contamination Prevention
- Official dataset splits (`train`, `dev`, `test`) are strictly preserved.
- Evaluation on STSb test set is performed only after fine-tuning completes.

> **Note:** Raw TSV archives (`data/*.tsv.gz`) are ignored by `.gitignore` to keep git history clean. Run `python scripts/prepare_data.py` to regenerate local data files and manifest.
