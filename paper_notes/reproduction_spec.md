# Stage 2: Reproduction Specification & Experiment Matrix

This document specifies the exact experiment matrix for reproducing Sentence-BERT (SBERT). Every experiment specifies inputs, architecture, optimization hyperparameters, evaluation protocols, expected targets, and confidence labels.

---

## Confidence Label Legend
- `PAPER_SPECIFIED`: Stated directly in the paper.
- `OFFICIAL_CODE_SPECIFIED`: Found in the official reference repository (`v0.3.9`).
- `PROJECT_DEFAULT`: Chosen for project reproducibility or hardware adaptation.
- `INFERRED`: Derived logically from standard practices.
- `UNKNOWN`: Not documented in paper or repository.

---

## Experiment Matrix

### Group A: TF-IDF Baseline
- **Experiment ID:** `EXP-A01`
- **Goal:** Evaluate classical TF-IDF bag-of-words representation on STS benchmark.
- **Dataset:** STS Benchmark (`STSb`)
- **Dataset Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz` (`OFFICIAL_CODE_SPECIFIED`)
- **Official Split:** `test` split (`PAPER_SPECIFIED`)
- **Input Columns:** `sentence1`, `sentence2` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** Lowercase, regex word tokenization, TF-IDF vectorizer fitted on `train` split (`PROJECT_DEFAULT`)
- **Model:** TF-IDF Cosine Similarity (`PROJECT_DEFAULT`)
- **Encoder:** N/A
- **Pooling:** N/A
- **Loss:** N/A (Unsupervised baseline)
- **Batch Size:** N/A (Scikit-learn batch/sparse matrix)
- **Learning Rate:** N/A
- **Optimizer:** N/A
- **Scheduler:** N/A
- **Warmup Steps:** N/A
- **Maximum Sequence Length:** N/A
- **Number of Epochs:** N/A
- **Number of Steps:** N/A
- **Random Seed:** `42` (`PROJECT_DEFAULT`)
- **Evaluation Metrics:** Spearman rank correlation $\rho \times 100$ (`PAPER_SPECIFIED`)
- **Expected Outputs:** Spearman correlation $\rho \approx 40.0 - 46.0$ on STSb (`PAPER_SPECIFIED` Table 3 / Table 1)
- **Acceptance Criteria:** Code produces deterministic Spearman $\rho$ output saved to JSON/CSV (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 4, Table 1, Table 3 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/training/avg_word_embeddings/training_stsbenchmark_tf-idf_word_embeddings.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `OFFICIAL_CODE_SPECIFIED`

---

### Group B: Averaged Word-Embedding Baseline
- **Experiment ID:** `EXP-B01`
- **Goal:** Evaluate static word embeddings (GloVe/FastText) averaged over sentence length.
- **Dataset:** STS Benchmark (`STSb`)
- **Dataset Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz` (`OFFICIAL_CODE_SPECIFIED`)
- **Official Split:** `test` split (`PAPER_SPECIFIED`)
- **Input Columns:** `sentence1`, `sentence2` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** Tokenization via SpaCy/NLTK, lookup in GloVe 300d embeddings (`PROJECT_DEFAULT`)
- **Model:** Mean GloVe Embedding Vectors (`PAPER_SPECIFIED`)
- **Encoder:** GloVe 300d (`PAPER_SPECIFIED`)
- **Pooling:** `MEAN` (`PAPER_SPECIFIED`)
- **Loss:** N/A (Unsupervised baseline)
- **Batch Size:** N/A
- **Learning Rate:** N/A
- **Optimizer:** N/A
- **Scheduler:** N/A
- **Warmup Steps:** N/A
- **Maximum Sequence Length:** Full sentence (`PAPER_SPECIFIED`)
- **Number of Epochs:** N/A
- **Number of Steps:** N/A
- **Random Seed:** `42` (`PROJECT_DEFAULT`)
- **Evaluation Metrics:** Spearman rank correlation $\rho \times 100$ (`PAPER_SPECIFIED`)
- **Expected Outputs:** Spearman correlation $\rho = 58.02$ on STSb (`PAPER_SPECIFIED` Table 1)
- **Acceptance Criteria:** Correlation within $\pm 1.0$ point of paper baseline (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 4, Table 1 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/training/avg_word_embeddings/training_stsbenchmark_avg_word_embeddings.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group C: Vanilla BERT CLS Baseline
- **Experiment ID:** `EXP-C01`
- **Goal:** Evaluate out-of-the-box pretrained `bert-base-uncased` using `[CLS]` token without fine-tuning.
- **Dataset:** STS Benchmark (`STSb`)
- **Dataset Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz` (`OFFICIAL_CODE_SPECIFIED`)
- **Official Split:** `test` split (`PAPER_SPECIFIED`)
- **Input Columns:** `sentence1`, `sentence2` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** HuggingFace `BertTokenizer` uncased (`PAPER_SPECIFIED`)
- **Model:** Pretrained `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Encoder:** `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Pooling:** `CLS` (`PAPER_SPECIFIED`)
- **Loss:** N/A (Zero-shot inference)
- **Batch Size:** 32 (`PROJECT_DEFAULT`)
- **Learning Rate:** N/A
- **Optimizer:** N/A
- **Scheduler:** N/A
- **Warmup Steps:** N/A
- **Maximum Sequence Length:** 128 tokens (`PROJECT_DEFAULT`)
- **Number of Epochs:** N/A
- **Number of Steps:** N/A
- **Random Seed:** `42` (`PROJECT_DEFAULT`)
- **Evaluation Metrics:** Spearman rank correlation $\rho \times 100$ (`PAPER_SPECIFIED`)
- **Expected Outputs:** Spearman correlation $\rho = 16.50$ on STSb (`PAPER_SPECIFIED` Table 1)
- **Acceptance Criteria:** Correlation matches paper within $\pm 1.5$ points (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 4, Table 1 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/evaluation/evaluation_stsbenchmark.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group D: Vanilla BERT Mean-Pooling Baseline
- **Experiment ID:** `EXP-D01`
- **Goal:** Evaluate out-of-the-box pretrained `bert-base-uncased` with `MEAN` pooling without fine-tuning.
- **Dataset:** STS Benchmark (`STSb`)
- **Dataset Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz` (`OFFICIAL_CODE_SPECIFIED`)
- **Official Split:** `test` split (`PAPER_SPECIFIED`)
- **Input Columns:** `sentence1`, `sentence2` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** HuggingFace `BertTokenizer` uncased (`PAPER_SPECIFIED`)
- **Model:** Pretrained `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Encoder:** `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Pooling:** `MEAN` (attention-mask aware) (`PAPER_SPECIFIED`)
- **Loss:** N/A (Zero-shot inference)
- **Batch Size:** 32 (`PROJECT_DEFAULT`)
- **Learning Rate:** N/A
- **Optimizer:** N/A
- **Scheduler:** N/A
- **Warmup Steps:** N/A
- **Maximum Sequence Length:** 128 tokens (`PROJECT_DEFAULT`)
- **Number of Epochs:** N/A
- **Number of Steps:** N/A
- **Random Seed:** `42` (`PROJECT_DEFAULT`)
- **Evaluation Metrics:** Spearman rank correlation $\rho \times 100$ (`PAPER_SPECIFIED`)
- **Expected Outputs:** Spearman correlation $\rho = 46.35$ on STSb (`PAPER_SPECIFIED` Table 1)
- **Acceptance Criteria:** Correlation matches paper within $\pm 1.5$ points (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 4, Table 1 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/evaluation/evaluation_stsbenchmark.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group E: SBERT Classification Objective (NLI Fine-tuning)
- **Experiment ID:** `EXP-E01` (`SBERT-NLI-base`)
- **Goal:** Fine-tune SBERT on AllNLI (SNLI + MultiNLI) using 3-way softmax classifier objective.
- **Dataset:** AllNLI (`SNLI` + `MultiNLI`)
- **Dataset Source:** `https://sbert.net/datasets/AllNLI.tsv.gz` (`OFFICIAL_CODE_SPECIFIED`)
- **Official Split:** `train` split (~1,000,000 sentence pairs) (`PAPER_SPECIFIED`)
- **Input Columns:** `sentence1`, `sentence2`, `label` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** Tokenization, max len 128, label mapping `{"contradiction": 0, "entailment": 1, "neutral": 2}` (`OFFICIAL_CODE_SPECIFIED`)
- **Model:** SBERT Siamese Network (`PAPER_SPECIFIED`)
- **Encoder:** `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Pooling:** `MEAN` (`PAPER_SPECIFIED`)
- **Loss:** `SoftmaxLoss` with concatenation $(u, v, |u - v|)$ (`PAPER_SPECIFIED` Eq. 1)
- **Batch Size:** 16 (`PAPER_SPECIFIED`)
- **Learning Rate:** $2e-5$ (`PAPER_SPECIFIED`)
- **Optimizer:** Adam / AdamW (`PAPER_SPECIFIED` / `PROJECT_DEFAULT`)
- **Scheduler:** Linear warmup (`PAPER_SPECIFIED`)
- **Warmup Steps:** 10% of total training steps (~6,250 steps) (`PAPER_SPECIFIED`)
- **Maximum Sequence Length:** 128 tokens (`PROJECT_DEFAULT`)
- **Number of Epochs:** 1 (`PAPER_SPECIFIED`)
- **Number of Steps:** ~62,500 steps (`INFERRED`)
- **Random Seed:** `42` (`PROJECT_DEFAULT`)
- **Evaluation Metrics:** Evaluates dev STSb during training; evaluates STS12-16, STSb, SICK-R test sets after epoch (`PAPER_SPECIFIED`)
- **Expected Outputs:** NLI Dev Accuracy $\approx 80.78\%$, STSb Test Spearman $\rho \approx 77.03$ (`PAPER_SPECIFIED` Table 1 & Table 6)
- **Acceptance Criteria:** STSb Unsupervised Test Spearman correlation $\ge 75.5$ (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 3.1, Section 4, Table 1 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/training/nli/training_nli.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group F: SBERT Regression Objective (STSb Fine-tuning)
- **Experiment ID:** `EXP-F01` (`SBERT-STSb-base`)
- **Goal:** Fine-tune SBERT directly on STSb using Cosine Similarity Regression Objective.
- **Dataset:** STS Benchmark (`STSb`)
- **Dataset Source:** `https://sbert.net/datasets/stsbenchmark.tsv.gz` (`OFFICIAL_CODE_SPECIFIED`)
- **Official Split:** `train` (5,749 pairs), `dev` (1,500 pairs), `test` (1,379 pairs) (`PAPER_SPECIFIED`)
- **Input Columns:** `sentence1`, `sentence2`, `score` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** Target normalized $score / 5.0$ to range $[0, 1]$ (`OFFICIAL_CODE_SPECIFIED`)
- **Model:** SBERT Siamese Network (`PAPER_SPECIFIED`)
- **Encoder:** `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Pooling:** `MEAN` (`PAPER_SPECIFIED`)
- **Loss:** `CosineSimilarityLoss` (MSE on Cosine Similarity) (`PAPER_SPECIFIED`)
- **Batch Size:** 16 (`PAPER_SPECIFIED`)
- **Learning Rate:** $2e-5$ (`PAPER_SPECIFIED`)
- **Optimizer:** Adam / AdamW (`PAPER_SPECIFIED`)
- **Scheduler:** Linear warmup (`PAPER_SPECIFIED`)
- **Warmup Steps:** 10% of total steps (`PAPER_SPECIFIED`)
- **Maximum Sequence Length:** 128 tokens (`PROJECT_DEFAULT`)
- **Number of Epochs:** 4 (`PAPER_SPECIFIED` Table 2 note)
- **Number of Steps:** ~1,440 steps (`INFERRED`)
- **Random Seed:** 10 random seeds (`42` to `51`) (`PAPER_SPECIFIED`)
- **Evaluation Metrics:** STSb Test Spearman rank correlation $\rho \times 100$ (`PAPER_SPECIFIED`)
- **Expected Outputs:** STSb Test Spearman $\rho = 84.67 \pm 0.19$ (`PAPER_SPECIFIED` Table 2)
- **Acceptance Criteria:** Mean test Spearman $\rho \ge 84.0$ across 10 seeds (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 4.2, Table 2 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/training/other/training_stsbenchmark.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

- **Experiment ID:** `EXP-F02` (`SBERT-NLI-STSb-base`)
- **Goal:** Fine-tune `SBERT-NLI-base` (from `EXP-E01`) on STSb (Two-stage fine-tuning).
- **Setup:** Same as `EXP-F01`, starting from `EXP-E01` checkpoint.
- **Expected Outputs:** STSb Test Spearman $\rho = 85.35 \pm 0.17$ (`PAPER_SPECIFIED` Table 2)
- **Acceptance Criteria:** Mean test Spearman $\rho \ge 84.8$ (`PROJECT_DEFAULT`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group G: SBERT Triplet Objective (Wikipedia Section Triplets)
- **Experiment ID:** `EXP-G01` (`SBERT-WikiSec-base`)
- **Goal:** Train SBERT on Wikipedia section triplets using Triplet Loss.
- **Dataset:** Wikipedia Section Triplets (Dor et al., 2018)
- **Official Split:** 1.8M train triplets, 222,957 test triplets (`PAPER_SPECIFIED`)
- **Input Columns:** `anchor`, `positive`, `negative` (`OFFICIAL_CODE_SPECIFIED`)
- **Preprocessing:** Tokenization, max len 128 (`PROJECT_DEFAULT`)
- **Model:** SBERT Triplet Network (`PAPER_SPECIFIED`)
- **Encoder:** `bert-base-uncased` (`PAPER_SPECIFIED`)
- **Pooling:** `MEAN` (`PAPER_SPECIFIED`)
- **Loss:** `TripletLoss` with Euclidean distance metric and margin $\epsilon = 1.0$ (`PAPER_SPECIFIED`)
- **Batch Size:** 16 (`PAPER_SPECIFIED`)
- **Learning Rate:** $2e-5$ (`PAPER_SPECIFIED`)
- **Optimizer:** Adam / AdamW (`PAPER_SPECIFIED`)
- **Scheduler:** Linear warmup 10% (`PAPER_SPECIFIED`)
- **Maximum Sequence Length:** 128 tokens (`PROJECT_DEFAULT`)
- **Number of Epochs:** 1 (`PAPER_SPECIFIED` Table 4 note)
- **Random Seed:** `42` (`PROJECT_DEFAULT`)
- **Evaluation Metrics:** Accuracy (Is $d(a, p) < d(a, n)$?) (`PAPER_SPECIFIED` Section 4.4)
- **Expected Outputs:** Test Accuracy $= 0.8042$ (80.42%) (`PAPER_SPECIFIED` Table 4)
- **Acceptance Criteria:** Triplet test accuracy $\ge 79.5\%$ (`PROJECT_DEFAULT`)
- **Paper Reference:** Section 4.4, Table 4 (`PAPER_SPECIFIED`)
- **Official Repository Reference:** `examples/training/other/training_wikipedia_sections.py` (`OFFICIAL_CODE_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group H: STS Evaluation Benchmark Suite
- **Experiment ID:** `EXP-H01`
- **Goal:** Evaluate trained SBERT models across 7 STS datasets (STS12-16, STSb, SICK-R).
- **Datasets:** STS12, STS13, STS14, STS15, STS16, STSb, SICK-R (`PAPER_SPECIFIED`)
- **Evaluation Metric:** Spearman rank correlation $\rho \times 100$ on Cosine Similarity (`PAPER_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group I: NLI & Transfer Evaluation (SentEval)
- **Experiment ID:** `EXP-I01`
- **Goal:** Evaluate SBERT sentence embeddings on 7 downstream classification tasks (MR, CR, SUBJ, MPQA, SST, TREC, MRPC).
- **Protocol:** Freeze sentence embeddings, fit logistic regression classifier via 10-fold cross-validation (`PAPER_SPECIFIED` Section 5)
- **Expected Output:** SBERT-NLI-base Average Accuracy $= 87.41\%$ (`PAPER_SPECIFIED` Table 5)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group J: Efficiency Benchmark Comparison
- **Experiment ID:** `EXP-J01`
- **Goal:** Benchmark inference speed and throughput (sentences/sec) of Cross-Encoder vs Bi-Encoder SBERT (with & without Smart Batching).
- **Corpus Sizes:** 100, 1,000, 10,000 sentences from STSb (`PAPER_SPECIFIED` Section 7)
- **Measured Metrics:** Forward passes count, embedding generation latency, cosine comparison time, QPS, Peak RAM (`PAPER_SPECIFIED`)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group K: Pooling Strategy Ablations
- **Experiment ID:** `EXP-K01` (`MEAN` vs `MAX` vs `CLS`)
- **Goal:** Compare impact of pooling method when fine-tuning on NLI and STSb.
- **Variants:**
  1. `MEAN` (NLI Acc: 80.78, STSb Dev: 87.44)
  2. `MAX` (NLI Acc: 79.07, STSb Dev: 69.92)
  3. `CLS` (NLI Acc: 79.80, STSb Dev: 86.62)
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group L: Encoder Architecture Ablations
- **Experiment ID:** `EXP-L01` (`bert-base` vs `bert-large` vs `roberta-base` vs `roberta-large`)
- **Goal:** Compare embedding performance across transformer backbone architectures.
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group M: Concatenation Vector Ablations
- **Experiment ID:** `EXP-M01`
- **Goal:** Test classification objective concatenation variants: $(u, v)$, $(|u - v|)$, $(u * v)$, $(|u - v|, u * v)$, $(u, v, |u - v|)$.
- **Confidence Status:** `PAPER_SPECIFIED`

---

### Group N: Distance Metric & Triplet Margin Ablations
- **Experiment ID:** `EXP-N01`
- **Goal:** Compare Euclidean distance with margin $\epsilon = 1.0$ vs $\epsilon = 5.0$ vs Cosine distance.
- **Confidence Status:** `PROJECT_DEFAULT`
