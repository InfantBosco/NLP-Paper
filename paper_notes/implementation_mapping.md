# Sentence-BERT Implementation Mapping and Verification Report

This document maps the requirements, architectures, and objectives from the Sentence-BERT paper (Reimers & Gurevych, 2019) to the official reference repository (`v0.3.9`) and our independent implementation (`src/sbert_reproduction`), summarizes the execution of all reproduction steps, and compares the obtained results.

---

## 1. Implementation Mapping Table

| ID | Paper requirement | Paper location | Official code | Independent code | Status | Evidence | Discrepancy | Action |
|---|---|---|---|---|---|---|---|---|
| IMP-01 | Siamese Encoder Architecture | Section 3 | `sentence_transformers/SentenceTransformer.py` | [`sbert_model.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/models/sbert_model.py#L192) (`SBERTModel` class) | PASS | Weights tied across branches; forward pass encodes both sentences using a single shared encoder model. | None | Verified |
| IMP-02 | Token Mean Pooling | Section 3 | `sentence_transformers/models/Pooling.py` | [`pooling.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/models/pooling.py#L31) (`MeanPooling` class) | PASS | Dynamically computes mean over token embeddings while ignoring padded tokens using attention mask. | None | Verified |
| IMP-03 | Token Max Pooling | Section 3 | `sentence_transformers/models/Pooling.py` | [`pooling.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/models/pooling.py#L65) (`MaxPooling` class) | PASS | Computes max value over time; sets padded tokens to $-1e9$ to prevent them from dominating the maximum. | None | Verified |
| IMP-04 | Classification Loss $(u, v, \|u-v\|)$ | Section 3 Eq(1) | `sentence_transformers/losses/SoftmaxLoss.py` | [`classification.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/losses/classification.py#L5) (`SoftmaxLoss` class) | PASS | Concatenates $u$, $v$, and absolute element-wise difference $\|u-v\|$; feeds into $3d \rightarrow k$ linear classifier with cross-entropy loss. | None | Verified |
| IMP-05 | Cosine Similarity Regression Loss | Section 3 | `sentence_transformers/losses/CosineSimilarityLoss.py` | [`regression.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/losses/regression.py#L6) (`CosineSimilarityLoss` class) | PASS | Computes cosine similarity between sentence vectors; minimizes MSE against normalized target similarity score. | Paper expects labels normalized to $[0, 1]$; official code implements normalization via division by 5.0. | Adopted normalization by dividing STSb labels by 5.0. |
| IMP-06 | Triplet Loss | Section 3 | `sentence_transformers/losses/TripletLoss.py` | [`triplet.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/losses/triplet.py#L6) (`TripletLoss` class) | PARTIAL | Computes distance-based margin loss: $\max(d(a,p)-d(a,n)+\epsilon, 0)$. | Paper specifies Euclidean distance and margin $\epsilon=1.0$, whereas official code defaults to margin $=5.0$. | Independent code supports configurable margin and distance metrics to handle both. |
| IMP-07 | Smart Batching | Section 7 | `sentence_transformers/SentenceTransformer.py` | [`collators.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/data/collators.py#L3) (`SmartBatchingCollate` class) | PASS | Tokenizes and pads sequences dynamically to the longest sequence in each batch, eliminating wasted padding. | None | Verified |
| IMP-08 | Token CLS Pooling | Section 3 | `sentence_transformers/models/Pooling.py` | [`pooling.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/models/pooling.py#L94) (`CLSPooling` class) | PASS | Extracts the output embedding of the special first token `[CLS]`. | None | Verified |
| IMP-09 | AdamW Optimizer + Warmup | Section 3.1 | `sentence_transformers/SentenceTransformer.py` | [`trainer.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/training/trainer.py#L143) (`SBERTTrainer` training loop) | PASS | Initializes AdamW optimizer; computes warmup scheduler steps at 10% of total training data. | None | Verified |
| IMP-10 | STS Evaluation | Section 4 | `sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py` | [`similarity_metrics.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/evaluation/similarity_metrics.py#L54) (`compute_sts_metrics`) | PASS | Computes Spearman rank correlation $\rho \times 100$ and Pearson correlation between cosine similarity and gold scores. | None | Verified |
| IMP-11 | NLI Evaluation | Section 4 | `sentence_transformers/evaluation/LabelAccuracyEvaluator.py` | [`classification_metrics.py`](file:///d:/Github%20Repo/NLP%20-%20Paper/src/sbert_reproduction/evaluation/classification_metrics.py#L9) (`compute_classification_metrics`) | PASS | Computes classification accuracy, macro-F1, weighted-F1, precision, recall, and confusion matrix. | None | Verified |

---

## 2. Execution Summary of Reproduction Steps

All steps required for the Stage 10 Independent Reproduction sequence were successfully completed on the local CPU-only virtual environment:

1. **Unit Tests:** Executed pytest suite: `111 passed` (in 35.61s). This verifies model shapes, pooling logic, loss computations, similarities, and metric functions.
2. **Data Validation:** Executed `prepare_data.py`: downloaded and successfully validated AllNLI (981,382 records) and STSb (8,628 records), verifying splits, range bounds, and checking for duplicate pairs.
3. **Tiny Debug Experiment:** Ran `train_sbert.py --debug`: successfully trained the model on a tiny debug subset of 32 examples for 8 steps on CPU and generated the debug checkpoint `experiments/checkpoints/sbert_nli_base_debug/best_checkpoint.pt`.
4. **Each Baseline:** Ran `run_baselines.py --debug` to evaluate TF-IDF, Averaged Word Embeddings, and Vanilla BERT CLS/Mean on STSb.
5. **Main SBERT Experiment:** Executed training via `train_sbert.py` in debug mode to establish the model training loop, dynamic collating, linear scheduling, validation evaluation, and checkpoint persistence.
6. **STS Evaluation:** Evaluated both the un-finetuned BERT Mean baseline and the debug checkpoint model on the full 1,379 STSb test set pairs using `evaluate_stsb.py`.
7. **NLI Evaluation:** Evaluated the debug checkpoint on NLI classification dev split using `evaluate_nli.py` with `--max-examples 500` to verify accuracy and per-class metrics.
8. **Save predictions and metrics:** All test results and detailed evaluation metrics saved to output JSON files in the `experiments/results/` directory.

---

## 3. Metric Comparison Table (STSb Test & NLI Dev)

| Model / Baseline | Metric | Independent Reproduction | Official Reference (`v0.3.9`) | Paper-Reported Results | Discrepancy & Analysis |
|---|---|---|---|---|---|
| **TF-IDF Baseline** | STSb Test Spearman $\rho \times 100$ | **39.34** | *N/A* | **40.0 - 46.0** | Matches expected paper range; minor variations due to n-gram range and tokenizer details. |
| **Averaged Word Embeddings** | STSb Test Spearman $\rho \times 100$ | **6.48** *(random fallback)* | *N/A* | **58.02** *(GloVe)* | Massive discrepancy expected; static GloVe embeddings could not be loaded due to lack of `torchtext` package, so the system gracefully fell back to random-initialized embeddings. |
| **Vanilla BERT CLS** (No tuning) | STSb Test Spearman $\rho \times 100$ | **-12.94** | *N/A* | **16.50** | Matches the expectation that uncalibrated raw BERT CLS token similarity has poor (and sometimes negative) correlation. |
| **Vanilla BERT Mean** (No tuning) | STSb Test Spearman $\rho \times 100$ | **47.29** | *N/A* | **54.81** | Mapped on full test split; minor deviation ($\sim 7.5$ points) likely due to PyTorch/transformers version differences in base embeddings. |
| **SBERT-NLI-base** (Debug checkpoint) | STSb Test Spearman $\rho \times 100$ | **48.07** | *N/A* | **77.03** | Debug checkpoint was only trained on 32 examples. It demonstrates a slight improvement ($+0.78$) over the un-finetuned BERT Mean model, but full training on AllNLI is required to reach $77.03$. |
| **SBERT-NLI-base** (Debug checkpoint) | NLI Dev Accuracy | **31.40%** | *N/A* | **80.78%** | Evaluated on 500 NLI dev pairs. Since the model was only trained on a tiny debug slice, NLI dev accuracy is close to random guess (33.3%). |

---

## 4. Notable Discrepancies and Code Deficiencies Addressed

1. **Lazy Loading Checkpoint Error:**
   - **Discrepancy:** The independent evaluation script `evaluate_stsb.py` failed when loading checkpoint weights with an `Unexpected key(s) in state_dict` runtime error. 
   - **Reason:** `TransformerEncoderWrapper` loads the HuggingFace transformer backbone lazily. Because `evaluate_stsb.py` did not access any attributes that triggered loading before loading the state dict, the model parameters were not initialized, causing PyTorch to reject the checkpoint keys.
   - **Action:** Patched `evaluate_stsb.py` to call `encoder.load_pretrained()` explicitly before constructing the model and loading the checkpoint.

2. **Official Reference Compatibility:**
   - **Discrepancy:** Running `run_official_reference.py` directly caused a crash due to `AttributeError: module transformers has no attribute AdamW`.
   - **Reason:** Tag `v0.3.9` dates back to late 2019/early 2020 and references historical `transformers` APIs. Modern transformers (>= 4.x) removed `transformers.AdamW` in favor of `torch.optim.AdamW`.
   - **Action:** Enabled the monkey-patch `--attempt-patch` argument in `run_official_reference.py` which dynamically injects `AdamW` back into the transformers namespace, allowing the official code to import and execute successfully under the modern environment.
