# Stage 0: Official Repository Audit & Inspection Report

**Paper Title:** Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks  
**Authors:** Nils Reimers and Iryna Gurevych (UKP-TUDA)  
**Official Repository:** https://github.com/UKPLab/sentence-transformers  
**Historical Version Tag:** `v0.3.9` (https://github.com/UKPLab/sentence-transformers/tree/v0.3.9)  

---

## 1. Current Workspace Contents

The root workspace (`d:\Github Repo\NLP - Paper`) currently contains:
- `NLP.pdf`: The primary research source PDF (arXiv:1908.10084v1 [cs.CL] 27 Aug 2019).
- `official_reference/`:
  - `README.md`: Explains isolation of reference code.
  - `sentence-transformers-v0.3.9/`: Cloned reference implementation at git tag `v0.3.9`.
- `.git/`: Workspace version control repository.

---

## 2. Environment Audit

- **Operating System:** Windows (10/11)
- **CPU:** 12th Gen Intel(R) Core(TM) i5-12450HX (8 physical cores, 12 logical threads)
- **GPU:** NVIDIA GeForce RTX 2050 (4 GB VRAM, Driver 581.86, CUDA 13.0 capability)
- **System Memory:** 11.71 GB total RAM (~1.60 GB free physical memory)
- **Python Interpreters Found:**
  - `C:\Python314\python.exe` (Python 3.14.3)
  - `C:\Users\infan\AppData\Local\Programs\Python\Python313\python.exe` (Python 3.13.5)
- **Status of ML Dependencies:**
  - PyTorch and HuggingFace Transformers are not currently installed in the global base Python environments.
  - A clean virtual environment will be created during Stage 3 with pinned dependencies.

---

## 3. Official Repository Structure (Historical v0.3.9)

```
official_reference/sentence-transformers-v0.3.9/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── sentence_transformers/
│   ├── SentenceTransformer.py       # Main model wrapper (inherits nn.Sequential)
│   ├── LoggingHandler.py
│   ├── util.py
│   ├── models/                      # Modules concatenated sequentially
│   │   ├── Transformer.py           # Wraps HuggingFace AutoModel / AutoTokenizer
│   │   ├── Pooling.py               # Token pooling (MEAN, MAX, CLS, MEAN_SQRT_LEN)
│   │   ├── BERT.py, RoBERTa.py, etc.# Legacy transformer modules
│   │   ├── Dense.py                 # Linear feed-forward layer
│   │   └── Normalize.py             # L2 normalization layer
│   ├── losses/                      # Siamese & Triplet objective functions
│   │   ├── SoftmaxLoss.py           # 3-way NLI classification loss
│   │   ├── CosineSimilarityLoss.py  # MSE loss on cosine similarity
│   │   ├── TripletLoss.py           # Margin triplet loss
│   │   ├── BatchHardTripletLoss.py
│   │   ├── MultipleNegativesRankingLoss.py
│   │   └── MSELoss.py
│   ├── readers/                     # Dataset loading utilities
│   │   ├── NLIDataReader.py
│   │   ├── STSDataReader.py
│   │   ├── TripletReader.py
│   │   └── InputExample.py
│   ├── evaluation/                  # Evaluators run during training / test
│   │   ├── EmbeddingSimilarityEvaluator.py # Spearman & Pearson correlations
│   │   ├── LabelAccuracyEvaluator.py
│   │   ├── TripletEvaluator.py
│   │   └── InformationRetrievalEvaluator.py
│   └── datasets/
│       ├── SentencesDataset.py      # PyTorch Dataset wrapper
│       └── EncodeDataset.py
└── examples/
    ├── training/
    │   ├── nli/training_nli.py
    │   ├── avg_word_embeddings/
    │   └── wikipedia_sections/
    └── evaluation/
        └── evaluation_stsbenchmark.py
```

---

## 4. Key Components Audit & Paper-to-Code Mapping

| Finding ID | File Path | Class / Function | Paper Ref | Explanation | Confidence |
|---|---|---|---|---|---|
| MAP-01 | `sentence_transformers/models/Transformer.py` | `Transformer` | Section 3 | Wraps Huggingface model & tokenizer to output `token_embeddings`, `cls_token_embeddings`, and `attention_mask`. | HIGH |
| MAP-02 | `sentence_transformers/models/Pooling.py` | `Pooling` | Section 3 | Computes sentence vector via MEAN (default), MAX, or CLS token with attention mask zeroing. | HIGH |
| MAP-03 | `sentence_transformers/losses/SoftmaxLoss.py` | `SoftmaxLoss` | Section 3 eq(1), Fig 1 | Concatenates $(u, v, |u - v|)$ into `nn.Linear(3d, 3)` with `CrossEntropyLoss`. | HIGH |
| MAP-04 | `sentence_transformers/losses/CosineSimilarityLoss.py` | `CosineSimilarityLoss` | Section 3, Fig 2 | Computes $\text{cosine\_similarity}(u, v)$ and minimizes MSE against scaled target $[0, 1]$. | HIGH |
| MAP-05 | `sentence_transformers/losses/TripletLoss.py` | `TripletLoss` | Section 3 | Computes $\max(\|s_a - s_p\| - \|s_a - s_n\| + \text{margin}, 0)$. Code default margin=5 vs Paper margin=1. | HIGH |
| MAP-06 | `sentence_transformers/evaluation/EmbeddingSimilarityEvaluator.py` | `EmbeddingSimilarityEvaluator` | Section 4 | Evaluates cosine, Manhattan, Euclidean distance against gold labels using Spearman $\rho$ and Pearson $r$. | HIGH |
| MAP-07 | `examples/training/nli/training_nli.py` | N/A | Section 3.1 | Fine-tunes BERT on AllNLI (SNLI + MultiNLI) for 1 epoch, batch size 16, lr $2e-5$, linear warmup 10%. | HIGH |
| MAP-08 | `sentence_transformers/readers/NLIDataReader.py` | `NLIDataReader` | Section 3.1 | Reads TSV with label mapping `{"contradiction": 0, "entailment": 1, "neutral": 2}`. | HIGH |

---

## 5. Paper vs Official Repository Discrepancies & Ambiguities

1. **Triplet Loss Margin Discrepancy:**
   - **Paper Section 3:** Explicitly states Euclidean distance metric with margin $\epsilon = 1.0$.
   - **Official Code (`TripletLoss.py`):** Default `triplet_margin` is set to `5` (`triplet_margin: float = 5`).
2. **NLI Softmax Classification Vector Concatenation:**
   - **Paper Section 3:** Formula is $o = \text{softmax}(W_t (u, v, |u - v|))$.
   - **Official Code (`SoftmaxLoss.py`):** Offers options for `(u, v)`, `|u - v|`, and `u * v`. Default is `(u, v, |u - v|)` matching paper equation (1).
3. **STSb Target Scaling:**
   - **Paper Section 4.2:** Mentions STS labels between 0 and 5.
   - **Official Code (`training_nli.py`):** Normalizes STSb target scores by dividing by 5.0 ($score / 5.0$).

---

## 6. Historical vs Current Repository Differences

- **v0.3.9 (Paper Era):** Relies on `transformers>=3.1.0,<3.6.0` and PyTorch `1.6.0`. Custom modular wrapper subclassing `nn.Sequential`.
- **Current Main Branch:** Uses modern HuggingFace `transformers>=4.0.0`, PyTorch 2.x, native HuggingFace Hub integrations (`AutoModel`), updated dataset loaders (`datasets` library), and trainer abstractions.

---

## 7. Environment & Compatibility Risks

1. **PyTorch/Transformers Version Shift:** `v0.3.9` pins `transformers<3.6.0`. Running `v0.3.9` directly under modern PyTorch/Transformers will trigger breaking API changes (e.g., return dicts vs tuples, `attention_mask` tensor signatures).
2. **GPU Memory Constraint:** VRAM is 4 GB (RTX 2050). Training `bert-base-uncased` with batch size 16 requires gradient checkpointing or mixed precision (`fp16`) to avoid CUDA Out-Of-Memory (OOM).

---

## 8. Recommended Next Steps for Reproduction

1. Complete Stage 1 (Paper Analysis & Assumptions Matrix).
2. Complete Stage 2 (Reproduction Specification & Experiment Matrix).
3. Build clean isolated virtual environment in Stage 3 (`sbert-reproduction` scaffold).
