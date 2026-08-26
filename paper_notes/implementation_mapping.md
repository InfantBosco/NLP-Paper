# Implementation Mapping Table

This table maps paper requirements and equations directly to code symbols in both the official reference repository (`v0.3.9`) and our independent implementation (`src/sbert_reproduction`).

| ID | Paper Requirement | Paper Location | Official Reference (`v0.3.9`) | Independent Code (`sbert_reproduction`) | Status | Evidence | Discrepancy | Action |
|---|---|---|---|---|---|---|---|---|
| IMP-01 | Siamese Encoder Architecture | Section 3 | `sentence_transformers/SentenceTransformer.py` | `sbert_reproduction.models.sbert_model.SBERTModel` | PASS | Code weights tied during pair forward pass | None | Verified |
| IMP-02 | Token Mean Pooling | Section 3 | `sentence_transformers/models/Pooling.py` | `sbert_reproduction.models.pooling.MeanPooling` | PASS | Masks padding tokens correctly | None | Verified |
| IMP-03 | Token Max Pooling | Section 3 | `sentence_transformers/models/Pooling.py` | `sbert_reproduction.models.pooling.MaxPooling` | PASS | Sets padding tokens to $-1e9$ | None | Verified |
| IMP-04 | Classification Loss $(u, v, \|u-v\|)$ | Section 3 Eq(1) | `sentence_transformers/losses/SoftmaxLoss.py` | `sbert_reproduction.losses.classification.SoftmaxLoss` | PASS | Linear classifier $(3d \rightarrow 3)$ | None | Verified |
| IMP-05 | Cosine Similarity Regression Loss | Section 3 | `sentence_transformers/losses/CosineSimilarityLoss.py` | `sbert_reproduction.losses.regression.CosineSimilarityLoss` | PASS | MSE loss on cosine similarity | Target scaled $score / 5.0$ | Adopt normalization |
| IMP-06 | Triplet Loss | Section 3 | `sentence_transformers/losses/TripletLoss.py` | `sbert_reproduction.losses.triplet.TripletLoss` | PARTIAL | Euclidean margin loss | Paper $\epsilon=1$ vs Code margin $=5$ | Support both as config |
| IMP-07 | Smart Batching | Section 7 | `sentence_transformers/SentenceTransformer.py` | `sbert_reproduction.data.collators.SmartBatchingCollate` | PASS | Length-sorted dynamic batching | None | Verified |
