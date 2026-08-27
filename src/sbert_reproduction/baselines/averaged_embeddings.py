"""
Averaged Word Embeddings baseline.

Description
-----------
Each sentence is encoded as the arithmetic mean of its token word vectors.
Cosine similarity between averaged vectors is used as the similarity score.

Embedding source
----------------
This implementation attempts to load GloVe 6B 300d vectors from the
torchtext dataset API (legal, public domain release by Stanford NLP).
If torchtext is unavailable or the download fails, the implementation
falls back to a random-initialised embedding matrix (vocabulary built from
the corpus), which serves as a random-baseline lower bound.

The fallback is clearly flagged in the output report.

Configuration
-------------
  dim:        300  (GloVe 300d) or configurable for random fallback
  source:     "glove.6B.300d" or "random_fallback"

Limitations
-----------
- GloVe vectors do not capture context (each word has a single embedding).
- OOV words are represented as zero vectors (no sub-word handling).
- Random-fallback mode is meaningless as a similarity model; it serves only
  as a lower-bound sanity check.
- The GloVe download (~822 MB) is performed lazily on first use.

Paper reference: Section 5 — averaged GloVe embeddings baseline.
Official code reference: sentence-transformers benchmarks (independent reimpl.)
"""

from __future__ import annotations

import time
import tracemalloc
import warnings
from typing import Dict, List, Optional

import numpy as np


class AveragedWordEmbeddingsBaseline:
    """
    Averaged word embedding baseline.

    Args:
        dim:    Embedding dimensionality.
        seed:   Random seed (used only for random-fallback initialisation).
        source: One of ``"glove"`` (try to load GloVe) or ``"random"``.
    """

    description = (
        "Averaged word embeddings with cosine similarity. "
        "Attempts GloVe 6B 300d; falls back to random init on failure."
    )

    def __init__(
        self,
        dim: int = 300,
        seed: int = 42,
        source: str = "glove",
    ) -> None:
        self.dim     = dim
        self.seed    = seed
        self.source  = source
        self.embedding_dict: Dict[str, np.ndarray] = {}
        self._actual_source = "not_loaded"

    # ------------------------------------------------------------------
    def load_glove(self) -> bool:
        """
        Attempt to load GloVe 6B 300d via torchtext.

        Returns True on success, False on failure.
        """
        try:
            from torchtext.vocab import GloVe
            glove = GloVe(name="6B", dim=self.dim)
            for token, idx in glove.stoi.items():
                self.embedding_dict[token] = glove.vectors[idx].numpy()
            self.dim = self.dim
            self._actual_source = "glove.6B.300d"
            print(f"[AveragedEmbeddings] Loaded GloVe 6B {self.dim}d "
                  f"({len(self.embedding_dict):,} tokens)")
            return True
        except Exception as exc:
            warnings.warn(
                f"GloVe load failed ({exc}). "
                "Falling back to random-initialised embeddings. "
                "Results in this mode are meaningless for comparison.",
                RuntimeWarning,
            )
            return False

    # ------------------------------------------------------------------
    def build_random_vocab(self, texts: List[str]) -> None:
        """
        Build a random vocabulary from *texts* (fallback when GloVe unavailable).

        Each unique whitespace-split token receives a fixed random vector
        seeded deterministically.
        """
        rng = np.random.RandomState(self.seed)
        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())
        for word in vocab:
            self.embedding_dict[word] = rng.randn(self.dim).astype(np.float32)
        self._actual_source = "random_fallback"
        print(f"[AveragedEmbeddings] Built random vocab of {len(self.embedding_dict):,} tokens "
              f"(dim={self.dim}). This is a random baseline — results are not meaningful.")

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> None:
        """
        Load embeddings.  If source is "glove", tries GloVe first and
        falls back to random if unavailable.
        """
        if self.source == "glove":
            if not self.load_glove():
                self.build_random_vocab(texts)
        else:
            self.build_random_vocab(texts)

    # ------------------------------------------------------------------
    def encode_sentence(self, text: str) -> np.ndarray:
        """Average word vectors for *text*; return zero vector for OOV sentences."""
        tokens  = text.lower().split()
        vectors = [self.embedding_dict[tok] for tok in tokens if tok in self.embedding_dict]
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vectors, axis=0)

    # ------------------------------------------------------------------
    def predict_similarity(
        self,
        texts_a: List[str],
        texts_b: List[str],
    ) -> np.ndarray:
        sims = []
        for ta, tb in zip(texts_a, texts_b):
            va = self.encode_sentence(ta)
            vb = self.encode_sentence(tb)
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            if na < 1e-9 or nb < 1e-9:
                sims.append(0.0)
            else:
                sims.append(float(np.dot(va, vb) / (na * nb)))
        return np.array(sims)

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return len(self.embedding_dict) * self.dim

    # ------------------------------------------------------------------
    def run_and_report(
        self,
        all_texts: List[str],
        test_texts_a: List[str],
        test_texts_b: List[str],
        gold_scores: np.ndarray,
    ) -> dict:
        from sbert_reproduction.evaluation.similarity_metrics import compute_sts_metrics

        self.fit(all_texts)

        tracemalloc.start()
        t0 = time.perf_counter()
        predictions = self.predict_similarity(test_texts_a, test_texts_b)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = compute_sts_metrics(predictions, gold_scores)

        return {
            "baseline": "averaged_word_embeddings",
            "description": self.description,
            "configuration": {
                "dim":    self.dim,
                "source": self._actual_source,
            },
            "dataset": "STSBenchmark",
            "split": "test",
            "embedding_size": self.dim,
            "parameter_count": self.parameter_count(),
            "spearman_rho": metrics["spearman_rho"],
            "pearson_r":    metrics["pearson_r"],
            "mse":          metrics["mse"],
            "mae":          metrics["mae"],
            "inference_time_sec": elapsed,
            "peak_memory_mb":     peak / (1024 ** 2),
            "seed": self.seed,
            "limitations": (
                f"Actual embedding source: {self._actual_source}. "
                "No context sensitivity (single vector per word). "
                "OOV handled as zero vector. "
                "random_fallback results are NOT comparable to GloVe results."
            ),
        }
