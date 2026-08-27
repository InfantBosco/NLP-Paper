"""
TF-IDF baseline with cosine similarity.

Description
-----------
Represents each sentence as a TF-IDF weighted bag-of-n-grams vector and
computes cosine similarity between pairs.  No learning on sentence pairs;
vocabulary is fitted on the training split.

Configuration
-------------
  max_features: 10000
  ngram_range:  (1, 2)  — unigrams and bigrams

Limitations
-----------
- Ignores word order and semantics entirely.
- OOV tokens outside the training vocabulary are silently dropped.
- Performance is sensitive to the training corpus used for vocabulary fitting.

Paper reference: Section 5 — TF-IDF baseline used for comparison.
Official code reference: sentence-transformers benchmarks (independent reimpl.)
"""

from __future__ import annotations

import time
import tracemalloc
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class TFIDFBaseline:
    """
    TF-IDF vectoriser with cosine similarity for sentence pair scoring.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range:  Tuple (min_n, max_n) for n-gram range.
        seed:         Random seed (stored for reproducibility record; TF-IDF
                      itself is deterministic).
    """

    description = (
        "TF-IDF bag-of-n-grams with cosine similarity. "
        "Vocabulary fitted on training split. No fine-tuning."
    )
    embedding_size = -1   # variable (depends on max_features)

    def __init__(
        self,
        max_features: int = 10_000,
        ngram_range: Tuple[int, int] = (1, 2),
        seed: int = 42,
    ) -> None:
        self.max_features = max_features
        self.ngram_range  = tuple(ngram_range)
        self.seed         = seed
        self.vectorizer   = TfidfVectorizer(
            max_features=max_features,
            ngram_range=self.ngram_range,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, texts: List[str]) -> None:
        """Fit the vocabulary on *texts* (should be the training corpus)."""
        self.vectorizer.fit(texts)
        self._fitted = True
        self.embedding_size = len(self.vectorizer.vocabulary_)

    # ------------------------------------------------------------------
    def predict_similarity(
        self,
        texts_a: List[str],
        texts_b: List[str],
    ) -> np.ndarray:
        """
        Compute cosine similarity for each (texts_a[i], texts_b[i]) pair.

        Returns:
            np.ndarray of shape [n] with values in [-1, 1].
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_similarity().")
        vecs_a = self.vectorizer.transform(texts_a)
        vecs_b = self.vectorizer.transform(texts_b)
        similarities = [
            float(sklearn_cosine(vecs_a[i], vecs_b[i])[0, 0])
            for i in range(len(texts_a))
        ]
        return np.array(similarities)

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        """Number of vocabulary entries (a proxy for model size)."""
        return self.embedding_size if self._fitted else 0

    # ------------------------------------------------------------------
    def run_and_report(
        self,
        train_texts: List[str],
        test_texts_a: List[str],
        test_texts_b: List[str],
        gold_scores: np.ndarray,
    ) -> dict:
        """
        Fit on *train_texts*, predict on test pairs, measure timing & memory.

        Returns:
            Full metadata dict suitable for :class:`EvaluationRecord`.
        """
        from sbert_reproduction.evaluation.similarity_metrics import compute_sts_metrics

        # Fit
        self.fit(train_texts)

        # Timed + memory-tracked prediction
        tracemalloc.start()
        t0 = time.perf_counter()
        predictions = self.predict_similarity(test_texts_a, test_texts_b)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = compute_sts_metrics(predictions, gold_scores)

        return {
            "baseline": "tfidf",
            "description": self.description,
            "configuration": {
                "max_features": self.max_features,
                "ngram_range": list(self.ngram_range),
                "sublinear_tf": True,
            },
            "dataset": "STSBenchmark",
            "split": "test",
            "embedding_size": self.embedding_size,
            "parameter_count": self.parameter_count(),
            "spearman_rho": metrics["spearman_rho"],
            "pearson_r":    metrics["pearson_r"],
            "mse":          metrics["mse"],
            "mae":          metrics["mae"],
            "inference_time_sec": elapsed,
            "peak_memory_mb":     peak / (1024 ** 2),
            "seed": self.seed,
            "limitations": (
                "Ignores word order and semantics. "
                "OOV tokens outside training vocabulary are dropped silently. "
                "Vocabulary size truncated to max_features."
            ),
        }
