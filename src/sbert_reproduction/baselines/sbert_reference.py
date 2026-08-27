"""
Standard Sentence-Transformers reference baseline (external comparison only).

This module wraps the official sentence-transformers library for use
ONLY as an external reference comparison.  It is NOT part of the
independent reproduction.

IMPORTANT: This class uses sentence-transformers, which imports the
official model.  It must NEVER be imported from sbert_reproduction
model modules.  It lives in sbert_reproduction.baselines only to
provide external comparison metrics.

Configuration
-------------
  model_name:    "sentence-transformers/all-MiniLM-L6-v2" (fast reference)
               OR "sentence-transformers/bert-base-nli-mean-tokens" (paper model)

Limitations
-----------
- Depends on the official sentence-transformers library.
- Results may differ from reproduction due to checkpoint, tokenizer
  version, or pooling implementation differences.
- Cannot be used in unit tests that must be independent of official code.

Paper reference: External comparison only — not part of the reproduction.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import List

import numpy as np


class SBERTReferenceBaseline:
    """
    Thin wrapper around the official sentence-transformers library for
    external comparison only.

    Args:
        model_name:  Official sentence-transformers model identifier.
        seed:        Stored for provenance record.
    """

    description = (
        "Official sentence-transformers model used as external reference. "
        "NOT part of the independent reproduction."
    )

    def __init__(
        self,
        model_name: str = "sentence-transformers/bert-base-nli-mean-tokens",
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.seed       = seed
        self._model     = None

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for SBERTReferenceBaseline. "
                    "Install with: pip install sentence-transformers"
                ) from e

    def predict_similarity(
        self,
        texts_a: List[str],
        texts_b: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        self._load()
        emb_a = self._model.encode(texts_a, batch_size=batch_size, normalize_embeddings=True)
        emb_b = self._model.encode(texts_b, batch_size=batch_size, normalize_embeddings=True)
        return np.sum(emb_a * emb_b, axis=1)

    def parameter_count(self) -> int:
        self._load()
        return sum(p.numel() for p in self._model.parameters())

    def run_and_report(
        self,
        test_texts_a: List[str],
        test_texts_b: List[str],
        gold_scores: np.ndarray,
        batch_size: int = 32,
    ) -> dict:
        from sbert_reproduction.evaluation.similarity_metrics import compute_sts_metrics

        self._load()

        tracemalloc.start()
        t0 = time.perf_counter()
        predictions = self.predict_similarity(test_texts_a, test_texts_b, batch_size)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = compute_sts_metrics(predictions, gold_scores)

        return {
            "baseline": "sbert_official_reference",
            "description": self.description,
            "configuration": {
                "model_name": self.model_name,
                "batch_size": batch_size,
            },
            "dataset": "STSBenchmark",
            "split": "test",
            "embedding_size": -1,  # varies by model
            "parameter_count": self.parameter_count(),
            "spearman_rho": metrics["spearman_rho"],
            "pearson_r":    metrics["pearson_r"],
            "mse":          metrics["mse"],
            "mae":          metrics["mae"],
            "inference_time_sec": elapsed,
            "peak_memory_mb":     peak / (1024 ** 2),
            "seed": self.seed,
            "result_source": "OFFICIAL_CODE",
            "limitations": (
                "Uses official sentence-transformers library. "
                "Results depend on library version and checkpoint. "
                "NOT part of the independent reproduction."
            ),
        }
