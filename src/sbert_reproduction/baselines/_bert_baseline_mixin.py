"""
Shared BERT encoder baseline helper.

Both VanillaBERTCLSBaseline and VanillaBERTMeanBaseline use this mixin
to avoid duplicating the batch-encoding and reporting logic.

These are *un-finetuned* pretrained BERT models used as baselines:
- CLS: extract the [CLS] token representation
- Mean: average all non-padding token representations

Paper reference: Section 5 — BERT without fine-tuning, Table 2.

Official code reference: sentence-transformers benchmarks (independent reimpl.)
"""

from __future__ import annotations

import time
import tracemalloc
from typing import List

import numpy as np
import torch


class _BERTBaselineMixin:
    """
    Shared encoding + reporting logic for vanilla BERT baselines.

    Subclasses must set:
        self.model         — a SentenceEncoder instance
        self.model_name    — str
        self.pooling_mode  — str
    """

    _description_template = (
        "Un-finetuned {model_name} with {pooling_mode} pooling. "
        "No task-specific fine-tuning."
    )

    @property
    def description(self) -> str:
        return self._description_template.format(
            model_name=self.model_name,
            pooling_mode=self.pooling_mode,
        )

    # ------------------------------------------------------------------
    def _get_tokenizer(self):
        from sbert_reproduction.models.encoder import TokenizerWrapper
        if not hasattr(self, "_tokenizer_wrapper"):
            self._tokenizer_wrapper = TokenizerWrapper(
                model_name=self.model_name,
                max_seq_length=getattr(self, "max_seq_length", 128),
            )
        return self._tokenizer_wrapper

    # ------------------------------------------------------------------
    def encode_sentences(
        self,
        sentences: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Tokenise and encode *sentences* in batches; return numpy embeddings.
        """
        tokenizer = self._get_tokenizer()
        embeddings = self.model.encode_text(tokenizer, sentences, batch_size=batch_size)
        return embeddings.numpy()

    # ------------------------------------------------------------------
    def predict_similarity(
        self,
        texts_a: List[str],
        texts_b: List[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        emb_a = self.encode_sentences(texts_a, batch_size)
        emb_b = self.encode_sentences(texts_b, batch_size)
        # Cosine similarity
        norms_a = np.linalg.norm(emb_a, axis=1, keepdims=True).clip(min=1e-9)
        norms_b = np.linalg.norm(emb_b, axis=1, keepdims=True).clip(min=1e-9)
        return np.sum((emb_a / norms_a) * (emb_b / norms_b), axis=1)

    # ------------------------------------------------------------------
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ------------------------------------------------------------------
    def embedding_size(self) -> int:
        try:
            return self.model.encoder.hidden_size
        except Exception:
            return -1

    # ------------------------------------------------------------------
    def run_and_report(
        self,
        test_texts_a: List[str],
        test_texts_b: List[str],
        gold_scores: np.ndarray,
        batch_size: int = 32,
        seed: int = 42,
    ) -> dict:
        """
        Predict similarities for all test pairs and return full metrics dict.
        """
        from sbert_reproduction.evaluation.similarity_metrics import compute_sts_metrics
        import sys

        tracemalloc.start()
        t0 = time.perf_counter()
        predictions = self.predict_similarity(test_texts_a, test_texts_b, batch_size)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        metrics = compute_sts_metrics(predictions, gold_scores)

        return {
            "baseline": f"bert_{self.pooling_mode}",
            "description": self.description,
            "configuration": {
                "model_name":    self.model_name,
                "pooling_mode":  self.pooling_mode,
                "max_seq_length": getattr(self, "max_seq_length", 128),
                "batch_size":    batch_size,
            },
            "dataset": "STSBenchmark",
            "split": "test",
            "embedding_size": self.embedding_size(),
            "parameter_count": self.parameter_count(),
            "spearman_rho": metrics["spearman_rho"],
            "pearson_r":    metrics["pearson_r"],
            "mse":          metrics["mse"],
            "mae":          metrics["mae"],
            "inference_time_sec": elapsed,
            "peak_memory_mb":     peak / (1024 ** 2),
            "seed": seed,
            "limitations": (
                "No task-specific fine-tuning. "
                "BERT [CLS] and mean representations are not trained for semantic similarity. "
                "Scores are expected to be lower than fine-tuned SBERT."
            ),
        }
