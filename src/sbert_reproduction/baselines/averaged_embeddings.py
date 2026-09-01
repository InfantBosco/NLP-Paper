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
    def load_glove(self, data_dir: str = "data") -> bool:
        """
        Attempt to load GloVe vectors from disk or via gensim.downloader.

        Returns True on success, False on failure.
        """
        import os

        glove_dir = os.path.join(data_dir, "glove")
        os.makedirs(glove_dir, exist_ok=True)
        txt_filename = f"glove.6B.{self.dim}d.txt"
        txt_path = os.path.join(glove_dir, txt_filename)

        # 1. Try loading from existing text file
        if os.path.exists(txt_path):
            try:
                print(f"[AveragedEmbeddings] Loading GloVe embeddings from {txt_path}...")
                count = 0
                with open(txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.rstrip().split(" ")
                        word = parts[0]
                        vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                        self.embedding_dict[word] = vec
                        count += 1
                self._actual_source = f"glove.6B.{self.dim}d"
                print(f"[AveragedEmbeddings] Successfully loaded {count:,} GloVe vectors ({self.dim}d).")
                return True
            except Exception as exc:
                print(f"[AveragedEmbeddings] Error reading {txt_path}: {exc}")

        # 2. Try loading via gensim.downloader
        try:
            import gensim.downloader as api
            model_name = f"glove-wiki-gigaword-{self.dim}" if self.dim in (50, 100, 200, 300) else "glove-wiki-gigaword-100"
            print(f"[AveragedEmbeddings] Loading GloVe via gensim ({model_name})...")
            glove_model = api.load(model_name)
            for word in glove_model.key_to_index:
                self.embedding_dict[word] = glove_model[word]
            self.dim = glove_model.vector_size
            self._actual_source = f"gensim.{model_name}"
            print(f"[AveragedEmbeddings] Loaded {len(self.embedding_dict):,} GloVe vectors ({self.dim}d) via gensim.")
            return True
        except Exception as exc:
            warnings.warn(f"Gensim GloVe download failed: {exc}")

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
