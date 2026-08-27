"""
Similarity metrics for STS evaluation.

Implements:
- compute_sts_metrics  — Spearman, Pearson, MSE, MAE
- rescale_scores       — convert raw label range to [0, 1]
- EvaluationRecord     — dataclass capturing full result provenance

Paper reference:
  Reimers & Gurevych (2019), Table 2 — Spearman rank correlation ×100
  reported on STS benchmark test set.

Result source classification:
  PAPER | OFFICIAL_CODE | INDEPENDENT_REPRODUCTION | EXTENSION
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from scipy.stats import spearmanr, pearsonr
from typing import Optional


def rescale_scores(
    scores: np.ndarray,
    src_min: float = 0.0,
    src_max: float = 5.0,
    dst_min: float = 0.0,
    dst_max: float = 1.0,
) -> np.ndarray:
    """
    Linearly rescale *scores* from [src_min, src_max] to [dst_min, dst_max].

    STS-B labels are in [0, 5]; normalise to [0, 1] before MSE with cosine.

    Args:
        scores:  1-D array of raw scores.
        src_min: Minimum of source range (default 0 for STS-B).
        src_max: Maximum of source range (default 5 for STS-B).
        dst_min: Minimum of target range.
        dst_max: Maximum of target range.

    Returns:
        Rescaled np.ndarray of same shape.
    """
    scores = np.array(scores, dtype=float)
    denom = src_max - src_min
    if abs(denom) < 1e-12:
        return np.full_like(scores, dst_min)
    return dst_min + (scores - src_min) * (dst_max - dst_min) / denom


def compute_sts_metrics(
    predictions: np.ndarray,
    gold_labels: np.ndarray,
) -> dict:
    """
    Compute Spearman ρ, Pearson r, MSE, and MAE between predictions and labels.

    Both arrays are expected to be in the same scale (either both raw or both
    normalised — the caller is responsible for consistent scaling).

    Returns a dict with keys:
        spearman_rho  — Spearman correlation × 100  (as reported in paper)
        pearson_r     — Pearson r × 100
        mse           — Mean Squared Error
        mae           — Mean Absolute Error
    """
    predictions = np.array(predictions, dtype=float)
    gold_labels  = np.array(gold_labels,  dtype=float)

    spearman_rho, _ = spearmanr(predictions, gold_labels)
    pearson_r,    _ = pearsonr( predictions, gold_labels)
    mse = float(np.mean((predictions - gold_labels) ** 2))
    mae = float(np.mean(np.abs(predictions - gold_labels)))

    return {
        "spearman_rho": float(spearman_rho * 100.0) if not np.isnan(spearman_rho) else 0.0,
        "pearson_r":    float(pearson_r    * 100.0) if not np.isnan(pearson_r)    else 0.0,
        "mse":          mse,
        "mae":          mae,
    }


# ---------------------------------------------------------------------------
# EvaluationRecord — full provenance for every result
# ---------------------------------------------------------------------------

@dataclass
class EvaluationRecord:
    """
    Captures a single evaluation result with full provenance.

    result_source must be one of:
        PAPER | OFFICIAL_CODE | INDEPENDENT_REPRODUCTION | EXTENSION
    """

    # Identity
    experiment_name: str
    model_description: str
    result_source: str   # PAPER | OFFICIAL_CODE | INDEPENDENT_REPRODUCTION | EXTENSION

    # Data
    dataset: str = "STSBenchmark"
    split: str = "test"
    num_examples: int = 0

    # Model metadata
    embedding_size: int = 0
    parameter_count: int = 0
    checkpoint: Optional[str] = None
    configuration: dict = field(default_factory=dict)
    seed: int = 42

    # Metrics
    spearman_rho: float = 0.0
    pearson_r: float = 0.0
    mse: float = 0.0
    mae: float = 0.0

    # Performance
    inference_time_sec: float = 0.0
    peak_memory_mb: float = 0.0
    sentences_per_sec: float = 0.0

    # Software
    software_versions: dict = field(default_factory=dict)
    metric_implementation: str = "sbert_reproduction.evaluation.similarity_metrics"

    # Limitations / notes
    limitations: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
