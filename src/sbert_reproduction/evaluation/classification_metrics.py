"""
Classification metrics — accuracy, macro-F1, weighted-F1, per-class stats,
confusion matrix, and confidence statistics.

Used by NLI / classification evaluation (Stage 8).

Official code reference:
  sentence-transformers evaluation (independent re-implementation).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from typing import List, Optional, Dict, Any


def compute_classification_metrics(
    predictions: np.ndarray,
    gold_labels: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    label_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute classification metrics: accuracy, macro/weighted F1, per-class
    precision / recall / F1, confusion matrix, and confidence statistics.

    Args:
        predictions:    1-D array of predicted class indices.
        gold_labels:    1-D array of ground-truth class indices.
        probabilities:  Optional 2-D array of predicted class probabilities [N, C].
        label_names:    Optional list of string names for each class index.

    Returns:
        Dict with keys:
            accuracy, macro_f1, weighted_f1,
            per_class (list of {label, precision, recall, f1}),
            confusion_matrix (list-of-lists),
            confidence_stats (optional dict),
            num_examples.
    """
    predictions = np.array(predictions)
    gold_labels  = np.array(gold_labels)
    classes = np.unique(np.concatenate([predictions, gold_labels]))

    if label_names is None:
        label_names = [str(c) for c in classes]

    acc         = float(accuracy_score(gold_labels, predictions))
    macro_f1    = float(f1_score(gold_labels, predictions, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(gold_labels, predictions, average="weighted", zero_division=0))

    prec_per_class = precision_score(gold_labels, predictions, average=None, zero_division=0, labels=list(classes))
    rec_per_class  = recall_score(   gold_labels, predictions, average=None, zero_division=0, labels=list(classes))
    f1_per_class   = f1_score(       gold_labels, predictions, average=None, zero_division=0, labels=list(classes))

    per_class = []
    for i, cls_idx in enumerate(classes):
        name = label_names[i] if i < len(label_names) else str(cls_idx)
        per_class.append({
            "label":     name,
            "precision": float(prec_per_class[i]),
            "recall":    float(rec_per_class[i]),
            "f1":        float(f1_per_class[i]),
        })

    cm = confusion_matrix(gold_labels, predictions, labels=list(classes))

    metrics: Dict[str, Any] = {
        "accuracy":         acc,
        "macro_f1":         macro_f1,
        "weighted_f1":      weighted_f1,
        "per_class":        per_class,
        "confusion_matrix": cm.tolist(),
        "num_examples":     int(len(gold_labels)),
    }

    if probabilities is not None:
        probs = np.array(probabilities)
        confidences = np.max(probs, axis=-1)
        correct_mask = (predictions == gold_labels)
        
        correct_conf = confidences[correct_mask]
        incorrect_conf = confidences[~correct_mask]

        metrics["confidence_stats"] = {
            "mean_confidence": float(np.mean(confidences)),
            "min_confidence":  float(np.min(confidences)),
            "max_confidence":  float(np.max(confidences)),
            "correct_confidence_mean": float(np.mean(correct_conf)) if len(correct_conf) > 0 else 0.0,
            "incorrect_confidence_mean": float(np.mean(incorrect_conf)) if len(incorrect_conf) > 0 else 0.0,
        }

    return metrics
