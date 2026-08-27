"""
Stage 8 evaluation metrics unit tests.

Covers:
- compute_sts_metrics (Spearman, Pearson, MSE, MAE)
- rescale_scores
- compute_classification_metrics (Accuracy, Macro-F1, Weighted-F1, per-class P/R/F1, confusion matrix, confidence stats)
- EvaluationRecord validation and serialization
"""

import pytest
import numpy as np

from sbert_reproduction.evaluation.similarity_metrics import (
    compute_sts_metrics,
    rescale_scores,
    EvaluationRecord,
)
from sbert_reproduction.evaluation.classification_metrics import (
    compute_classification_metrics,
)
from sbert_reproduction.evaluation.reporting import (
    validate_result_record,
    generate_summary_table,
)


class TestSTSMetrics:

    def test_perfect_correlation(self):
        preds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        golds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        res = compute_sts_metrics(preds, golds)
        assert abs(res["spearman_rho"] - 100.0) < 1e-4
        assert abs(res["pearson_r"] - 100.0) < 1e-4
        assert abs(res["mse"] - 0.0) < 1e-6
        assert abs(res["mae"] - 0.0) < 1e-6

    def test_negative_correlation(self):
        preds = np.array([1.0, 0.75, 0.5, 0.25, 0.0])
        golds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        res = compute_sts_metrics(preds, golds)
        assert abs(res["spearman_rho"] - (-100.0)) < 1e-4
        assert abs(res["pearson_r"] - (-100.0)) < 1e-4

    def test_rescale_scores(self):
        raw = np.array([0.0, 2.5, 5.0])
        scaled = rescale_scores(raw, src_min=0.0, src_max=5.0, dst_min=0.0, dst_max=1.0)
        assert np.allclose(scaled, np.array([0.0, 0.5, 1.0]))


class TestClassificationMetrics:

    def test_perfect_classification(self):
        preds = np.array([0, 1, 2, 0, 1, 2])
        golds = np.array([0, 1, 2, 0, 1, 2])
        probs = np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
        ])
        metrics = compute_classification_metrics(preds, golds, probs, ["entailment", "neutral", "contradiction"])

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["weighted_f1"] == 1.0
        assert len(metrics["per_class"]) == 3
        assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        assert "confidence_stats" in metrics
        assert abs(metrics["confidence_stats"]["mean_confidence"] - 0.9) < 1e-5
        assert abs(metrics["confidence_stats"]["correct_confidence_mean"] - 0.9) < 1e-5

    def test_imperfect_classification(self):
        preds = np.array([0, 0, 1, 1])
        golds = np.array([0, 1, 1, 1])
        probs = np.array([
            [0.8, 0.2],
            [0.7, 0.3],  # error (predicted 0, gold 1)
            [0.1, 0.9],
            [0.2, 0.8],
        ])
        metrics = compute_classification_metrics(preds, golds, probs, ["c0", "c1"])

        assert metrics["accuracy"] == 0.75
        assert metrics["confusion_matrix"] == [[1, 0], [1, 2]]
        assert metrics["confidence_stats"]["incorrect_confidence_mean"] == 0.7


class TestEvaluationRecordValidation:

    def test_record_serialization(self):
        rec = EvaluationRecord(
            experiment_name="test_exp",
            model_description="Test Model",
            result_source="INDEPENDENT_REPRODUCTION",
            spearman_rho=80.5,
            pearson_r=81.2,
        )
        d = rec.to_dict()
        validate_result_record(d)
        assert d["result_source"] == "INDEPENDENT_REPRODUCTION"
        assert d["spearman_rho"] == 80.5

    def test_invalid_source_raises(self):
        rec = EvaluationRecord(
            experiment_name="test_exp",
            model_description="Test Model",
            result_source="INVALID_SOURCE",
        )
        d = rec.to_dict()
        with pytest.raises(ValueError, match="Invalid result_source"):
            validate_result_record(d)
