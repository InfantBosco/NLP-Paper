import numpy as np
from sbert_reproduction.evaluation.similarity_metrics import compute_sts_metrics

def test_compute_sts_metrics_perfect_correlation():
    predictions = np.array([0.1, 0.4, 0.7, 0.9])
    gold_labels = np.array([0.1, 0.4, 0.7, 0.9])
    metrics = compute_sts_metrics(predictions, gold_labels)

    assert abs(metrics["spearman_rho"] - 100.0) < 1e-4
    assert abs(metrics["pearson_r"] - 100.0) < 1e-4
    assert abs(metrics["mse"] - 0.0) < 1e-6
