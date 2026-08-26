import numpy as np
from scipy.stats import spearmanr, pearsonr

def compute_sts_metrics(predictions: np.ndarray, gold_labels: np.ndarray) -> dict:
    """Computes Spearman rank correlation (rho * 100), Pearson r * 100, and MSE."""
    predictions = np.array(predictions)
    gold_labels = np.array(gold_labels)

    spearman_rho, _ = spearmanr(predictions, gold_labels)
    pearson_r, _ = pearsonr(predictions, gold_labels)
    mse = float(np.mean((predictions - gold_labels) ** 2))

    return {
        "spearman_rho": float(spearman_rho * 100.0) if not np.isnan(spearman_rho) else 0.0,
        "pearson_r": float(pearson_r * 100.0) if not np.isnan(pearson_r) else 0.0,
        "mse": mse
    }
