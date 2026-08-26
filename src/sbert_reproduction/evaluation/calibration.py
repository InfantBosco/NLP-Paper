import numpy as np

def compute_error_bounds(predictions: np.ndarray, targets: np.ndarray) -> dict:
    errors = np.abs(predictions - targets)
    return {
        "mean_absolute_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "std_error": float(np.std(errors))
    }
