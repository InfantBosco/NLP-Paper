import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def compute_classification_metrics(predictions: np.ndarray, gold_labels: np.ndarray) -> dict:
    acc = accuracy_score(gold_labels, predictions)
    macro_f1 = f1_score(gold_labels, predictions, average="macro")
    weighted_f1 = f1_score(gold_labels, predictions, average="weighted")
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1)
    }
