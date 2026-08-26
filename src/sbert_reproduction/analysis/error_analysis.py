import numpy as np
import pandas as pd
from typing import List, Dict, Any

def extract_top_errors(samples: List[Dict[str, Any]], predictions: np.ndarray, gold_labels: np.ndarray, top_k: int = 20) -> pd.DataFrame:
    """Extracts top-k largest prediction error examples for STS/NLI."""
    errors = np.abs(predictions - gold_labels)
    df = pd.DataFrame(samples)
    df["prediction"] = predictions
    df["gold_label"] = gold_labels
    df["absolute_error"] = errors
    return df.sort_values(by="absolute_error", ascending=False).head(top_k)
