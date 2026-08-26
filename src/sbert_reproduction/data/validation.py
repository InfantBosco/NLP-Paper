from typing import Dict, Any, List

def validate_stsb_data(rows: List[Dict[str, Any]]) -> bool:
    """Validates required columns and target range [0.0, 5.0] for STSb dataset."""
    required_cols = {"split", "score", "sentence1", "sentence2"}
    for idx, row in enumerate(rows):
        if not required_cols.issubset(row.keys()):
            raise ValueError(f"Row {idx} missing required columns: {required_cols - set(row.keys())}")
        score = float(row["score"])
        if score < 0.0 or score > 5.0:
            raise ValueError(f"Row {idx} score {score} out of range [0, 5]")
    return True

def validate_nli_data(rows: List[Dict[str, Any]]) -> bool:
    """Validates required columns and labels for AllNLI dataset."""
    valid_labels = {"contradiction", "entailment", "neutral"}
    required_cols = {"split", "label", "sentence1", "sentence2"}
    for idx, row in enumerate(rows):
        if not required_cols.issubset(row.keys()):
            raise ValueError(f"Row {idx} missing required columns: {required_cols - set(row.keys())}")
        label = row["label"]
        if label not in valid_labels:
            raise ValueError(f"Row {idx} invalid NLI label: {label}")
    return True
