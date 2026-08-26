from typing import List, Dict, Any
from collections import Counter

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
        label = row["label"].strip().lower()
        if label not in valid_labels:
            raise ValueError(f"Row {idx} invalid NLI label: {label}")
    return True

def validate_stsb_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates STSb dataset splits, scores, and checks for duplicates."""
    validate_stsb_data(records)
    split_counts = Counter()
    duplicate_count = 0
    seen_pairs = set()

    for r in records:
        split = r["split"]
        s1 = r["sentence1"].strip()
        s2 = r["sentence2"].strip()
        split_counts[split] += 1
        pair_key = tuple(sorted([s1, s2]))
        if pair_key in seen_pairs:
            duplicate_count += 1
        else:
            seen_pairs.add(pair_key)

    return {
        "total_records": len(records),
        "split_counts": dict(split_counts),
        "duplicates_found": duplicate_count,
        "valid": True
    }

def validate_nli_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validates AllNLI dataset splits, label distributions, and checks for duplicates."""
    validate_nli_data(records)
    split_counts = Counter()
    label_counts = Counter()
    duplicate_count = 0
    seen_pairs = set()

    for r in records:
        split = r["split"]
        label = r["label"].strip().lower()
        s1 = r["sentence1"].strip()
        s2 = r["sentence2"].strip()
        split_counts[split] += 1
        label_counts[label] += 1
        pair_key = (s1, s2)
        if pair_key in seen_pairs:
            duplicate_count += 1
        else:
            seen_pairs.add(pair_key)

    return {
        "total_records": len(records),
        "split_counts": dict(split_counts),
        "label_counts": dict(label_counts),
        "duplicates_found": duplicate_count,
        "valid": True
    }
