NLI_LABEL_MAP = {"contradiction": 0, "entailment": 1, "neutral": 2}

def preprocess_nli_label(label_str: str) -> int:
    return NLI_LABEL_MAP[label_str.strip().lower()]

def normalize_stsb_score(score: float) -> float:
    """Normalizes STSb score from [0, 5] to [0.0, 1.0]."""
    return float(score) / 5.0
