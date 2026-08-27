"""
Training data loading utilities for SBERT reproduction.

Provides Dataset classes and loader functions for:
- NLI (AllNLI) — classification objective
- STS-B — regression objective
- Triplet — triplet loss objective
- Debug mode subset

Paper reference: Section 4 — NLI training data (SNLI + MultiNLI = AllNLI).
Official code reference: sentence-transformers/examples/training_nli/training_nli.py

Label mapping (NLI):
  contradiction → 0
  entailment    → 1
  neutral       → 2
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

NLI_LABEL_MAP: Dict[str, int] = {
    "contradiction": 0,
    "entailment":    1,
    "neutral":       2,
}

NLI_LABEL_NAMES: List[str] = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
# NLI Dataset
# ---------------------------------------------------------------------------

class NLIDataset(Dataset):
    """
    Dataset for NLI (classification) training.

    Each sample provides:
        sentence1, sentence2, label (int 0/1/2)

    Args:
        records:        List of dicts with keys sentence1, sentence2, label.
        label_map:      String-to-int label mapping.
        max_examples:   Optional cap for debug mode.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        label_map: Dict[str, int] = NLI_LABEL_MAP,
        max_examples: Optional[int] = None,
    ) -> None:
        self.records   = records[:max_examples] if max_examples else records
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.records[idx]
        raw_label = r.get("label", r.get("label_id", "neutral"))
        if isinstance(raw_label, str):
            label = self.label_map.get(raw_label, 2)
        else:
            label = int(raw_label)
        return {
            "sentence1": r["sentence1"],
            "sentence2": r["sentence2"],
            "label":     label,
        }


# ---------------------------------------------------------------------------
# STS-B Dataset
# ---------------------------------------------------------------------------

class STSBDataset(Dataset):
    """
    Dataset for STS-B (regression) training and evaluation.

    Scores are normalised to [0, 1] from the original [0, 5] range.

    Args:
        records:      List of dicts with keys sentence1, sentence2, score.
        max_examples: Optional cap for debug mode.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        max_examples: Optional[int] = None,
    ) -> None:
        self.records = records[:max_examples] if max_examples else records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r     = self.records[idx]
        score = float(r["score"])
        # Normalise from [0, 5] to [0, 1]
        norm_score = score / 5.0
        return {
            "sentence1": r["sentence1"],
            "sentence2": r["sentence2"],
            "label":     norm_score,
        }


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class SBERTCollator:
    """
    Collates a list of samples into model-ready batch tensors.

    Tokenises sentence1 and sentence2 separately so they can be fed
    through the Siamese encoder.

    Args:
        tokenizer:      HuggingFace tokenizer (from AutoTokenizer).
        max_seq_length: Maximum token length.
        objective:      "classification" | "regression" | "triplet"
    """

    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 128,
        objective: str = "classification",
    ) -> None:
        self.tokenizer      = tokenizer
        self.max_seq_length = max_seq_length
        self.objective      = objective

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts_a  = [item["sentence1"] for item in batch]
        texts_b  = [item["sentence2"] for item in batch]

        enc_a = self.tokenizer(
            texts_a,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        enc_b = self.tokenizer(
            texts_b,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        result = {"features_a": enc_a, "features_b": enc_b}

        if "label" in batch[0]:
            labels = [item["label"] for item in batch]
            if self.objective == "classification":
                result["labels"] = torch.tensor(labels, dtype=torch.long)
            else:
                result["labels"] = torch.tensor(labels, dtype=torch.float)

        return result


# ---------------------------------------------------------------------------
# Loader factory
# ---------------------------------------------------------------------------

def make_nli_dataloader(
    records: List[Dict[str, Any]],
    tokenizer,
    batch_size: int = 16,
    max_seq_length: int = 128,
    shuffle: bool = True,
    max_examples: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for NLI records."""
    dataset  = NLIDataset(records, max_examples=max_examples)
    collator = SBERTCollator(tokenizer, max_seq_length, objective="classification")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )


def make_stsb_dataloader(
    records: List[Dict[str, Any]],
    tokenizer,
    batch_size: int = 16,
    max_seq_length: int = 128,
    shuffle: bool = False,
    max_examples: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader for STS-B records."""
    dataset  = STSBDataset(records, max_examples=max_examples)
    collator = SBERTCollator(tokenizer, max_seq_length, objective="regression")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )


def load_debug_dataset(debug_path: str = "data/debug_dataset.json") -> Dict[str, List]:
    """Load the tiny debug dataset created by prepare_data.py."""
    with open(debug_path, "r", encoding="utf-8") as fh:
        return json.load(fh)
