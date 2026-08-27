"""
Reporting utilities for SBERT evaluation results.

Implements summary table formatting, markdown export, and result classification formatting:
- PAPER
- OFFICIAL_CODE
- INDEPENDENT_REPRODUCTION
- EXTENSION
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict, Any


VALID_RESULT_SOURCES = {
    "PAPER",
    "OFFICIAL_CODE",
    "INDEPENDENT_REPRODUCTION",
    "EXTENSION"
}


def validate_result_record(record: Dict[str, Any]) -> None:
    """Validate that an evaluation result record contains all mandatory provenance fields."""
    required_fields = [
        "dataset", "split", "num_examples", "model",
        "checkpoint", "configuration", "seed",
        "software_versions", "metric_implementation", "result_source"
    ]
    missing = [f for f in required_fields if f not in record]
    if missing:
        raise ValueError(f"Evaluation record is missing required fields: {missing}")

    source = record["result_source"]
    if source not in VALID_RESULT_SOURCES:
        raise ValueError(f"Invalid result_source '{source}'. Must be one of {VALID_RESULT_SOURCES}")


def generate_summary_table(results_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates comparison summary dataframe for metrics reporting.
    Validates provenance fields for every record.
    """
    for rec in results_list:
        validate_result_record(rec)
    return pd.DataFrame(results_list)


def format_markdown_table(results_list: List[Dict[str, Any]]) -> str:
    """
    Format evaluation results list into GitHub Flavored Markdown table.
    """
    df = generate_summary_table(results_list)
    cols = [c for c in [
        "model", "dataset", "split", "spearman_rho", "pearson_r",
        "accuracy", "macro_f1", "result_source"
    ] if c in df.columns]
    
    return df[cols].to_markdown(index=False)
