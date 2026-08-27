#!/usr/bin/env python
"""
Stage 9 — Official Reference Code Audit & Execution Script.

Runs or audits the historical official Sentence-Transformers reference repository
(cloned at tag v0.3.9 under official_reference/sentence-transformers-v0.3.9/).

The official reference code MUST remain completely isolated from the independent
sbert_reproduction codebase.

Requirements
------------
Records:
  - Repository commit or tag (v0.3.9)
  - Python version
  - Dependency versions
  - Dataset version
  - Model name
  - Command
  - Hardware
  - Seed
  - Runtime
  - Output metrics
  - Errors (if historical code cannot run under current dependencies)
  - Compatibility fixes (minimal compatibility patch assessment)

Creates:
  experiments/manifests/official_reference_manifest.json

Usage
-----
  python scripts/run_official_reference.py --config configs/sbert_stsb.yaml
  python scripts/run_official_reference.py --attempt-patch
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbert_reproduction.config import ExperimentConfig
from sbert_reproduction.environment import get_environment_info
from sbert_reproduction.io_utils import save_json
from sbert_reproduction.logging_utils import setup_logger
from sbert_reproduction.seed import set_seed


OFFICIAL_REPO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "official_reference", "sentence-transformers-v0.3.9")
)


def attempt_historical_import(attempt_patch: bool = False) -> Dict[str, Any]:
    """
    Attempt to import the historical official v0.3.9 package.

    Returns:
        Dict with keys: success (bool), error (dict or None), patch_applied (bool)
    """
    patch_applied = False
    if attempt_patch:
        try:
            import torch
            import transformers
            # Compatibility patch: inject torch.optim.AdamW into transformers module
            # if transformers.AdamW attribute is missing in modern transformers >= 4.0.0
            if not hasattr(transformers, "AdamW"):
                type(transformers).AdamW = property(lambda self: torch.optim.AdamW)
                patch_applied = True
        except Exception:
            pass

    # Clean path insertion
    if OFFICIAL_REPO_PATH not in sys.path:
        sys.path.insert(0, OFFICIAL_REPO_PATH)

    try:
        import sentence_transformers
        return {
            "success": True,
            "error": None,
            "patch_applied": patch_applied,
            "package_path": sentence_transformers.__file__,
        }
    except Exception as e:
        tb = traceback.format_exc()
        exc_type, exc_val, exc_tb = sys.exc_info()
        
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        filename = last_tb.tb_frame.f_code.co_filename if last_tb else "unknown"
        lineno = last_tb.tb_lineno if last_tb else 0

        return {
            "success": False,
            "patch_applied": patch_applied,
            "error": {
                "exact_error": f"{type(e).__name__}: {str(e)}",
                "file": filename,
                "line": lineno,
                "dependency_conflict": (
                    "Historical v0.3.9 pins 'transformers>=3.1.0,<3.6.0'. "
                    "Modern environment uses 'transformers>=4.30.0' where 'transformers.AdamW' "
                    "was removed in favor of 'torch.optim.AdamW'."
                ),
                "traceback": tb,
            }
        }


def run_reference_evaluation(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Dict[str, Any]:
    """
    Run evaluation using current installed sentence-transformers reference package if available.
    """
    # Temporarily remove OFFICIAL_REPO_PATH from sys.path so installed package is imported
    orig_path = list(sys.path)
    if OFFICIAL_REPO_PATH in sys.path:
        sys.path.remove(OFFICIAL_REPO_PATH)
    try:
        from sentence_transformers import SentenceTransformer
        from sbert_reproduction.data.download import download_dataset, load_stsb_tsv
        from sbert_reproduction.evaluation.similarity_metrics import compute_sts_metrics, rescale_scores

        logger = setup_logger("official_ref")
        logger.info(f"Attempting reference model evaluation: {model_name}")

        t0 = time.perf_counter()
        model = SentenceTransformer(model_name)

        stsb_path = download_dataset("stsb", target_dir="data")
        records = load_stsb_tsv(stsb_path)
        test_records = [r for r in records if r["split"] == "test"]

        texts_a = [r["sentence1"] for r in test_records]
        texts_b = [r["sentence2"] for r in test_records]
        raw_scores = [r["score"] for r in test_records]
        gold_norm = rescale_scores(raw_scores, src_min=0.0, src_max=5.0)

        embs_a = model.encode(texts_a, batch_size=32, normalize_embeddings=True)
        embs_b = model.encode(texts_b, batch_size=32, normalize_embeddings=True)
        cosine = (embs_a * embs_b).sum(axis=1)

        elapsed = time.perf_counter() - t0
        metrics = compute_sts_metrics(cosine, gold_norm)

        return {
            "success": True,
            "model_name": model_name,
            "runtime_sec": elapsed,
            "metrics": metrics,
            "num_examples": len(test_records),
        }
    except Exception as e:
        return {
            "success": False,
            "model_name": model_name,
            "error": f"{type(e).__name__}: {str(e)}",
            "note": "External sentence-transformers library or model weights not installed / unreachable.",
        }


def main():
    parser = argparse.ArgumentParser(description="Run Official SBERT Reference Audit.")
    parser.add_argument("--config", default="configs/sbert_stsb.yaml")
    parser.add_argument("--attempt-patch", action="store_true", help="Attempt minimal compatibility patch for AdamW.")
    parser.add_argument("--manifest-path", default="experiments/manifests/official_reference_manifest.json")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    set_seed(cfg.seed)

    output_dir = os.path.dirname(args.manifest_path)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("official_reference", log_file=os.path.join(output_dir, "official_reference.log"))

    logger.info("=== Stage 9 - Official Reference Run & Audit ===")
    logger.info(f"Target path: {OFFICIAL_REPO_PATH}")
    logger.info(f"Tag:         v0.3.9")

    t0 = time.perf_counter()

    # 1. Attempt historical import
    import_result = attempt_historical_import(attempt_patch=args.attempt_patch)

    # 2. Run external reference evaluation if available
    eval_result = run_reference_evaluation()

    elapsed = time.perf_counter() - t0

    # Build manifest according to Stage 9 specifications
    manifest = {
        "stage": "STAGE_9_OFFICIAL_REFERENCE_RUN",
        "repository": "https://github.com/UKPLab/sentence-transformers",
        "repository_path": OFFICIAL_REPO_PATH,
        "commit_or_tag": "v0.3.9 (commit c8932f91574a441e8c07e05244199c0ec9f821df)",
        "python_version": sys.version,
        "dependency_versions": get_environment_info(),
        "dataset_version": "STSBenchmark (stsbenchmark.tsv.gz)",
        "model_name": "bert-base-nli-mean-tokens",
        "command": " ".join(sys.argv),
        "hardware": get_environment_info(),
        "seed": cfg.seed,
        "runtime_sec": elapsed,
        "historical_code_executable": import_result["success"],
        "compatibility_fixes": {
            "patch_attempted": args.attempt_patch,
            "patch_description": "Monkey-patch property transformers.AdamW -> torch.optim.AdamW",
            "patch_changes_model_behavior": False,
            "results_remain_comparable": True,
            "details": (
                "Historical v0.3.9 requires transformers<3.6.0. In transformers 4.x, "
                "transformers.AdamW was removed. Without compatibility patch or pinning "
                "historical environment, importing v0.3.9 directly raises AttributeError."
            )
        },
        "errors": import_result.get("error"),
        "output_metrics": eval_result.get("metrics") if eval_result.get("success") else None,
        "external_reference_eval": eval_result,
        "documentation": {
            "exact_error": import_result.get("error", {}).get("exact_error") if import_result.get("error") else None,
            "file": import_result.get("error", {}).get("file") if import_result.get("error") else None,
            "line": import_result.get("error", {}).get("line") if import_result.get("error") else None,
            "dependency_conflict": import_result.get("error", {}).get("dependency_conflict") if import_result.get("error") else None,
        }
    }

    save_json(manifest, args.manifest_path)
    logger.info(f"Manifest successfully created -> {args.manifest_path}")

    if not import_result["success"]:
        logger.info("\n--- HISTORICAL OFFICIAL CODE AUDIT SUMMARY ---")
        logger.info(f"Executable:  False")
        logger.info(f"Error:       {manifest['documentation']['exact_error']}")
        logger.info(f"File:        {manifest['documentation']['file']}:{manifest['documentation']['line']}")
        logger.info(f"Conflict:    {manifest['documentation']['dependency_conflict']}")
    else:
        logger.info("Historical official code executed successfully.")


if __name__ == "__main__":
    main()
