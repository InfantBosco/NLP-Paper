#!/usr/bin/env python
"""
Stage 6 — Run all baselines and save results.

Baselines:
  1. TF-IDF + cosine similarity
  2. Averaged word embeddings (GloVe or random fallback)
  3. Vanilla BERT CLS pooling (no fine-tuning)
  4. Vanilla BERT mean pooling (no fine-tuning)
  5. Official SBERT reference  (external comparison, optional)

Each baseline reports:
  description, configuration, dataset split, embedding size,
  parameter count, Spearman ρ, Pearson r, MSE, MAE,
  inference time, peak memory, random seed, limitations.

Results are saved to:
  <output_dir>/baseline_results.json
  <output_dir>/baseline_results.csv

Usage
-----
  python scripts/run_baselines.py --config configs/baseline_tfidf.yaml
  python scripts/run_baselines.py --config configs/baseline_tfidf.yaml --debug
  python scripts/run_baselines.py --config configs/baseline_tfidf.yaml --baselines tfidf averaged
  python scripts/run_baselines.py --config configs/baseline_tfidf.yaml --include-sbert-ref
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from typing import Any, Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Allow running from project root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbert_reproduction.config import ExperimentConfig
from sbert_reproduction.data.download import (
    download_dataset,
    load_stsb_tsv,
)
from sbert_reproduction.evaluation.similarity_metrics import rescale_scores
from sbert_reproduction.io_utils import save_json
from sbert_reproduction.logging_utils import setup_logger
from sbert_reproduction.seed import set_seed

# ---------------------------------------------------------------------------
AVAILABLE_BASELINES = ["tfidf", "averaged", "bert_cls", "bert_mean", "sbert_ref"]
# ---------------------------------------------------------------------------


def load_stsb_data(data_dir: str, split: str = "test"):
    """Download (if missing) and load STSb records for the given split."""
    stsb_path = download_dataset("stsb", target_dir=data_dir)
    records   = load_stsb_tsv(stsb_path)
    subset    = [r for r in records if r["split"] == split]
    texts_a   = [r["sentence1"] for r in subset]
    texts_b   = [r["sentence2"] for r in subset]
    # Scores are in [0, 5]; normalise to [0, 1] for cosine-similarity comparison
    raw_scores        = np.array([r["score"] for r in subset])
    normalised_scores = rescale_scores(raw_scores, src_min=0.0, src_max=5.0)
    return texts_a, texts_b, normalised_scores, raw_scores, subset


def load_stsb_train_texts(data_dir: str) -> List[str]:
    """Load all training sentences (for TF-IDF vocabulary fitting)."""
    stsb_path = download_dataset("stsb", target_dir=data_dir)
    records   = load_stsb_tsv(stsb_path)
    train     = [r for r in records if r["split"] == "train"]
    return [r["sentence1"] for r in train] + [r["sentence2"] for r in train]


def save_results(results: List[Dict[str, Any]], output_dir: str, logger) -> None:
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "baseline_results.json")
    save_json({"baselines": results}, json_path)
    logger.info(f"Saved JSON results → {json_path}")

    csv_path = os.path.join(output_dir, "baseline_results.csv")
    if results:
        flat_keys = [
            "baseline", "description", "dataset", "split",
            "embedding_size", "parameter_count",
            "spearman_rho", "pearson_r", "mse", "mae",
            "inference_time_sec", "peak_memory_mb", "seed",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=flat_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        logger.info(f"Saved CSV  results → {csv_path}")

    # Print summary table to console
    logger.info("\n{'='*70}")
    logger.info(f"{'Baseline':<25} {'Spearman':>10} {'Pearson':>10} {'MSE':>8} {'MAE':>8}")
    logger.info(f"{'-'*25} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")
    for r in results:
        logger.info(
            f"{r.get('baseline','?'):<25} "
            f"{r.get('spearman_rho',0):>10.2f} "
            f"{r.get('pearson_r',0):>10.2f} "
            f"{r.get('mse',0):>8.4f} "
            f"{r.get('mae',0):>8.4f}"
        )


# ---------------------------------------------------------------------------
def run_tfidf(train_texts, texts_a, texts_b, gold, cfg_raw, seed, logger):
    from sbert_reproduction.baselines import TFIDFBaseline

    max_features = cfg_raw.get("model", {}).get("max_features", 10_000)
    ngram_range  = tuple(cfg_raw.get("model", {}).get("ngram_range", [1, 2]))

    logger.info("[1/5] Running TF-IDF baseline...")
    baseline = TFIDFBaseline(max_features=max_features, ngram_range=ngram_range, seed=seed)
    result   = baseline.run_and_report(train_texts, texts_a, texts_b, gold)
    logger.info(f"  Spearman: {result['spearman_rho']:.2f}  Pearson: {result['pearson_r']:.2f}")
    return result


def run_averaged(all_texts, texts_a, texts_b, gold, cfg_raw, seed, logger):
    from sbert_reproduction.baselines import AveragedWordEmbeddingsBaseline

    dim    = cfg_raw.get("model", {}).get("embedding_dim", 300)
    source = cfg_raw.get("model", {}).get("embedding_source", "glove")

    logger.info("[2/5] Running Averaged Word Embeddings baseline...")
    baseline = AveragedWordEmbeddingsBaseline(dim=dim, seed=seed, source=source)
    result   = baseline.run_and_report(all_texts, texts_a, texts_b, gold)
    logger.info(f"  Spearman: {result['spearman_rho']:.2f}  Pearson: {result['pearson_r']:.2f}")
    return result


def run_bert_cls(texts_a, texts_b, gold, cfg_raw, seed, batch_size, logger):
    from sbert_reproduction.baselines import VanillaBERTCLSBaseline

    model_name = cfg_raw.get("model", {}).get("encoder_name", "bert-base-uncased")

    logger.info("[3/5] Running Vanilla BERT CLS baseline (no fine-tuning)...")
    logger.info("  (Loading BERT model — may take a moment...)")
    baseline = VanillaBERTCLSBaseline(model_name=model_name)
    result   = baseline.run_and_report(texts_a, texts_b, gold, batch_size=batch_size, seed=seed)
    logger.info(f"  Spearman: {result['spearman_rho']:.2f}  Pearson: {result['pearson_r']:.2f}")
    return result


def run_bert_mean(texts_a, texts_b, gold, cfg_raw, seed, batch_size, logger):
    from sbert_reproduction.baselines import VanillaBERTMeanBaseline

    model_name = cfg_raw.get("model", {}).get("encoder_name", "bert-base-uncased")

    logger.info("[4/5] Running Vanilla BERT Mean baseline (no fine-tuning)...")
    baseline = VanillaBERTMeanBaseline(model_name=model_name)
    result   = baseline.run_and_report(texts_a, texts_b, gold, batch_size=batch_size, seed=seed)
    logger.info(f"  Spearman: {result['spearman_rho']:.2f}  Pearson: {result['pearson_r']:.2f}")
    return result


def run_sbert_ref(texts_a, texts_b, gold, cfg_raw, seed, batch_size, logger):
    from sbert_reproduction.baselines import SBERTReferenceBaseline

    model_name = cfg_raw.get("sbert_reference", {}).get(
        "model_name", "sentence-transformers/bert-base-nli-mean-tokens"
    )

    logger.info("[5/5] Running Official SBERT Reference baseline (external comparison)...")
    try:
        baseline = SBERTReferenceBaseline(model_name=model_name, seed=seed)
        result   = baseline.run_and_report(texts_a, texts_b, gold, batch_size=batch_size)
        logger.info(f"  Spearman: {result['spearman_rho']:.2f}  Pearson: {result['pearson_r']:.2f}")
        return result
    except ImportError as e:
        logger.warning(f"  Skipped SBERT reference: {e}")
        return {
            "baseline": "sbert_official_reference",
            "description": "Skipped — sentence-transformers not installed.",
            "spearman_rho": float("nan"),
            "pearson_r":    float("nan"),
            "mse":          float("nan"),
            "mae":          float("nan"),
            "limitations": str(e),
        }


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run SBERT reproduction baselines.")
    parser.add_argument("--config", default="configs/baseline_tfidf.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--debug", action="store_true",
                        help="Use tiny subset (50 pairs) for fast smoke-testing.")
    parser.add_argument(
        "--baselines", nargs="+",
        choices=AVAILABLE_BASELINES,
        default=["tfidf", "averaged"],
        help="Baselines to run. BERT baselines require GPU for reasonable speed.",
    )
    parser.add_argument("--include-sbert-ref", action="store_true",
                        help="Also run the official sentence-transformers reference.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="test",
                        help="Dataset split to evaluate on (train/dev/test).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    cfg    = ExperimentConfig.from_yaml(args.config)
    seed   = cfg.seed
    set_seed(seed)

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "baselines.log")
    logger   = setup_logger("baselines", log_file=log_file)

    logger.info(f"=== Stage 6 — Baseline Experiments ===")
    logger.info(f"Config: {args.config}")
    logger.info(f"Seed:   {seed}")
    logger.info(f"Split:  {args.split}")
    logger.info(f"Debug:  {args.debug}")

    # ------------------------------------------------------------------
    # Load data
    data_dir = args.data_dir or cfg.raw_config.get("data", {}).get("data_dir", "data")
    logger.info("Loading STSb dataset...")
    texts_a, texts_b, gold_norm, gold_raw, records = load_stsb_data(data_dir, split=args.split)
    train_texts = load_stsb_train_texts(data_dir)

    if args.debug:
        n = 50
        texts_a, texts_b, gold_norm, gold_raw = texts_a[:n], texts_b[:n], gold_norm[:n], gold_raw[:n]
        train_texts = train_texts[:500]
        logger.info(f"[DEBUG] Using {n} test pairs and 500 training sentences.")

    logger.info(f"Test pairs:     {len(texts_a)}")
    logger.info(f"Train sentences: {len(train_texts)}")

    all_texts = train_texts + texts_a + texts_b

    # ------------------------------------------------------------------
    # Run selected baselines
    results: List[Dict] = []
    selected = list(args.baselines)
    if args.include_sbert_ref and "sbert_ref" not in selected:
        selected.append("sbert_ref")

    for name in selected:
        try:
            if name == "tfidf":
                r = run_tfidf(train_texts, texts_a, texts_b, gold_norm,
                              cfg.raw_config, seed, logger)
            elif name == "averaged":
                r = run_averaged(all_texts, texts_a, texts_b, gold_norm,
                                 cfg.raw_config, seed, logger)
            elif name == "bert_cls":
                r = run_bert_cls(texts_a, texts_b, gold_norm,
                                 cfg.raw_config, seed, args.batch_size, logger)
            elif name == "bert_mean":
                r = run_bert_mean(texts_a, texts_b, gold_norm,
                                  cfg.raw_config, seed, args.batch_size, logger)
            elif name == "sbert_ref":
                r = run_sbert_ref(texts_a, texts_b, gold_norm,
                                  cfg.raw_config, seed, args.batch_size, logger)
            else:
                logger.warning(f"Unknown baseline: {name}")
                continue

            r["num_examples"]  = len(texts_a)
            r["result_source"] = "INDEPENDENT_REPRODUCTION"
            results.append(r)

        except Exception:
            logger.error(f"Baseline '{name}' FAILED:\n{traceback.format_exc()}")
            results.append({
                "baseline": name,
                "error": traceback.format_exc(),
                "result_source": "INDEPENDENT_REPRODUCTION",
            })

    # ------------------------------------------------------------------
    # Save
    save_results(results, output_dir, logger)
    logger.info(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
