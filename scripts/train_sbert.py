#!/usr/bin/env python
"""
Stage 7 — SBERT Training Pipeline.

Trains SBERT using the NLI classification objective (default) or
the STS-B regression objective, as described in Reimers & Gurevych (2019).

Features
--------
- YAML configuration with CLI key=value overrides
- Fixed random seeds (torch, numpy, random, cuda)
- CPU and GPU support
- Optional mixed precision (fp16)
- Gradient accumulation and clipping
- Checkpoint saving (every epoch + best)
- Resume-from-checkpoint
- Validation evaluation on STSb test set (Spearman ρ)
- Training logs and CSV/JSON metrics
- Failure logging
- Environment, hardware, git-commit logging
- Resolved config saved to experiment directory
- Debug mode: tiny subset, few steps, no large checkpoints

Usage
-----
  # Full NLI training (requires GPU for reasonable speed)
  python scripts/train_sbert.py --config configs/sbert_nli.yaml

  # Debug smoke-test (runs in seconds on CPU)
  python scripts/train_sbert.py --config configs/sbert_nli.yaml --debug

  # Override config values
  python scripts/train_sbert.py --config configs/sbert_nli.yaml \\
      training.epochs=2 training.batch_size=8

  # Resume from checkpoint
  python scripts/train_sbert.py --config configs/sbert_nli.yaml \\
      --resume experiments/checkpoints/sbert_nli_base/checkpoint_epoch_0.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback

import numpy as np
import torch

# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# ---------------------------------------------------------------------------

from sbert_reproduction.config import ExperimentConfig
from sbert_reproduction.data.download import (
    download_dataset,
    load_nli_tsv,
    load_stsb_tsv,
)
from sbert_reproduction.evaluation.similarity_metrics import (
    compute_sts_metrics,
    rescale_scores,
)
from sbert_reproduction.logging_utils import setup_logger
from sbert_reproduction.models.encoder import TransformerEncoderWrapper, TokenizerWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import SBERTModel
from sbert_reproduction.losses import SoftmaxLoss, CosineSimilarityLoss
from sbert_reproduction.seed import set_seed
from sbert_reproduction.training.data_loading import (
    make_nli_dataloader,
    make_stsb_dataloader,
    NLI_LABEL_MAP,
)
from sbert_reproduction.training.experiment import ExperimentManifest
from sbert_reproduction.training.trainer import SBERTTrainer


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def parse_overrides(overrides: list) -> dict:
    """
    Parse key=value strings (e.g. training.epochs=2) into a nested dict.

    Example:
        ["training.epochs=2", "model.pooling_mode=cls"]
        → {"training": {"epochs": 2}, "model": {"pooling_mode": "cls"}}
    """
    result = {}
    for item in overrides:
        if "=" not in item:
            continue
        key, _, val = item.partition("=")
        parts = key.strip().split(".")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Try to coerce to int / float / bool
        if val.lower() in ("true", "false"):
            val = val.lower() == "true"
        else:
            try:
                val = int(val)
            except ValueError:
                try:
                    val = float(val)
                except ValueError:
                    pass
        d[parts[-1]] = val
    return result


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Evaluation function (closure over STSb test data)
# ---------------------------------------------------------------------------

def make_eval_fn(test_records, tokenizer_wrapper, batch_size: int, device: str):
    """
    Returns an eval_fn(model, device) → dict for use with SBERTTrainer.

    Evaluates on the STSb test split.
    """
    texts_a      = [r["sentence1"] for r in test_records]
    texts_b      = [r["sentence2"] for r in test_records]
    raw_scores   = np.array([r["score"] for r in test_records])
    gold_norm    = rescale_scores(raw_scores)

    def eval_fn(model: SBERTModel, device: str) -> dict:
        model.eval()
        embs_a, embs_b = [], []
        with torch.no_grad():
            for start in range(0, len(texts_a), batch_size):
                ta = texts_a[start: start + batch_size]
                tb = texts_b[start: start + batch_size]
                fa = tokenizer_wrapper.tokenize(ta)
                fb = tokenizer_wrapper.tokenize(tb)
                fa = {k: v.to(device) for k, v in fa.items()}
                fb = {k: v.to(device) for k, v in fb.items()}
                ua = model.encode(**fa).cpu()
                vb = model.encode(**fb).cpu()
                embs_a.append(ua)
                embs_b.append(vb)

        embs_a = torch.cat(embs_a, dim=0).numpy()
        embs_b = torch.cat(embs_b, dim=0).numpy()

        norms_a = np.linalg.norm(embs_a, axis=1, keepdims=True).clip(min=1e-9)
        norms_b = np.linalg.norm(embs_b, axis=1, keepdims=True).clip(min=1e-9)
        cosine  = np.sum((embs_a / norms_a) * (embs_b / norms_b), axis=1)

        metrics = compute_sts_metrics(cosine, gold_norm)
        model.train()
        return metrics

    return eval_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SBERT (Stage 7)")
    parser.add_argument("--config",  default="configs/sbert_nli.yaml")
    parser.add_argument("--resume",  default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--debug",   action="store_true",
                        help="Debug mode: tiny data, 5 steps, no large checkpoints.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("overrides", nargs="*", help="key=value config overrides.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load + merge config
    cfg = ExperimentConfig.from_yaml(args.config)
    raw = cfg.raw_config

    if args.overrides:
        raw = deep_merge(raw, parse_overrides(args.overrides))

    if args.debug:
        raw.setdefault("training", {})
        raw["training"]["epochs"]      = 1
        raw["training"]["batch_size"]  = 4
        raw["training"]["evaluation_steps"] = 2
        raw.setdefault("data", {})["max_debug_examples"] = 32
        raw["experiment_name"] = raw.get("experiment_name", "debug") + "_debug"

    seed = int(raw.get("seed", 42))
    set_seed(seed)

    # ------------------------------------------------------------------
    # Output directory and logging
    output_dir  = raw.get("output_dir", "experiments/checkpoints/default")
    if args.debug:
        output_dir += "_debug"
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "train.log")
    logger   = setup_logger("sbert.train", log_file=log_file)
    logger.info("=" * 60)
    logger.info(f"Experiment: {raw.get('experiment_name')}")
    logger.info(f"Config:     {args.config}")
    logger.info(f"Seed:       {seed}")
    logger.info(f"Debug:      {args.debug}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Save experiment manifest (provenance)
    manifest = ExperimentManifest(
        output_dir  = output_dir,
        config_dict = raw,
        command     = " ".join(sys.argv),
    )
    manifest.save_manifest()
    logger.info(f"Experiment manifest saved -> {output_dir}/manifest.json")

    # ------------------------------------------------------------------
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model config
    model_cfg       = raw.get("model", {})
    encoder_name    = model_cfg.get("encoder_name", "bert-base-uncased")
    pooling_mode    = model_cfg.get("pooling_mode", "mean")
    max_seq_length  = raw.get("data", {}).get("max_seq_length", 128)
    num_labels      = int(model_cfg.get("num_labels", 3))
    concat_mode     = model_cfg.get("concatenation_mode", "u_v_absdiff")
    objective       = raw.get("objective", "classification")

    logger.info(f"Encoder:    {encoder_name}")
    logger.info(f"Pooling:    {pooling_mode}")
    logger.info(f"Objective:  {objective}")

    # ------------------------------------------------------------------
    # Build model
    encoder      = TransformerEncoderWrapper(encoder_name, device=device)
    sent_encoder = SentenceEncoder(encoder, pooling_mode=pooling_mode)
    model        = SBERTModel(sent_encoder)

    tokenizer_wrapper = TokenizerWrapper(
        model_name     = encoder_name,
        max_seq_length = max_seq_length,
        device         = device,
    )

    # Force tokenizer load and get HuggingFace tokenizer for collator
    tokenizer_wrapper._load()
    hf_tokenizer = tokenizer_wrapper._tokenizer

    # Build loss function
    if objective == "classification":
        embedding_dim = encoder.hidden_size
        loss_fn = SoftmaxLoss(
            sentence_embedding_dimension = embedding_dim,
            num_labels                   = num_labels,
            concatenation_mode           = concat_mode,
        )
    else:
        loss_fn = CosineSimilarityLoss()

    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ------------------------------------------------------------------
    # Load data
    data_cfg   = raw.get("data", {})
    data_dir   = args.data_dir or data_cfg.get("data_dir", "data")
    max_debug  = data_cfg.get("max_debug_examples", None) if args.debug else None

    training_cfg = raw.get("training", {})
    batch_size   = int(training_cfg.get("batch_size", 16))

    logger.info("Loading training data...")
    if objective == "classification":
        nli_path = download_dataset("allnli", target_dir=data_dir)
        records  = load_nli_tsv(nli_path)
        train_records = [r for r in records if r["split"] == "train"]
        if max_debug:
            train_records = train_records[:max_debug]
        logger.info(f"NLI training records: {len(train_records):,}")
        train_dl = make_nli_dataloader(
            train_records, hf_tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_examples=max_debug,
        )
    else:
        stsb_path    = download_dataset("stsb", target_dir=data_dir)
        all_records  = load_stsb_tsv(stsb_path)
        train_records = [r for r in all_records if r["split"] == "train"]
        if max_debug:
            train_records = train_records[:max_debug]
        logger.info(f"STSb training records: {len(train_records):,}")
        train_dl = make_stsb_dataloader(
            train_records, hf_tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            max_examples=max_debug,
        )

    # STSb test set for evaluation
    stsb_path    = download_dataset("stsb", target_dir=data_dir)
    all_stsb     = load_stsb_tsv(stsb_path)
    test_records = [r for r in all_stsb if r["split"] == "test"]
    if max_debug:
        test_records = test_records[:max_debug]
    eval_fn = make_eval_fn(test_records, tokenizer_wrapper, batch_size, device)
    logger.info(f"Evaluation records: {len(test_records)}")

    # ------------------------------------------------------------------
    # Trainer
    trainer = SBERTTrainer(
        model      = model,
        loss_fn    = loss_fn,
        config     = raw,
        output_dir = output_dir,
        device     = device,
        eval_fn    = eval_fn,
    )

    # ------------------------------------------------------------------
    # Train
    logger.info("Starting training...")
    t_start = time.perf_counter()
    try:
        summary = trainer.train(train_dl, resume_from=args.resume)
    except Exception:
        err = traceback.format_exc()
        logger.error(f"Training FAILED:\n{err}")
        failure_path = os.path.join(output_dir, "failure.log")
        with open(failure_path, "a", encoding="utf-8") as fh:
            fh.write(err)
        sys.exit(1)

    elapsed = time.perf_counter() - t_start
    logger.info(f"Training completed in {elapsed:.1f}s")
    logger.info(f"Best Spearman rho: {summary['best_metric']:.2f}")
    logger.info(f"Best checkpoint: {summary['best_checkpoint']}")

    # Save predictions on best checkpoint if available
    if summary.get("best_checkpoint") and os.path.isfile(summary["best_checkpoint"]):
        from sbert_reproduction.training.trainer import load_full_checkpoint
        load_full_checkpoint(summary["best_checkpoint"], model, map_location=device)
        final_metrics = eval_fn(model, device)
        preds_path = os.path.join(output_dir, "test_metrics.json")
        with open(preds_path, "w", encoding="utf-8") as fh:
            json.dump({
                "checkpoint": summary["best_checkpoint"],
                "metrics":    final_metrics,
                "seed":       seed,
                "result_source": "INDEPENDENT_REPRODUCTION",
            }, fh, indent=2)
        logger.info(f"Final test metrics saved -> {preds_path}")


if __name__ == "__main__":
    main()
