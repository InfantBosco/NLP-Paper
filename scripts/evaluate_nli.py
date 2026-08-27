#!/usr/bin/env python
"""
Stage 8 — NLI Classification Evaluation Script.

Evaluates an SBERT classification checkpoint on AllNLI / SNLI / MultiNLI.
Computes:
  - Accuracy
  - Macro-F1 & Weighted-F1
  - Per-class Precision, Recall, and F1 score
  - Confusion matrix
  - Confidence statistics

Output includes full provenance:
  - Dataset & Split
  - Number of examples
  - Model & Checkpoint
  - Configuration & Seed
  - Software versions
  - Metric implementation
  - Result source classification (INDEPENDENT_REPRODUCTION)

Usage
-----
  python scripts/evaluate_nli.py --checkpoint experiments/checkpoints/sbert_nli_base/best_checkpoint.pt
  python scripts/evaluate_nli.py --config configs/sbert_nli.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbert_reproduction.config import ExperimentConfig
from sbert_reproduction.data.download import download_dataset, load_nli_tsv
from sbert_reproduction.environment import get_environment_info
from sbert_reproduction.evaluation.classification_metrics import compute_classification_metrics
from sbert_reproduction.io_utils import save_json
from sbert_reproduction.logging_utils import setup_logger
from sbert_reproduction.models.encoder import TransformerEncoderWrapper, TokenizerWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import SBERTModel, ClassificationHead
from sbert_reproduction.losses import SoftmaxLoss
from sbert_reproduction.training.data_loading import NLI_LABEL_MAP, NLI_LABEL_NAMES
from sbert_reproduction.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate SBERT on NLI Classification.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file or directory.")
    parser.add_argument("--config", type=str, default="configs/sbert_nli.yaml", help="Path to config YAML.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Dataset split.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing datasets.")
    parser.add_argument("--output-dir", type=str, default="experiments/results/evaluate_nli", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("evaluate_nli", log_file=os.path.join(args.output_dir, "evaluate.log"))

    logger.info("=== Stage 8 - NLI Classification Evaluation ===")
    logger.info(f"Config:     {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split:      {args.split}")

    cfg = ExperimentConfig.from_yaml(args.config) if os.path.exists(args.config) else ExperimentConfig()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device:     {device}")

    # Load Model
    encoder_name = cfg.raw_config.get("model", {}).get("encoder_name", "bert-base-uncased")
    pooling_mode = cfg.raw_config.get("model", {}).get("pooling_mode", "mean")
    max_seq_len  = cfg.raw_config.get("data", {}).get("max_seq_length", 128)
    num_labels   = int(cfg.raw_config.get("model", {}).get("num_labels", 3))
    concat_mode  = cfg.raw_config.get("model", {}).get("concatenation_mode", "u_v_absdiff")

    encoder = TransformerEncoderWrapper(encoder_name, device=device)
    sent_encoder = SentenceEncoder(encoder, pooling_mode=pooling_mode)
    sbert_model = SBERTModel(sent_encoder)

    loss_fn = SoftmaxLoss(
        sentence_embedding_dimension=encoder.hidden_size,
        num_labels=num_labels,
        concatenation_mode=concat_mode,
    )

    if args.checkpoint:
        if os.path.isdir(args.checkpoint):
            ckpt_path = os.path.join(args.checkpoint, "best_checkpoint.pt")
            if not os.path.isfile(ckpt_path):
                ckpt_path = os.path.join(args.checkpoint, "checkpoint_epoch_0.pt")
        else:
            ckpt_path = args.checkpoint

        if os.path.isfile(ckpt_path):
            logger.info(f"Loading checkpoint weights from {ckpt_path}...")
            state = torch.load(ckpt_path, map_location=device)
            state_dict = state["model_state"] if "model_state" in state else state
            sbert_model.load_state_dict(state_dict, strict=False)
            
            # Check if loss_fn / classifier weights are in state_dict
            loss_state = {k.replace("loss_fn.", ""): v for k, v in state_dict.items() if k.startswith("loss_fn.")}
            if loss_state:
                loss_fn.load_state_dict(loss_state)
        else:
            logger.warning(f"Checkpoint file {ckpt_path} not found. Evaluating un-finetuned model.")

    sbert_model.to(device)
    loss_fn.to(device)
    sbert_model.eval()
    loss_fn.eval()

    # Load Data
    nli_path = download_dataset("allnli", target_dir=args.data_dir)
    records = load_nli_tsv(nli_path)
    split_records = [r for r in records if r["split"] == args.split]

    texts_a = [r["sentence1"] for r in split_records]
    texts_b = [r["sentence2"] for r in split_records]
    gold_labels = np.array([NLI_LABEL_MAP.get(r["label"], 2) for r in split_records])

    logger.info(f"Evaluating {len(split_records)} NLI sentence pairs from split: {args.split}")

    tokenizer_wrapper = TokenizerWrapper(encoder_name, max_seq_length=max_seq_len, device=device)

    # Encode & Evaluate
    t0 = time.perf_counter()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts_a), args.batch_size):
            batch_a = texts_a[i : i + args.batch_size]
            batch_b = texts_b[i : i + args.batch_size]

            fa = tokenizer_wrapper.tokenize(batch_a)
            fb = tokenizer_wrapper.tokenize(batch_b)

            u, v = sbert_model(fa, fb)
            logits = loss_fn(u, v)  # [batch, num_labels]
            all_logits.append(logits.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_probs = F.softmax(all_logits, dim=-1).numpy()
    predictions = np.argmax(all_probs, axis=-1)
    elapsed = time.perf_counter() - t0

    # Compute Classification Metrics
    metrics = compute_classification_metrics(
        predictions=predictions,
        gold_labels=gold_labels,
        probabilities=all_probs,
        label_names=NLI_LABEL_NAMES,
    )

    # Build Provenance Record
    record = {
        "experiment_name": cfg.experiment_name,
        "model_description": f"SBERT NLI ({encoder_name}, pooling={pooling_mode}, concat={concat_mode})",
        "result_source": "INDEPENDENT_REPRODUCTION",
        "dataset": "AllNLI",
        "split": args.split,
        "num_examples": len(split_records),
        "embedding_size": int(encoder.hidden_size),
        "parameter_count": sum(p.numel() for p in sbert_model.parameters()) + sum(p.numel() for p in loss_fn.parameters()),
        "checkpoint": args.checkpoint,
        "configuration": cfg.raw_config,
        "seed": cfg.seed,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "confidence_stats": metrics.get("confidence_stats", {}),
        "inference_time_sec": elapsed,
        "sentences_per_sec": len(texts_a) * 2 / max(elapsed, 1e-6),
        "software_versions": get_environment_info(),
        "metric_implementation": "sbert_reproduction.evaluation.classification_metrics.compute_classification_metrics",
    }

    logger.info("-" * 50)
    logger.info(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    logger.info(f"Macro-F1:    {metrics['macro_f1'] * 100:.2f}%")
    logger.info(f"Weighted-F1: {metrics['weighted_f1'] * 100:.2f}%")
    logger.info(f"Inference:   {elapsed:.2f}s")
    logger.info("-" * 50)

    out_json = os.path.join(args.output_dir, f"nli_{args.split}_metrics.json")
    save_json(record, out_json)
    logger.info(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
