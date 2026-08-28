#!/usr/bin/env python
"""
Stage 8 — STS Benchmark Evaluation Script.

Evaluates an SBERT checkpoint or baseline model on STS Benchmark (dev or test split).
Computes Spearman rank correlation (rho * 100), Pearson correlation (r * 100), MSE, and MAE.

Output includes full provenance:
  - Dataset name & split
  - Number of examples
  - Model name & checkpoint path
  - Configuration & Seed
  - Software versions
  - Metric implementation
  - Result source classification (INDEPENDENT_REPRODUCTION)

Usage
-----
  python scripts/evaluate_stsb.py --checkpoint experiments/checkpoints/sbert_nli_base/best_checkpoint.pt
  python scripts/evaluate_stsb.py --config configs/sbert_stsb.yaml --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sbert_reproduction.config import ExperimentConfig
from sbert_reproduction.data.download import download_dataset, load_stsb_tsv
from sbert_reproduction.environment import get_environment_info
from sbert_reproduction.evaluation.similarity_metrics import (
    compute_sts_metrics,
    rescale_scores,
    EvaluationRecord,
)
from sbert_reproduction.io_utils import save_json
from sbert_reproduction.logging_utils import setup_logger
from sbert_reproduction.models.encoder import TransformerEncoderWrapper, TokenizerWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import SBERTModel
from sbert_reproduction.training.checkpointing import load_checkpoint
from sbert_reproduction.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description="Evaluate SBERT on STS Benchmark.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file or directory.")
    parser.add_argument("--config", type=str, default="configs/sbert_stsb.yaml", help="Path to config YAML.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "dev", "test"], help="Dataset split.")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing datasets.")
    parser.add_argument("--output-dir", type=str, default="experiments/results/evaluate_stsb", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger("evaluate_stsb", log_file=os.path.join(args.output_dir, "evaluate.log"))

    logger.info("=== Stage 8 - STS Benchmark Evaluation ===")
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

    encoder = TransformerEncoderWrapper(encoder_name, device=device)
    encoder.load_pretrained()
    sent_encoder = SentenceEncoder(encoder, pooling_mode=pooling_mode)
    model = SBERTModel(sent_encoder)

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
            if "model_state" in state:
                model.load_state_dict(state["model_state"])
            else:
                model.load_state_dict(state)
        else:
            logger.warning(f"Checkpoint file {ckpt_path} not found. Evaluating un-finetuned model.")

    model.to(device)
    model.eval()

    # Load Data
    stsb_path = download_dataset("stsb", target_dir=args.data_dir)
    records = load_stsb_tsv(stsb_path)
    split_records = [r for r in records if r["split"] == args.split]

    texts_a = [r["sentence1"] for r in split_records]
    texts_b = [r["sentence2"] for r in split_records]
    raw_scores = np.array([r["score"] for r in split_records])
    gold_norm = rescale_scores(raw_scores, src_min=0.0, src_max=5.0)

    logger.info(f"Evaluating {len(split_records)} sentence pairs from STSb split: {args.split}")

    tokenizer_wrapper = TokenizerWrapper(encoder_name, max_seq_length=max_seq_len, device=device)

    # Encode & Predict
    t0 = time.perf_counter()
    embs_a, embs_b = [], []
    with torch.no_grad():
        for i in range(0, len(texts_a), args.batch_size):
            batch_a = texts_a[i : i + args.batch_size]
            batch_b = texts_b[i : i + args.batch_size]

            fa = tokenizer_wrapper.tokenize(batch_a)
            fb = tokenizer_wrapper.tokenize(batch_b)

            ua = model.encode(**fa).cpu()
            vb = model.encode(**fb).cpu()

            embs_a.append(ua)
            embs_b.append(vb)

    embs_a = torch.cat(embs_a, dim=0).numpy()
    embs_b = torch.cat(embs_b, dim=0).numpy()
    elapsed = time.perf_counter() - t0

    # Compute Cosine Similarity
    norms_a = np.linalg.norm(embs_a, axis=1, keepdims=True).clip(min=1e-9)
    norms_b = np.linalg.norm(embs_b, axis=1, keepdims=True).clip(min=1e-9)
    cosine = np.sum((embs_a / norms_a) * (embs_b / norms_b), axis=1)

    # Compute Metrics
    metrics = compute_sts_metrics(cosine, gold_norm)

    # Record Provenance
    record = EvaluationRecord(
        experiment_name=cfg.experiment_name,
        model_description=f"SBERT ({encoder_name}, pooling={pooling_mode})",
        result_source="INDEPENDENT_REPRODUCTION",
        dataset="STSBenchmark",
        split=args.split,
        num_examples=len(split_records),
        embedding_size=int(embs_a.shape[1]),
        parameter_count=sum(p.numel() for p in model.parameters()),
        checkpoint=args.checkpoint,
        configuration=cfg.raw_config,
        seed=cfg.seed,
        spearman_rho=metrics["spearman_rho"],
        pearson_r=metrics["pearson_r"],
        mse=metrics["mse"],
        mae=metrics["mae"],
        inference_time_sec=elapsed,
        sentences_per_sec=len(texts_a) * 2 / max(elapsed, 1e-6),
        software_versions=get_environment_info(),
        metric_implementation="sbert_reproduction.evaluation.similarity_metrics.compute_sts_metrics",
    )

    record_dict = record.to_dict()

    logger.info("-" * 50)
    logger.info(f"Spearman rho: {metrics['spearman_rho']:.2f}")
    logger.info(f"Pearson r:    {metrics['pearson_r']:.2f}")
    logger.info(f"MSE:          {metrics['mse']:.4f}")
    logger.info(f"MAE:          {metrics['mae']:.4f}")
    logger.info(f"Inference:    {elapsed:.2f}s ({record.sentences_per_sec:.1f} sents/sec)")
    logger.info("-" * 50)

    out_json = os.path.join(args.output_dir, f"stsb_{args.split}_metrics.json")
    save_json(record_dict, out_json)
    logger.info(f"Results saved to {out_json}")


if __name__ == "__main__":
    main()
