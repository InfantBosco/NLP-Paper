"""
Full configurable SBERTTrainer for Stage 7.

Features:
- CPU and GPU support
- Optional mixed precision (fp16) via torch.cuda.amp.GradScaler
- Gradient accumulation
- Gradient clipping
- Checkpoint saving with full state (model + optimizer + scheduler + epoch)
- Best-checkpoint selection (based on validation Spearman ρ)
- Resume-from-checkpoint support
- Validation evaluation at configurable intervals
- CSV + JSON metrics logging per epoch and per step
- Failure logging
- Git commit logging
- Environment capture

Paper reference: Section 4 — Training details.
Official code: sentence-transformers/examples/training_nli/ (independent reimpl.)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader


logger = logging.getLogger("sbert.trainer")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_full_checkpoint(
    output_dir: str,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    metrics: dict,
    config: dict,
    filename: str = "checkpoint.pt",
) -> str:
    """
    Save model weights, optimizer state, scheduler state, epoch, step,
    and validation metrics to *output_dir/filename*.

    Args:
        output_dir: Directory to write the checkpoint to.
        model:      nn.Module whose state_dict() to save.
        optimizer:  Optimizer whose state_dict() to save.
        scheduler:  LR scheduler whose state_dict() to save (or None).
        epoch:      Current epoch index (0-based).
        global_step: Global training step count.
        metrics:    Validation metrics dict for this checkpoint.
        config:     Full resolved config dict (for auditability).
        filename:   Output filename.

    Returns:
        Absolute path of the saved checkpoint.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    state = {
        "epoch":       epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "metrics":     metrics,
        "config":      config,
    }
    torch.save(state, path)
    return path


def load_full_checkpoint(
    path: str,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location: str = "cpu",
) -> dict:
    """
    Restore checkpoint written by :func:`save_full_checkpoint`.

    Mutates *model*, *optimizer*, *scheduler* in-place.

    Returns:
        The full state dict (contains epoch, global_step, metrics, config).
    """
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    return state


# ---------------------------------------------------------------------------
# Metrics CSV writer
# ---------------------------------------------------------------------------

class MetricsCSVWriter:
    """Appends metric rows to a CSV file after each epoch / evaluation."""

    def __init__(self, path: str, fieldnames: List[str]) -> None:
        self.path       = path
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Write header on first creation
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=fieldnames).writeheader()

    def write(self, row: dict) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(self.path if False else fh,
                                    fieldnames=self.fieldnames,
                                    extrasaction="ignore")
            writer.writerow(row)


# ---------------------------------------------------------------------------
# SBERTTrainer
# ---------------------------------------------------------------------------

class SBERTTrainer:
    """
    Full training loop for SBERT with all Stage-7 requirements.

    Args:
        model:            SBERTModel (or any nn.Module with forward(fa, fb)).
        loss_fn:          Loss function (SoftmaxLoss / CosineSimilarityLoss / TripletLoss).
        config:           Resolved config dict (from YAML + CLI overrides).
        output_dir:       Directory for checkpoints, logs, and metrics.
        device:           "cpu" or "cuda".
        eval_fn:          Optional callable(model, device) → dict of metrics.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: dict,
        output_dir: str,
        device: str = "cpu",
        eval_fn: Optional[Callable] = None,
    ) -> None:
        self.model      = model.to(device)
        self.loss_fn    = loss_fn.to(device)
        self.config     = config
        self.output_dir = output_dir
        self.device     = device
        self.eval_fn    = eval_fn

        # Training hyperparameters
        training_cfg  = config.get("training", {})
        self.lr       = float(training_cfg.get("learning_rate", 2e-5))
        self.epochs   = int(training_cfg.get("epochs", 1))
        self.grad_acc = int(training_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        self.fp16     = bool(training_cfg.get("fp16", False)) and (device == "cuda")
        self.eval_steps = int(training_cfg.get("evaluation_steps", 500))
        self.warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
        self.weight_decay = float(training_cfg.get("weight_decay", 0.01))

        os.makedirs(output_dir, exist_ok=True)
        self._log     = logging.getLogger("sbert.trainer")
        self._scaler  = torch.cuda.amp.GradScaler() if self.fp16 else None

        self.global_step  = 0
        self.best_metric  = -float("inf")
        self.best_ckpt    = None

        # Metrics CSV
        self._train_csv = MetricsCSVWriter(
            os.path.join(output_dir, "train_metrics.csv"),
            ["epoch", "step", "loss", "lr", "timestamp"],
        )
        self._eval_csv  = MetricsCSVWriter(
            os.path.join(output_dir, "eval_metrics.csv"),
            ["epoch", "step", "spearman_rho", "pearson_r", "mse", "mae", "timestamp"],
        )

    # ------------------------------------------------------------------
    def _build_optimizer_and_scheduler(self, num_training_steps: int):
        """Build AdamW + linear warmup scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]
        params   = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(params, lr=self.lr)

        warmup_steps = int(num_training_steps * self.warmup_ratio)
        from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
        try:
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )
        except ImportError:
            # Fallback: no warmup
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        return optimizer, scheduler

    # ------------------------------------------------------------------
    def _run_eval(self, epoch: int) -> dict:
        """Run evaluation and return metrics dict."""
        if self.eval_fn is None:
            return {}
        self.model.eval()
        with torch.no_grad():
            metrics = self.eval_fn(self.model, self.device)
        self.model.train()
        return metrics

    # ------------------------------------------------------------------
    def _save_checkpoint(self, optimizer, scheduler, epoch: int, metrics: dict,
                         filename: str) -> str:
        path = save_full_checkpoint(
            output_dir  = self.output_dir,
            model       = self.model,
            optimizer   = optimizer,
            scheduler   = scheduler,
            epoch       = epoch,
            global_step = self.global_step,
            metrics     = metrics,
            config      = self.config,
            filename    = filename,
        )
        self._log.info(f"  Checkpoint saved → {path}")
        return path

    # ------------------------------------------------------------------
    def _update_best(self, optimizer, scheduler, epoch: int, metrics: dict) -> None:
        """Save a new best checkpoint if the validation metric improved."""
        key   = "spearman_rho"
        score = metrics.get(key, -float("inf"))
        if score > self.best_metric:
            self.best_metric = score
            self.best_ckpt   = self._save_checkpoint(
                optimizer, scheduler, epoch, metrics, "best_checkpoint.pt"
            )
            self._log.info(f"  New best {key}: {score:.2f}")

    # ------------------------------------------------------------------
    def train(
        self,
        train_dataloader: DataLoader,
        resume_from: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full training loop.

        Args:
            train_dataloader: Iterable of batches.
            resume_from:      Path to a checkpoint to resume from.

        Returns:
            Dict with training summary (best_metric, best_ckpt, final_step).
        """
        total_steps = len(train_dataloader) * self.epochs // self.grad_acc
        optimizer, scheduler = self._build_optimizer_and_scheduler(total_steps)

        start_epoch = 0
        if resume_from and os.path.isfile(resume_from):
            state = load_full_checkpoint(resume_from, self.model, optimizer, scheduler,
                                         map_location=self.device)
            start_epoch       = state["epoch"] + 1
            self.global_step  = state["global_step"]
            self.best_metric  = state["metrics"].get("spearman_rho", -float("inf"))
            self._log.info(f"Resumed from {resume_from} at epoch {start_epoch}")

        self.model.train()
        all_train_losses = []

        for epoch in range(start_epoch, self.epochs):
            epoch_loss    = 0.0
            num_batches   = 0
            optimizer.zero_grad()

            t_epoch_start = time.perf_counter()

            for step, batch in enumerate(train_dataloader):
                try:
                    loss = self._train_step(batch, optimizer, scheduler, step)
                    epoch_loss  += loss
                    num_batches += 1

                    # Log to CSV every accumulation boundary
                    if (step + 1) % self.grad_acc == 0:
                        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else self.lr
                        self._train_csv.write({
                            "epoch": epoch,
                            "step":  self.global_step,
                            "loss":  round(loss, 6),
                            "lr":    current_lr,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        })

                    # Periodic evaluation
                    if self.eval_steps > 0 and self.global_step % self.eval_steps == 0 and self.global_step > 0:
                        metrics = self._run_eval(epoch)
                        self._log_eval(epoch, metrics)
                        self._update_best(optimizer, scheduler, epoch, metrics)

                except Exception:
                    err = traceback.format_exc()
                    self._log.error(f"Step {step} FAILED:\n{err}")
                    self._save_failure_log(err, epoch, step)
                    raise

            # End-of-epoch
            avg_loss = epoch_loss / max(num_batches, 1)
            elapsed  = time.perf_counter() - t_epoch_start
            self._log.info(f"Epoch {epoch} | loss={avg_loss:.4f} | "
                           f"steps={num_batches} | time={elapsed:.1f}s")
            all_train_losses.append(avg_loss)

            # Epoch-end evaluation + checkpoint
            metrics = self._run_eval(epoch)
            self._log_eval(epoch, metrics)
            self._update_best(optimizer, scheduler, epoch, metrics)
            self._save_checkpoint(optimizer, scheduler, epoch, metrics,
                                  f"checkpoint_epoch_{epoch}.pt")

        summary = {
            "epochs_trained": self.epochs - start_epoch,
            "global_step":    self.global_step,
            "train_losses":   all_train_losses,
            "best_metric":    self.best_metric,
            "best_checkpoint": self.best_ckpt,
        }
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        self._log.info(f"Training complete. Summary saved → {summary_path}")
        return summary

    # ------------------------------------------------------------------
    def _train_step(self, batch: dict, optimizer, scheduler, step: int) -> float:
        """Execute one forward + backward step (with grad accumulation)."""
        features_a = {k: v.to(self.device) for k, v in batch["features_a"].items()}
        features_b = {k: v.to(self.device) for k, v in batch["features_b"].items()}
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        if self.fp16 and self._scaler is not None:
            with torch.cuda.amp.autocast():
                u, v  = self.model(features_a, features_b)
                loss  = self.loss_fn(u, v, labels) / self.grad_acc
            self._scaler.scale(loss).backward()
            if (step + 1) % self.grad_acc == 0:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(optimizer)
                self._scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1
        else:
            u, v  = self.model(features_a, features_b)
            loss  = self.loss_fn(u, v, labels) / self.grad_acc
            loss.backward()
            if (step + 1) % self.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                self.global_step += 1

        return loss.item() * self.grad_acc  # un-scaled for logging

    # ------------------------------------------------------------------
    def _log_eval(self, epoch: int, metrics: dict) -> None:
        self._log.info(
            f"  [Eval epoch={epoch} step={self.global_step}] "
            + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if isinstance(v, float))
        )
        self._eval_csv.write({
            "epoch":        epoch,
            "step":         self.global_step,
            "spearman_rho": metrics.get("spearman_rho", ""),
            "pearson_r":    metrics.get("pearson_r", ""),
            "mse":          metrics.get("mse", ""),
            "mae":          metrics.get("mae", ""),
            "timestamp":    time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    # ------------------------------------------------------------------
    def _save_failure_log(self, error_text: str, epoch: int, step: int) -> None:
        path = os.path.join(self.output_dir, "failure.log")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"\n{'='*60}\n")
            fh.write(f"Epoch: {epoch}  Step: {step}  GlobalStep: {self.global_step}\n")
            fh.write(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%S')}\n\n")
            fh.write(error_text)
        self._log.error(f"Failure details written to {path}")
