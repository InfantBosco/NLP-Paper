#!/usr/bin/env python
"""
STAGE 12: ABLATION STUDIES
Comprehensive ablation studies for Sentence-BERT reproduction.

Required ablations:
1. CLS pooling vs Mean pooling vs Max pooling
2. Different encoder sizes (bert-base, bert-large, roberta-base, roberta-large)
3. Different learning rates or batch sizes

Optional ablations (implemented if practical):
- With vs without normalization
- Different maximum sequence lengths
- Frozen vs fine-tuned encoder
- Different NLI training subsets
- Different loss functions (will use classification on NLI)
- Multiple random seeds
- Different pooling implementations

For each ablation:
- State hypothesis
- Change one major factor
- Keep evaluation constant
- Save CSV results
- Generate figure
- Explain result
- Record limitations
"""

import argparse
import json
import os
import time
import random
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sbert_reproduction.models.encoder import TransformerEncoderWrapper, TokenizerWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import SBERTModel, ClassificationHead
from sbert_reproduction.losses.classification import SoftmaxLoss


def get_device(device: str = "auto") -> str:
    """Get the appropriate device."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device in ["cuda", "cpu"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("[!] CUDA not available, falling back to CPU")
            return "cpu"
        return device
    else:
        raise ValueError(f"Invalid device: {device}")


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    description: str
    encoder_name: str
    pooling_mode: str
    normalize: bool
    max_seq_length: int
    batch_size: int
    learning_rate: float
    epochs: int
    freeze_encoder: bool
    random_seed: int


class NLIDataLoader:
    """Helper to load NLI data for training."""

    def __init__(
        self,
        data_path: str = "data/AllNLI.tsv.gz",
        max_samples: Optional[int] = None,
        train_fraction: float = 0.01,  # Use 1% for faster ablations
    ):
        self.data_path = data_path
        self.max_samples = max_samples
        self.train_fraction = train_fraction
        self.data = None

    def load(self) -> Tuple[List[str], List[str], List[int]]:
        """Load NLI training data."""
        import gzip
        
        sentences1, sentences2, labels = [], [], []
        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}

        with gzip.open(self.data_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # Skip header

                parts = line.strip().split("\t")
                if len(parts) < 4:
                    continue

                split, label, sent1, sent2 = parts[0], parts[1], parts[2], parts[3]
                
                if split != "train":
                    continue

                if label not in label_map:
                    continue

                sentences1.append(sent1)
                sentences2.append(sent2)
                labels.append(label_map[label])

                if len(sentences1) >= int(len(sentences1) * self.train_fraction) if self.train_fraction < 1 else False:
                    break

                if self.max_samples and len(sentences1) >= self.max_samples:
                    break

        # Apply train fraction if not exceeded
        if self.train_fraction < 1.0 and len(sentences1) > 1000:
            indices = np.random.RandomState(42).choice(
                len(sentences1), 
                size=int(len(sentences1) * self.train_fraction), 
                replace=False
            )
            sentences1 = [sentences1[i] for i in indices]
            sentences2 = [sentences2[i] for i in indices]
            labels = [labels[i] for i in indices]

        return sentences1, sentences2, labels


class AblationStudy:
    """Main ablation study runner."""

    def __init__(
        self,
        output_dir: str = "experiments/results/ablations_stage12",
        device: str = "auto",
        quick_mode: bool = True,  # Use 1% of NLI data for speed
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = get_device(device)
        self.quick_mode = quick_mode
        self.results = []

    def set_seed(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_small_nli_batch(
        self,
        config: AblationConfig,
        num_batches: int = 5,
    ) -> Dict[str, Any]:
        """
        Train on a small batch of NLI data and record metrics.
        This is for quick ablation studies.
        """
        self.set_seed(config.random_seed)

        # Load tokenizer and encoder
        tokenizer = TokenizerWrapper(
            model_name=config.encoder_name,
            max_seq_length=config.max_seq_length,
            device=self.device,
        )

        # Load encoder
        transformer = TransformerEncoderWrapper(
            config.encoder_name,
            device=self.device,
        )

        # Create sentence encoder
        sentence_encoder = SentenceEncoder(
            transformer,
            pooling_mode=config.pooling_mode,
            normalize=config.normalize,
        )

        # Freeze encoder if needed
        if config.freeze_encoder:
            for param in sentence_encoder.encoder.parameters():
                param.requires_grad = False

        # Create SBERT model
        model = SBERTModel(sentence_encoder)
        model = model.to(self.device)

        # Add classification head
        hidden_dim = transformer.hidden_size
        classification_head = ClassificationHead(
            embedding_dim=hidden_dim,
            num_labels=3,
            concatenation_mode="u_v_absdiff",
        )
        classification_head = classification_head.to(self.device)

        # Create optimizer (only for unfrozen params)
        trainable_params = list(model.parameters()) + list(classification_head.parameters())
        if config.freeze_encoder:
            trainable_params = [p for p in trainable_params if p.requires_grad]

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            weight_decay=0.01,
        )

        # Loss function
        loss_fct = nn.CrossEntropyLoss()

        # Load small NLI batch
        loader = NLIDataLoader(train_fraction=0.01 if self.quick_mode else 0.1)
        try:
            sent1, sent2, labels = loader.load()
        except FileNotFoundError:
            print(f"[!] NLI data not found, using synthetic data")
            sent1 = [f"Sentence {i}" for i in range(100)]
            sent2 = [f"Another {i}" for i in range(100)]
            labels = [i % 3 for i in range(100)]

        # Limit to num_batches for speed
        batch_size = config.batch_size
        max_samples = batch_size * num_batches
        if len(sent1) > max_samples:
            indices = np.random.RandomState(42).choice(
                len(sent1), size=max_samples, replace=False
            )
            sent1 = [sent1[i] for i in indices]
            sent2 = [sent2[i] for i in indices]
            labels = [labels[i] for i in indices]

        # Training loop
        model.train()
        classification_head.train()
        total_loss = 0.0
        num_steps = 0
        times = []

        for epoch in range(config.epochs):
            for start_idx in range(0, len(sent1), batch_size):
                end_idx = min(start_idx + batch_size, len(sent1))
                batch_sent1 = sent1[start_idx:end_idx]
                batch_sent2 = sent2[start_idx:end_idx]
                batch_labels = torch.tensor(
                    labels[start_idx:end_idx],
                    dtype=torch.long,
                    device=self.device,
                )

                step_start = time.perf_counter()

                # Tokenize
                features_1 = tokenizer.tokenize(batch_sent1)
                features_2 = tokenizer.tokenize(batch_sent2)

                # Encode
                with torch.no_grad() if config.freeze_encoder else torch.enable_grad():
                    embeddings_1 = model.sentence_encoder(
                        features_1["input_ids"],
                        features_1["attention_mask"],
                        features_1.get("token_type_ids"),
                    )
                    embeddings_2 = model.sentence_encoder(
                        features_2["input_ids"],
                        features_2["attention_mask"],
                        features_2.get("token_type_ids"),
                    )

                # Classify
                logits = classification_head(embeddings_1, embeddings_2)
                loss = loss_fct(logits, batch_labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                step_end = time.perf_counter()
                times.append(step_end - step_start)
                total_loss += loss.item()
                num_steps += 1

        avg_loss = total_loss / num_steps if num_steps > 0 else 0
        avg_time = np.mean(times)

        return {
            "final_loss": avg_loss,
            "avg_step_time": avg_time,
            "num_steps": num_steps,
            "num_samples_trained": len(sent1),
        }

    def run_ablation_1_pooling(self) -> List[Dict[str, Any]]:
        """
        ABLATION 1: Pooling Strategy
        Hypothesis: Different pooling methods affect embedding quality differently.
        Mean pooling should work best (as per paper).
        """
        print("\n" + "="*80)
        print("ABLATION 1: POOLING STRATEGIES")
        print("="*80)

        pooling_modes = ["mean", "max", "cls"]
        results = []

        for pooling_mode in pooling_modes:
            print(f"\n[*] Testing pooling mode: {pooling_mode}")

            config = AblationConfig(
                name=f"Pooling-{pooling_mode.upper()}",
                description=f"SBERT with {pooling_mode} pooling",
                encoder_name="bert-base-uncased",
                pooling_mode=pooling_mode,
                normalize=False,
                max_seq_length=128,
                batch_size=16,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=False,
                random_seed=42,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Pooling",
                "parameter": pooling_mode,
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")
            print(f"  Avg step time: {metrics['avg_step_time']:.4f}s")

        return results

    def run_ablation_2_encoder_size(self) -> List[Dict[str, Any]]:
        """
        ABLATION 2: Encoder Size
        Hypothesis: Larger encoders produce better embeddings but slower training.
        Test: bert-base-uncased, bert-large-uncased, roberta-base
        """
        print("\n" + "="*80)
        print("ABLATION 2: ENCODER SIZES")
        print("="*80)

        encoders = [
            "bert-base-uncased",
            "bert-large-uncased",
            "roberta-base",
        ]
        results = []

        for encoder_name in encoders:
            print(f"\n[*] Testing encoder: {encoder_name}")

            config = AblationConfig(
                name=f"Encoder-{encoder_name.replace('-', '_').upper()}",
                description=f"SBERT with {encoder_name}",
                encoder_name=encoder_name,
                pooling_mode="mean",
                normalize=False,
                max_seq_length=128,
                batch_size=16,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=False,
                random_seed=42,
            )

            try:
                metrics = self.train_small_nli_batch(config, num_batches=3)

                result = {
                    "ablation": "Encoder",
                    "parameter": encoder_name,
                    "final_loss": metrics["final_loss"],
                    "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                    "num_steps": metrics["num_steps"],
                    **asdict(config),
                }
                results.append(result)
                print(f"  Final loss: {metrics['final_loss']:.4f}")
                print(f"  Avg step time: {metrics['avg_step_time']:.4f}s")
            except Exception as e:
                print(f"  [!] Error: {e}")

        return results

    def run_ablation_3_hyperparameters(self) -> List[Dict[str, Any]]:
        """
        ABLATION 3: Hyperparameters (Learning Rate & Batch Size)
        Hypothesis: Different hyperparameters affect convergence speed and final loss.
        Test: LR [1e-5, 2e-5, 5e-5], Batch size [8, 16, 32]
        """
        print("\n" + "="*80)
        print("ABLATION 3: HYPERPARAMETERS (LR & Batch Size)")
        print("="*80)

        learning_rates = [1e-5, 2e-5, 5e-5]
        batch_sizes = [8, 16, 32]
        results = []

        for lr in learning_rates:
            print(f"\n[*] Testing learning rate: {lr}")
            
            config = AblationConfig(
                name=f"LR-{lr:.0e}",
                description=f"SBERT with LR={lr}",
                encoder_name="bert-base-uncased",
                pooling_mode="mean",
                normalize=False,
                max_seq_length=128,
                batch_size=16,
                learning_rate=lr,
                epochs=1,
                freeze_encoder=False,
                random_seed=42,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Learning Rate",
                "parameter": f"LR={lr}",
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")

        for bs in batch_sizes:
            print(f"\n[*] Testing batch size: {bs}")

            config = AblationConfig(
                name=f"BatchSize-{bs}",
                description=f"SBERT with batch_size={bs}",
                encoder_name="bert-base-uncased",
                pooling_mode="mean",
                normalize=False,
                max_seq_length=128,
                batch_size=bs,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=False,
                random_seed=42,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Batch Size",
                "parameter": f"BS={bs}",
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")

        return results

    def run_ablation_4_normalization(self) -> List[Dict[str, Any]]:
        """
        ABLATION 4: Normalization
        Hypothesis: L2 normalization improves cosine similarity quality.
        Test: With vs Without normalization
        """
        print("\n" + "="*80)
        print("ABLATION 4: NORMALIZATION")
        print("="*80)

        results = []

        for normalize in [False, True]:
            print(f"\n[*] Testing normalization: {normalize}")

            config = AblationConfig(
                name=f"Normalize-{normalize}",
                description=f"SBERT with normalize={normalize}",
                encoder_name="bert-base-uncased",
                pooling_mode="mean",
                normalize=normalize,
                max_seq_length=128,
                batch_size=16,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=False,
                random_seed=42,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Normalization",
                "parameter": f"normalize={normalize}",
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")

        return results

    def run_ablation_5_max_seq_length(self) -> List[Dict[str, Any]]:
        """
        ABLATION 5: Maximum Sequence Length
        Hypothesis: Longer sequences provide more context but slower training.
        Test: [64, 128, 256]
        """
        print("\n" + "="*80)
        print("ABLATION 5: MAXIMUM SEQUENCE LENGTH")
        print("="*80)

        max_lengths = [64, 128, 256]
        results = []

        for max_len in max_lengths:
            print(f"\n[*] Testing max_seq_length: {max_len}")

            config = AblationConfig(
                name=f"MaxLen-{max_len}",
                description=f"SBERT with max_seq_length={max_len}",
                encoder_name="bert-base-uncased",
                pooling_mode="mean",
                normalize=False,
                max_seq_length=max_len,
                batch_size=16,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=False,
                random_seed=42,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Max Seq Length",
                "parameter": f"max_len={max_len}",
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")

        return results

    def run_ablation_6_frozen_encoder(self) -> List[Dict[str, Any]]:
        """
        ABLATION 6: Frozen vs Fine-tuned Encoder
        Hypothesis: Frozen BERT acts as fixed feature extractor, fine-tuned adapts to task.
        Fine-tuned should produce better results but is slower.
        """
        print("\n" + "="*80)
        print("ABLATION 6: FROZEN VS FINE-TUNED ENCODER")
        print("="*80)

        results = []

        for freeze in [False, True]:
            status = "FROZEN" if freeze else "FINE-TUNED"
            print(f"\n[*] Testing encoder: {status}")

            config = AblationConfig(
                name=f"Encoder-{status}",
                description=f"SBERT with frozen_encoder={freeze}",
                encoder_name="bert-base-uncased",
                pooling_mode="mean",
                normalize=False,
                max_seq_length=128,
                batch_size=16,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=freeze,
                random_seed=42,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Encoder State",
                "parameter": status,
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")

        return results

    def run_ablation_7_random_seeds(self) -> List[Dict[str, Any]]:
        """
        ABLATION 7: Multiple Random Seeds
        Hypothesis: Different random seeds affect training stability.
        Test: [42, 123, 256, 512, 1024]
        """
        print("\n" + "="*80)
        print("ABLATION 7: MULTIPLE RANDOM SEEDS")
        print("="*80)

        seeds = [42, 123, 256, 512, 1024]
        results = []

        for seed in seeds:
            print(f"\n[*] Testing random seed: {seed}")

            config = AblationConfig(
                name=f"Seed-{seed}",
                description=f"SBERT with random_seed={seed}",
                encoder_name="bert-base-uncased",
                pooling_mode="mean",
                normalize=False,
                max_seq_length=128,
                batch_size=16,
                learning_rate=2e-5,
                epochs=1,
                freeze_encoder=False,
                random_seed=seed,
            )

            metrics = self.train_small_nli_batch(config, num_batches=5)

            result = {
                "ablation": "Random Seed",
                "parameter": f"seed={seed}",
                "final_loss": metrics["final_loss"],
                "avg_step_time_ms": metrics["avg_step_time"] * 1000,
                "num_steps": metrics["num_steps"],
                **asdict(config),
            }
            results.append(result)
            print(f"  Final loss: {metrics['final_loss']:.4f}")

        return results

    def run_all_ablations(self) -> pd.DataFrame:
        """Run all ablation studies."""
        print("\n" + "="*80)
        print("STAGE 12: ABLATION STUDIES - STARTING")
        print("="*80)

        all_results = []

        # Run ablations
        all_results.extend(self.run_ablation_1_pooling())
        all_results.extend(self.run_ablation_2_encoder_size())
        all_results.extend(self.run_ablation_3_hyperparameters())
        all_results.extend(self.run_ablation_4_normalization())
        all_results.extend(self.run_ablation_5_max_seq_length())
        all_results.extend(self.run_ablation_6_frozen_encoder())
        all_results.extend(self.run_ablation_7_random_seeds())

        # Save results
        self._save_results(all_results)

        # Create plots
        self._create_plots(all_results)

        return pd.DataFrame(all_results)

    def _save_results(self, results: List[Dict[str, Any]]):
        """Save results to CSV and JSON."""
        # Save CSV
        df = pd.DataFrame(results)
        csv_path = self.output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[OK] Saved CSV: {csv_path}")

        # Save JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[OK] Saved JSON: {json_path}")

        # Summary statistics
        summary_path = self.output_dir / "ablation_summary.txt"
        with open(summary_path, "w") as f:
            f.write("ABLATION STUDY SUMMARY\n")
            f.write("="*80 + "\n\n")

            for ablation in df["ablation"].unique():
                f.write(f"\n{ablation.upper()}\n")
                f.write("-"*80 + "\n")
                subset = df[df["ablation"] == ablation]
                for _, row in subset.iterrows():
                    f.write(f"  {row['parameter']:30} | Loss: {row['final_loss']:8.4f} | Time: {row['avg_step_time_ms']:8.2f}ms\n")

        print(f"[OK] Saved summary: {summary_path}")

    def _create_plots(self, results: List[Dict[str, Any]]):
        """Create visualization plots for each ablation."""
        df = pd.DataFrame(results)

        # Plot 1: Pooling
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pooling_df = df[df["ablation"] == "Pooling"]
        axes[0].bar(pooling_df["parameter"], pooling_df["final_loss"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[0].set_title("Pooling Strategy - Final Loss", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("Final Loss", fontsize=12)
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(pooling_df["parameter"], pooling_df["avg_step_time_ms"], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1].set_title("Pooling Strategy - Step Time", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Avg Step Time (ms)", fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.output_dir / "ablation_1_pooling.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved plot: {plot_path}")

        # Plot 2: Encoder sizes
        encoder_df = df[df["ablation"] == "Encoder"]
        if len(encoder_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].barh(encoder_df["parameter"], encoder_df["final_loss"], color='#d62728')
            axes[0].set_title("Encoder Size - Final Loss", fontsize=14, fontweight='bold')
            axes[0].set_xlabel("Final Loss", fontsize=12)
            axes[0].grid(True, alpha=0.3, axis='x')

            axes[1].barh(encoder_df["parameter"], encoder_df["avg_step_time_ms"], color='#d62728')
            axes[1].set_title("Encoder Size - Step Time", fontsize=14, fontweight='bold')
            axes[1].set_xlabel("Avg Step Time (ms)", fontsize=12)
            axes[1].grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plot_path = self.output_dir / "ablation_2_encoders.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved plot: {plot_path}")

        # Plot 3: Learning rates and batch sizes
        lr_bs_df = df[df["ablation"].isin(["Learning Rate", "Batch Size"])]
        if len(lr_bs_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            lr_df = df[df["ablation"] == "Learning Rate"]
            if len(lr_df) > 0:
                axes[0].plot(range(len(lr_df)), lr_df["final_loss"], marker='o', linewidth=2, markersize=8)
                axes[0].set_xticks(range(len(lr_df)))
                axes[0].set_xticklabels(lr_df["parameter"])
                axes[0].set_title("Learning Rate - Final Loss", fontsize=14, fontweight='bold')
                axes[0].set_ylabel("Final Loss", fontsize=12)
                axes[0].grid(True, alpha=0.3)

            bs_df = df[df["ablation"] == "Batch Size"]
            if len(bs_df) > 0:
                axes[1].bar(bs_df["parameter"], bs_df["avg_step_time_ms"], color='#2ca02c')
                axes[1].set_title("Batch Size - Step Time", fontsize=14, fontweight='bold')
                axes[1].set_ylabel("Avg Step Time (ms)", fontsize=12)
                axes[1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plot_path = self.output_dir / "ablation_3_hyperparams.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved plot: {plot_path}")

        # Plot 4: Additional ablations (Normalization, Max Length, Frozen)
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        norm_df = df[df["ablation"] == "Normalization"]
        if len(norm_df) > 0:
            axes[0].bar(norm_df["parameter"], norm_df["final_loss"], color=['#ff7f0e', '#1f77b4'])
            axes[0].set_title("Normalization - Final Loss", fontsize=12, fontweight='bold')
            axes[0].set_ylabel("Final Loss", fontsize=11)
            axes[0].grid(True, alpha=0.3, axis='y')

        maxlen_df = df[df["ablation"] == "Max Seq Length"]
        if len(maxlen_df) > 0:
            axes[1].plot(range(len(maxlen_df)), maxlen_df["final_loss"], marker='s', linewidth=2, markersize=8, color='#2ca02c')
            axes[1].set_xticks(range(len(maxlen_df)))
            axes[1].set_xticklabels(maxlen_df["parameter"])
            axes[1].set_title("Max Seq Length - Final Loss", fontsize=12, fontweight='bold')
            axes[1].set_ylabel("Final Loss", fontsize=11)
            axes[1].grid(True, alpha=0.3)

        frozen_df = df[df["ablation"] == "Encoder State"]
        if len(frozen_df) > 0:
            axes[2].bar(frozen_df["parameter"], frozen_df["final_loss"], color=['#d62728', '#9467bd'])
            axes[2].set_title("Frozen vs Fine-tuned - Final Loss", fontsize=12, fontweight='bold')
            axes[2].set_ylabel("Final Loss", fontsize=11)
            axes[2].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plot_path = self.output_dir / "ablation_4_other.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[OK] Saved plot: {plot_path}")

        # Plot 5: Random seeds stability
        seed_df = df[df["ablation"] == "Random Seed"]
        if len(seed_df) > 0:
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
            axes.plot(seed_df["parameter"], seed_df["final_loss"], marker='o', linewidth=2, markersize=8, color='#ff7f0e')
            axes.fill_between(
                range(len(seed_df)),
                seed_df["final_loss"] - seed_df["final_loss"].std(),
                seed_df["final_loss"] + seed_df["final_loss"].std(),
                alpha=0.2,
                color='#ff7f0e'
            )
            axes.set_title("Random Seed Stability - Final Loss", fontsize=14, fontweight='bold')
            axes.set_xlabel("Random Seed", fontsize=12)
            axes.set_ylabel("Final Loss", fontsize=12)
            axes.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = self.output_dir / "ablation_5_seeds.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"[OK] Saved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="STAGE 12: Ablation Studies for SBERT"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/ablations_stage12",
        help="Output directory for ablation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--quick-mode",
        action="store_true",
        default=True,
        help="Use subset of data for faster ablations",
    )

    args = parser.parse_args()

    ablation_study = AblationStudy(
        output_dir=args.output_dir,
        device=args.device,
        quick_mode=args.quick_mode,
    )

    results_df = ablation_study.run_all_ablations()

    print("\n" + "="*80)
    print("STAGE 12: ABLATION STUDIES - COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {ablation_study.output_dir}")
    print(f"Total ablations run: {len(results_df)}")


if __name__ == "__main__":
    main()
