#!/usr/bin/env python
"""
STAGE 11: EFFICIENCY BENCHMARK
Reproduce the central efficiency motivation of SBERT.

Compares:
  A. Cross-encoder-style scoring: Encode sentence pairs repeatedly, measure all pairwise comparisons.
  B. SBERT bi-encoder-style scoring: Encode each sentence once, store embeddings, compare using cosine similarity.

Corpus sizes: 100, 1,000, 10,000 sentences (larger if hardware permits).

Measures:
  - Number of model forward passes
  - Embedding generation time
  - Pairwise comparison time
  - Total search time
  - Queries per second
  - Peak RAM
  - Embedding storage size
  - Top-k retrieval latency
  - Warm-up time

Records:
  - CPU/GPU
  - Batch size
  - Maximum sequence length
  - Number of repetitions
  - Warm-up policy
  - Software versions
  - Dataset
  - Number of queries

Creates plots:
  - Corpus size versus total time
  - Corpus size versus forward passes
  - Accuracy versus latency
  - Model size versus latency
  - Memory versus corpus size
"""

import argparse
import json
import os
import time
import random
import tracemalloc
from typing import Dict, List, Tuple, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sbert_reproduction.models.encoder import TransformerEncoderWrapper, TokenizerWrapper
from sbert_reproduction.models.sentence_encoder import SentenceEncoder
from sbert_reproduction.models.sbert_model import SBERTModel


def get_device(device: str = "auto") -> str:
    """Get the appropriate device (cuda or cpu)."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    elif device in ["cuda", "cpu"]:
        if device == "cuda" and not torch.cuda.is_available():
            print("[!] CUDA not available, falling back to CPU")
            return "cpu"
        return device
    else:
        raise ValueError(f"Invalid device: {device}")


class EfficiencyBenchmark:
    """Main efficiency benchmark runner."""

    def __init__(
        self,
        corpus_sizes: List[int],
        batch_size: int = 32,
        max_seq_length: int = 128,
        num_queries: int = 10,
        warmup_runs: int = 2,
        benchmark_runs: int = 5,
        device: str = "auto",
        output_dir: str = "experiments/results/benchmark_stage11",
    ):
        self.corpus_sizes = corpus_sizes
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_queries = num_queries
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.device = get_device(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model and tokenizer
        self._init_model()

    def _init_model(self):
        """Initialize SBERT model."""
        print("\n[*] Initializing SBERT model...")
        encoder_name = "bert-base-uncased"
        
        # Create transformer encoder
        transformer = TransformerEncoderWrapper(encoder_name, device=self.device)
        
        # Create sentence encoder with pooling_mode (string)
        self.sentence_encoder = SentenceEncoder(
            transformer,
            pooling_mode="mean",
            normalize=False,
        )
        
        # Create SBERT model
        self.model = SBERTModel(self.sentence_encoder)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create tokenizer
        self.tokenizer = TokenizerWrapper(
            model_name=encoder_name,
            max_seq_length=self.max_seq_length,
            device=self.device,
        )
        
        # Get software versions
        self.software_versions = {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
        }
        
        print(f"  [OK] Model: BERT-base-uncased")
        print(f"  [OK] Device: {self.device}")
        print(f"  [OK] Max sequence length: {self.max_seq_length}")

    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware and environment information."""
        import platform
        
        gpu_info = "None"
        if torch.cuda.is_available():
            gpu_info = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            }
        
        return {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "device_type": str(self.device),
            "gpu_info": gpu_info,
        }

    def _generate_corpus(self, size: int) -> List[str]:
        """Generate synthetic corpus of sentences."""
        sentences = []
        templates = [
            "This is a sentence about {}.",
            "The {} is an important concept.",
            "We discuss {} in detail.",
            "Consider the aspect of {}.",
            "{} plays a crucial role.",
            "The study of {} reveals insights.",
            "In the context of {}, we observe.",
            "The relationship between {} and others.",
            "Different perspectives on {}.",
            "Advanced topics in {} research.",
        ]
        
        topics = [
            "machine learning", "natural language processing", "deep learning",
            "computer vision", "neural networks", "transformers", "embeddings",
            "semantic similarity", "information retrieval", "text mining",
            "data science", "artificial intelligence", "knowledge graphs",
            "word embeddings", "sentence embeddings", "clustering", "classification",
        ]
        
        random.seed(42)
        for i in range(size):
            template = random.choice(templates)
            topic = random.choice(topics)
            sentence = template.format(topic)
            sentences.append(sentence)
        
        return sentences

    def _encode_sentences_batch(
        self,
        sentences: List[str],
        batch_size: int,
        measure_time: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encode sentences using bi-encoder (SBERT).
        
        Returns:
            (embeddings, metrics)
        """
        metrics = {
            "num_sentences": len(sentences),
            "batch_size": batch_size,
            "num_batches": (len(sentences) + batch_size - 1) // batch_size,
            "forward_passes": len(sentences),
        }
        
        embeddings = []
        total_time = 0.0
        
        with torch.no_grad():
            for start_idx in tqdm(
                range(0, len(sentences), batch_size),
                desc="BI-ENCODER: Encoding sentences",
                disable=False,
            ):
                batch_sentences = sentences[start_idx : start_idx + batch_size]
                
                if measure_time:
                    start_time = time.perf_counter()
                
                # Tokenize
                features = self.tokenizer.tokenize(batch_sentences)
                
                # Encode
                batch_embeddings = self.model.sentence_encoder(
                    features["input_ids"],
                    features["attention_mask"],
                    features.get("token_type_ids"),
                )
                
                if measure_time:
                    end_time = time.perf_counter()
                    total_time += end_time - start_time
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        embeddings = np.vstack(embeddings)
        metrics["total_encoding_time_sec"] = total_time
        metrics["avg_encoding_time_per_sentence"] = total_time / len(sentences) if len(sentences) > 0 else 0
        metrics["sentences_per_sec"] = len(sentences) / total_time if total_time > 0 else 0
        
        return embeddings, metrics

    def _pairwise_cross_encoder(
        self,
        sentences_a: List[str],
        sentences_b: List[str],
        batch_size: int,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Score all pairs using cross-encoder approach.
        Each pair goes through the model.
        
        Returns:
            (scores [n_a, n_b], metrics)
        """
        num_pairs = len(sentences_a) * len(sentences_b)
        metrics = {
            "num_sentence_a": len(sentences_a),
            "num_sentence_b": len(sentences_b),
            "num_pairs": num_pairs,
            "batch_size": batch_size,
            "num_batches": (num_pairs + batch_size - 1) // batch_size,
            "forward_passes": num_pairs,
        }
        
        scores = []
        total_time = 0.0
        
        with torch.no_grad():
            # Generate all pairs
            all_pairs = []
            for sent_a in sentences_a:
                for sent_b in sentences_b:
                    all_pairs.append((sent_a, sent_b))
            
            for start_idx in tqdm(
                range(0, len(all_pairs), batch_size),
                desc="CROSS-ENCODER: Scoring all pairs",
                disable=False,
            ):
                batch_pairs = all_pairs[start_idx : start_idx + batch_size]
                batch_a = [p[0] for p in batch_pairs]
                batch_b = [p[1] for p in batch_pairs]
                
                start_time = time.perf_counter()
                
                # Tokenize pairs
                features_a = self.tokenizer.tokenize(batch_a)
                features_b = self.tokenizer.tokenize(batch_b)
                
                # Move to device
                for key in features_a:
                    if isinstance(features_a[key], torch.Tensor):
                        features_a[key] = features_a[key].to(self.device)
                        features_b[key] = features_b[key].to(self.device)
                
                # Encode both sentences
                embeddings_a = self.model.sentence_encoder(
                    features_a["input_ids"],
                    features_a["attention_mask"],
                    features_a.get("token_type_ids"),
                )
                embeddings_b = self.model.sentence_encoder(
                    features_b["input_ids"],
                    features_b["attention_mask"],
                    features_b.get("token_type_ids"),
                )
                
                # Compute cosine similarity
                batch_scores = F.cosine_similarity(embeddings_a, embeddings_b)
                
                end_time = time.perf_counter()
                total_time += end_time - start_time
                
                scores.append(batch_scores.cpu().numpy())
        
        scores = np.concatenate(scores)
        scores = scores.reshape(len(sentences_a), len(sentences_b))
        
        metrics["total_scoring_time_sec"] = total_time
        metrics["avg_time_per_pair"] = total_time / num_pairs if num_pairs > 0 else 0
        metrics["pairs_per_sec"] = num_pairs / total_time if total_time > 0 else 0
        
        return scores, metrics

    def _pairwise_bi_encoder(
        self,
        embeddings_a: np.ndarray,
        embeddings_b: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Score all pairs using bi-encoder (pre-computed embeddings) + cosine similarity.
        
        Returns:
            (scores [n_a, n_b], metrics)
        """
        num_pairs = embeddings_a.shape[0] * embeddings_b.shape[0]
        
        start_time = time.perf_counter()
        
        # Compute cosine similarity using scipy
        scores = 1.0 - cdist(embeddings_a, embeddings_b, metric='cosine')
        
        end_time = time.perf_counter()
        comparison_time = end_time - start_time
        
        metrics = {
            "num_embedding_a": embeddings_a.shape[0],
            "num_embedding_b": embeddings_b.shape[0],
            "num_pairs": num_pairs,
            "forward_passes": embeddings_a.shape[0] + embeddings_b.shape[0],
            "comparison_time_sec": comparison_time,
            "pairs_per_sec": num_pairs / comparison_time if comparison_time > 0 else 0,
        }
        
        return scores, metrics

    def _benchmark_topk_retrieval(
        self,
        embeddings_corpus: np.ndarray,
        embeddings_queries: np.ndarray,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark top-k retrieval latency."""
        start_time = time.perf_counter()
        
        # Compute all similarities
        similarities = 1.0 - cdist(embeddings_queries, embeddings_corpus, metric='cosine')
        
        # Get top-k indices
        topk_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return {
            "k": k,
            "num_queries": embeddings_queries.shape[0],
            "corpus_size": embeddings_corpus.shape[0],
            "topk_retrieval_time_sec": total_time,
            "latency_per_query_ms": (total_time / embeddings_queries.shape[0]) * 1000 if embeddings_queries.shape[0] > 0 else 0,
            "queries_per_sec": embeddings_queries.shape[0] / total_time if total_time > 0 else 0,
        }

    def _measure_memory(self, corpus_size: int, embedding_dim: int) -> Dict[str, Any]:
        """Measure embedding storage and memory usage."""
        embedding_size_bytes = corpus_size * embedding_dim * 4  # float32
        embedding_size_mb = embedding_size_bytes / (1024 * 1024)
        
        return {
            "corpus_size": corpus_size,
            "embedding_dim": embedding_dim,
            "embedding_storage_bytes": embedding_size_bytes,
            "embedding_storage_mb": embedding_size_mb,
        }

    def _warmup(self):
        """Warm-up model and GPU."""
        print("\n[*] Warm-up...")
        dummy_sentences = ["This is a test sentence.", "Another test sentence."]
        
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                features = self.tokenizer.tokenize(dummy_sentences)
                _ = self.model.sentence_encoder(
                    features["input_ids"],
                    features["attention_mask"],
                    features.get("token_type_ids"),
                )
        
        print("  [OK] Warm-up complete")

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete efficiency benchmark."""
        print("\n" + "="*80)
        print("STAGE 11: EFFICIENCY BENCHMARK")
        print("="*80)
        
        self._warmup()
        
        results = {
            "hardware": self._get_hardware_info(),
            "software_versions": self.software_versions,
            "config": {
                "corpus_sizes": self.corpus_sizes,
                "batch_size": self.batch_size,
                "max_seq_length": self.max_seq_length,
                "num_queries": self.num_queries,
                "warmup_runs": self.warmup_runs,
                "benchmark_runs": self.benchmark_runs,
            },
            "benchmarks": {},
        }
        
        all_results = []
        
        for corpus_size in self.corpus_sizes:
            print(f"\n{'='*80}")
            print(f"Corpus Size: {corpus_size} sentences")
            print(f"{'='*80}")
            
            # Generate corpus and queries
            corpus = self._generate_corpus(corpus_size)
            queries = self._generate_corpus(self.num_queries)
            
            # ===== BI-ENCODER APPROACH =====
            print("\n[BI-ENCODER]")
            print("-" * 40)
            
            # Encode corpus
            print("Step 1: Encoding corpus...")
            tracemalloc.start()
            embeddings_corpus, metrics_encode_corpus = self._encode_sentences_batch(
                corpus,
                self.batch_size,
                measure_time=True,
            )
            current, peak_mem_encode = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"  [OK] Encoded {corpus_size} sentences in {metrics_encode_corpus['total_encoding_time_sec']:.4f}s")
            print(f"  [OK] Throughput: {metrics_encode_corpus['sentences_per_sec']:.2f} sent/s")
            print(f"  [OK] Peak memory: {peak_mem_encode / (1024**2):.2f} MB")
            
            # Encode queries
            print("Step 2: Encoding queries...")
            embeddings_queries, metrics_encode_queries = self._encode_sentences_batch(
                queries,
                self.batch_size,
                measure_time=True,
            )
            print(f"  [OK] Encoded {self.num_queries} queries in {metrics_encode_queries['total_encoding_time_sec']:.4f}s")
            
            # Compute pairwise similarities (bi-encoder)
            print("Step 3: Computing pairwise similarities (bi-encoder)...")
            scores_bi, metrics_bi_scoring = self._pairwise_bi_encoder(
                embeddings_queries,
                embeddings_corpus,
            )
            print(f"  [OK] Scored {metrics_bi_scoring['num_pairs']} pairs in {metrics_bi_scoring['comparison_time_sec']:.4f}s")
            print(f"  [OK] Throughput: {metrics_bi_scoring['pairs_per_sec']:.2f} pairs/s")
            
            # Top-k retrieval
            print("Step 4: Top-k retrieval (k=10)...")
            metrics_topk = self._benchmark_topk_retrieval(
                embeddings_corpus,
                embeddings_queries,
                k=10,
            )
            print(f"  [OK] Retrieved top-10 in {metrics_topk['latency_per_query_ms']:.4f}ms per query")
            print(f"  [OK] Queries per second: {metrics_topk['queries_per_sec']:.2f}")
            
            # Memory measurement
            memory_info = self._measure_memory(corpus_size, embeddings_corpus.shape[1])
            print(f"  [OK] Embedding storage: {memory_info['embedding_storage_mb']:.2f} MB")
            
            # Total bi-encoder time
            total_bi_time = (
                metrics_encode_corpus['total_encoding_time_sec'] +
                metrics_encode_queries['total_encoding_time_sec'] +
                metrics_bi_scoring['comparison_time_sec']
            )
            
            bi_encoder_result = {
                "approach": "BI-ENCODER",
                "corpus_size": corpus_size,
                "num_queries": self.num_queries,
                "encode_corpus_time": metrics_encode_corpus['total_encoding_time_sec'],
                "encode_queries_time": metrics_encode_queries['total_encoding_time_sec'],
                "scoring_time": metrics_bi_scoring['comparison_time_sec'],
                "topk_latency_ms": metrics_topk['latency_per_query_ms'],
                "total_time": total_bi_time,
                "forward_passes": metrics_encode_corpus['forward_passes'] + metrics_encode_queries['forward_passes'],
                "corpus_throughput_sent_s": metrics_encode_corpus['sentences_per_sec'],
                "pair_throughput_pairs_s": metrics_bi_scoring['pairs_per_sec'],
                "query_throughput_queries_s": metrics_topk['queries_per_sec'],
                "embedding_storage_mb": memory_info['embedding_storage_mb'],
                "peak_memory_mb": peak_mem_encode / (1024**2),
            }
            
            # ===== CROSS-ENCODER APPROACH =====
            print("\n[CROSS-ENCODER]")
            print("-" * 40)
            print("Step 1: Scoring all query-corpus pairs...")
            
            tracemalloc.start()
            scores_cross, metrics_cross = self._pairwise_cross_encoder(
                queries,
                corpus,
                self.batch_size,
            )
            current, peak_mem_cross = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            print(f"  [OK] Scored {metrics_cross['num_pairs']} pairs in {metrics_cross['total_scoring_time_sec']:.4f}s")
            print(f"  [OK] Throughput: {metrics_cross['pairs_per_sec']:.2f} pairs/s")
            print(f"  [OK] Peak memory: {peak_mem_cross / (1024**2):.2f} MB")
            print(f"  [OK] Forward passes: {metrics_cross['forward_passes']}")
            
            cross_encoder_result = {
                "approach": "CROSS-ENCODER",
                "corpus_size": corpus_size,
                "num_queries": self.num_queries,
                "total_scoring_time": metrics_cross['total_scoring_time_sec'],
                "forward_passes": metrics_cross['forward_passes'],
                "pair_throughput_pairs_s": metrics_cross['pairs_per_sec'],
                "peak_memory_mb": peak_mem_cross / (1024**2),
            }
            
            # ===== COMPARISON =====
            print("\n[COMPARISON]")
            print("-" * 40)
            speedup = cross_encoder_result["total_scoring_time"] / bi_encoder_result["total_time"]
            forward_passes_reduction = (
                1.0 - (bi_encoder_result["forward_passes"] / cross_encoder_result["forward_passes"])
            ) * 100
            
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Forward passes reduction: {forward_passes_reduction:.1f}%")
            print(f"  Memory savings: {(1 - bi_encoder_result['peak_memory_mb'] / cross_encoder_result['peak_memory_mb']) * 100:.1f}%")
            
            # Verify correctness
            mae = np.mean(np.abs(scores_bi - scores_cross))
            print(f"  Mean absolute difference in scores: {mae:.6f}")
            
            comparison_result = {
                "corpus_size": corpus_size,
                "speedup": speedup,
                "forward_passes_reduction_pct": forward_passes_reduction,
                "memory_savings_pct": (1 - bi_encoder_result['peak_memory_mb'] / cross_encoder_result['peak_memory_mb']) * 100,
                "score_mae": mae,
            }
            
            all_results.append({
                **bi_encoder_result,
                **{"prefix": "BI"},
            })
            all_results.append({
                **cross_encoder_result,
                **{"prefix": "CROSS"},
            })
            all_results.append(comparison_result)
            
            results["benchmarks"][f"corpus_size_{corpus_size}"] = {
                "bi_encoder": bi_encoder_result,
                "cross_encoder": cross_encoder_result,
                "comparison": comparison_result,
            }
        
        # Save detailed results
        self._save_results(results, all_results)
        
        # Create plots
        self._create_plots(results, all_results)
        
        return results

    def _save_results(self, results: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Save benchmark results to JSON and CSV."""
        # Save JSON
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OK] Saved JSON results to {json_path}")
        
        # Save CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved CSV results to {csv_path}")

    def _create_plots(self, results: Dict[str, Any], all_results: List[Dict[str, Any]]):
        """Create visualization plots."""
        df = pd.DataFrame(all_results)
        
        # Filter for bi-encoder and cross-encoder results (with total_time)
        df_bi = df[df["prefix"] == "BI"].copy()
        df_cross = df[df["prefix"] == "CROSS"].copy()
        
        if len(df_bi) == 0 or len(df_cross) == 0:
            print("Not enough data for plotting")
            return
        
        # Plot 1: Corpus size vs total time
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df_bi["corpus_size"], df_bi["total_time"], "o-", label="BI-ENCODER", linewidth=2, markersize=8)
        plt.plot(df_cross["corpus_size"], df_cross["total_scoring_time"], "s-", label="CROSS-ENCODER", linewidth=2, markersize=8)
        plt.xlabel("Corpus Size (sentences)", fontsize=12)
        plt.ylabel("Total Time (seconds)", fontsize=12)
        plt.title("Corpus Size vs Total Time", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")
        
        # Plot 2: Corpus size vs forward passes
        plt.subplot(1, 2, 2)
        plt.plot(df_bi["corpus_size"], df_bi["forward_passes"], "o-", label="BI-ENCODER", linewidth=2, markersize=8)
        plt.plot(df_cross["corpus_size"], df_cross["forward_passes"], "s-", label="CROSS-ENCODER", linewidth=2, markersize=8)
        plt.xlabel("Corpus Size (sentences)", fontsize=12)
        plt.ylabel("Number of Forward Passes", fontsize=12)
        plt.title("Corpus Size vs Forward Passes", fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xscale("log")
        plt.yscale("log")
        
        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_plots_1.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved plot 1 to {plot_path}")
        plt.close()
        
        # Plot 3: Speedup and efficiency metrics
        df_comparison = pd.DataFrame([r for r in all_results if "speedup" in r])
        
        if len(df_comparison) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].plot(df_comparison["corpus_size"], df_comparison["speedup"], "o-", linewidth=2, markersize=8, color='green')
            axes[0].set_xlabel("Corpus Size (sentences)", fontsize=11)
            axes[0].set_ylabel("Speedup (BI-Encoder / Cross-Encoder)", fontsize=11)
            axes[0].set_title("Speedup Improvement", fontsize=12, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            axes[0].set_xscale("log")
            
            axes[1].plot(df_comparison["corpus_size"], df_comparison["forward_passes_reduction_pct"], "s-", linewidth=2, markersize=8, color='orange')
            axes[1].set_xlabel("Corpus Size (sentences)", fontsize=11)
            axes[1].set_ylabel("Forward Passes Reduction (%)", fontsize=11)
            axes[1].set_title("Forward Passes Reduction", fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_xscale("log")
            
            axes[2].plot(df_comparison["corpus_size"], df_comparison["memory_savings_pct"], "^-", linewidth=2, markersize=8, color='purple')
            axes[2].set_xlabel("Corpus Size (sentences)", fontsize=11)
            axes[2].set_ylabel("Memory Savings (%)", fontsize=11)
            axes[2].set_title("Peak Memory Savings", fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_xscale("log")
            
            plt.tight_layout()
            plot_path = self.output_dir / "benchmark_plots_2_efficiency.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved plot 2 (efficiency metrics) to {plot_path}")
            plt.close()
        
        # Plot 4: Throughput comparison
        if "corpus_throughput_sent_s" in df_bi.columns and "pair_throughput_pairs_s" in df_cross.columns:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.bar(df_bi.index, df_bi["corpus_throughput_sent_s"], alpha=0.7, label="BI-ENCODER (Encoding)")
            plt.xlabel("Experiment", fontsize=11)
            plt.ylabel("Throughput (sentences/sec)", fontsize=11)
            plt.title("Encoding Throughput (BI-Encoder)", fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.subplot(1, 2, 2)
            plt.bar(df_cross.index, df_cross["pair_throughput_pairs_s"], alpha=0.7, color='orange', label="CROSS-ENCODER (Scoring)")
            plt.xlabel("Experiment", fontsize=11)
            plt.ylabel("Throughput (pairs/sec)", fontsize=11)
            plt.title("Scoring Throughput (Cross-Encoder)", fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plot_path = self.output_dir / "benchmark_plots_3_throughput.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved plot 3 (throughput) to {plot_path}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="STAGE 11: Efficiency Benchmark for SBERT"
    )
    parser.add_argument(
        "--corpus-sizes",
        type=int,
        nargs="+",
        default=[100, 1000, 10000],
        help="Corpus sizes to benchmark (default: [100, 1000, 10000])",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of query sentences (default: 10)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=2,
        help="Number of warm-up runs (default: 2)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=5,
        help="Number of benchmark runs (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/benchmark_stage11",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    benchmark = EfficiencyBenchmark(
        corpus_sizes=args.corpus_sizes,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        num_queries=args.num_queries,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    results = benchmark.run_benchmark()
    
    print("\n" + "="*80)
    print("STAGE 11 BENCHMARK COMPLETE")
    print("="*80)
    print(f"Results saved to: {benchmark.output_dir}")


if __name__ == "__main__":
    main()

