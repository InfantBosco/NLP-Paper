#!/usr/bin/env python
import argparse
from sbert_reproduction.config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline_tfidf.yaml")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Running Baseline Experiment: {cfg.experiment_name}")

if __name__ == "__main__":
    main()
