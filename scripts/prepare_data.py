#!/usr/bin/env python
import argparse
from sbert_reproduction.data.download import download_dataset
from sbert_reproduction.config import ExperimentConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sbert_stsb.yaml")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Preparing datasets for experiment: {cfg.experiment_name}")
    download_dataset("stsb")
    download_dataset("allnli")
    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
