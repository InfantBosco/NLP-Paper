#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ablations.yaml")
    args = parser.parse_args()
    print("Running SBERT Ablation Studies...")

if __name__ == "__main__":
    main()
