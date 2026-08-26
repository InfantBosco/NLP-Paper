#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/sbert_stsb.yaml")
    args = parser.parse_args()
    print(f"Evaluating STSb Benchmark with checkpoint: {args.checkpoint}")

if __name__ == "__main__":
    main()
