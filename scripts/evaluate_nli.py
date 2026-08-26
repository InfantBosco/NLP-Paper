#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    print(f"Evaluating NLI Classification Accuracy with checkpoint: {args.checkpoint}")

if __name__ == "__main__":
    main()
