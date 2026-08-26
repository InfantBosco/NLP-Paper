#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="experiments/onnx/")
    args = parser.parse_args()
    print("Exporting SBERT model to ONNX runtime format...")

if __name__ == "__main__":
    main()
