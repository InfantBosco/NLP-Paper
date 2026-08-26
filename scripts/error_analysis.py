#!/usr/bin/env python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=None)
    args = parser.parse_args()
    print("Running Error Analysis Diagnostics...")

if __name__ == "__main__":
    main()
