#!/usr/bin/env python
import os
import argparse
from sbert_reproduction.data.download import (
    download_dataset,
    load_stsb_tsv,
    load_nli_tsv,
    create_debug_dataset,
    compute_checksum
)
from sbert_reproduction.data.validation import validate_stsb_records, validate_nli_records
from sbert_reproduction.io_utils import save_json

def main():
    parser = argparse.ArgumentParser(description="Download and validate SBERT datasets.")
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    print("=== STAGE 4 — DATASET PREPARATION & AUDIT ===")

    # 1. Create debug dataset
    debug_path = create_debug_dataset(os.path.join(args.data_dir, "debug_dataset.json"))

    # 2. Download STSb and AllNLI
    stsb_path = download_dataset("stsb", target_dir=args.data_dir)
    nli_path = download_dataset("allnli", target_dir=args.data_dir)

    # 3. Load & Validate STSb
    print("\nValidating STS Benchmark dataset...")
    stsb_records = load_stsb_tsv(stsb_path)
    stsb_audit = validate_stsb_records(stsb_records)
    stsb_checksum = compute_checksum(stsb_path)
    print(f"STSb Audit: {stsb_audit}")

    # 4. Load & Validate AllNLI
    print("\nValidating AllNLI dataset...")
    nli_records = load_nli_tsv(nli_path)
    nli_audit = validate_nli_records(nli_records)
    nli_checksum = compute_checksum(nli_path)
    print(f"AllNLI Audit: {nli_audit}")

    # 5. Generate Manifest
    manifest = {
        "datasets": {
            "stsbenchmark": {
                "file_path": stsb_path,
                "sha256": stsb_checksum,
                "total_records": stsb_audit["total_records"],
                "split_counts": stsb_audit["split_counts"],
                "duplicates_found": stsb_audit["duplicates_found"]
            },
            "allnli": {
                "file_path": nli_path,
                "sha256": nli_checksum,
                "total_records": nli_audit["total_records"],
                "split_counts": nli_audit["split_counts"],
                "label_counts": nli_audit["label_counts"],
                "duplicates_found": nli_audit["duplicates_found"]
            },
            "debug_subset": {
                "file_path": debug_path,
                "sample_counts": {"stsb_train": 3, "nli_train": 3}
            }
        }
    }

    manifest_path = os.path.join(args.data_dir, "manifest.json")
    save_json(manifest, manifest_path)
    print(f"\nDataset manifest successfully written to {manifest_path}")

if __name__ == "__main__":
    main()
