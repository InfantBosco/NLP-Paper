import os
import gzip
import csv
import hashlib
import json
import requests
from typing import Dict, Any, List

DATASET_URLS = {
    "stsb": "https://sbert.net/datasets/stsbenchmark.tsv.gz",
    "allnli": "https://sbert.net/datasets/AllNLI.tsv.gz"
}

def compute_checksum(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_dataset(dataset_name: str, target_dir: str = "data") -> str:
    """Downloads dataset from sbert.net mirrors with User-Agent header if not present locally."""
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Valid keys: {list(DATASET_URLS.keys())}")

    os.makedirs(target_dir, exist_ok=True)
    url = DATASET_URLS[dataset_name]
    filename = url.split("/")[-1]
    target_path = os.path.join(target_dir, filename)

    if not os.path.exists(target_path):
        print(f"Downloading {dataset_name} from {url} to {target_path}...")
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Download complete: {target_path}")
    else:
        print(f"Dataset {dataset_name} already exists at {target_path}")

    return target_path

def load_stsb_tsv(filepath: str) -> List[Dict[str, Any]]:
    """Reads STSb TSV file and returns parsed records."""
    records = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            records.append({
                "split": row["split"],
                "genre": row.get("genre", ""),
                "dataset": row.get("dataset", ""),
                "year": row.get("year", ""),
                "sid": row.get("sid", ""),
                "score": float(row["score"]),
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"]
            })
    return records

def load_nli_tsv(filepath: str) -> List[Dict[str, Any]]:
    """Reads AllNLI TSV file and returns parsed records."""
    records = []
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            records.append({
                "split": row["split"],
                "dataset": row.get("dataset", ""),
                "label": row["label"],
                "sentence1": row["sentence1"],
                "sentence2": row["sentence2"]
            })
    return records

def create_debug_dataset(target_path: str = "data/debug_dataset.json") -> str:
    """Creates a tiny local debug dataset containing STSb and NLI samples."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    debug_data = {
        "stsb_train": [
            {"sentence1": "A plane is taking off.", "sentence2": "An airplane is taking off.", "score": 5.0, "normalized_score": 1.0},
            {"sentence1": "A man is playing a guitar.", "sentence2": "A man is playing a flute.", "score": 2.5, "normalized_score": 0.5},
            {"sentence1": "A dog is running.", "sentence2": "A cat is sleeping.", "score": 0.0, "normalized_score": 0.0}
        ],
        "nli_train": [
            {"sentence1": "A man is playing chess.", "sentence2": "A person is playing a board game.", "label": "entailment", "label_id": 1},
            {"sentence1": "A man is sleeping.", "sentence2": "A man is running a marathon.", "label": "contradiction", "label_id": 0},
            {"sentence1": "A man is eating an apple.", "sentence2": "The man likes fruits.", "label": "neutral", "label_id": 2}
        ]
    }
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2)
    print(f"Debug dataset created at {target_path}")
    return target_path
