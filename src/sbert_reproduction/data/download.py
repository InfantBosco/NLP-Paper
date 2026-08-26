import os
import urllib.request
import gzip
import shutil

DATASET_URLS = {
    "stsb": "https://sbert.net/datasets/stsbenchmark.tsv.gz",
    "allnli": "https://sbert.net/datasets/AllNLI.tsv.gz"
}

def download_dataset(dataset_name: str, target_dir: str = "data") -> str:
    """Downloads dataset from sbert.net mirrors if not present locally."""
    if dataset_name not in DATASET_URLS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Valid keys: {list(DATASET_URLS.keys())}")

    os.makedirs(target_dir, exist_ok=True)
    url = DATASET_URLS[dataset_name]
    filename = url.split("/")[-1]
    target_path = os.path.join(target_dir, filename)

    if not os.path.exists(target_path):
        print(f"Downloading {dataset_name} from {url} to {target_path}...")
        urllib.request.urlretrieve(url, target_path)
        print("Download complete.")
    else:
        print(f"Dataset {dataset_name} already exists at {target_path}")

    return target_path
