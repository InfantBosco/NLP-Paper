import os
from sbert_reproduction.io_utils import save_json
from sbert_reproduction.environment import get_environment_info

class ExperimentManifest:
    """Creates directory & metadata manifest for reproducible experiments."""
    def __init__(self, output_dir: str, config_dict: dict):
        self.output_dir = output_dir
        self.config_dict = config_dict
        os.makedirs(output_dir, exist_ok=True)

    def save_manifest(self) -> str:
        manifest = {
            "config": self.config_dict,
            "environment": get_environment_info()
        }
        path = os.path.join(self.output_dir, "manifest.json")
        save_json(manifest, path)
        return path
