import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ExperimentConfig:
    experiment_name: str = "default_experiment"
    seed: int = 42
    output_dir: str = "experiments/results/default"
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg_dict = yaml.safe_load(f) or {}
        return cls(
            experiment_name=cfg_dict.get("experiment_name", "default"),
            seed=cfg_dict.get("seed", 42),
            output_dir=cfg_dict.get("output_dir", "experiments/results/default"),
            raw_config=cfg_dict
        )
