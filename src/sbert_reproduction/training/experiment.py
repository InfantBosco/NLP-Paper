"""
Experiment directory management for Stage 7.

Each experiment creates a unique directory containing:
  - resolved_config.json      — full config with all defaults filled in
  - environment.json          — Python + library versions + hardware
  - hardware.json             — CPU/GPU info
  - git_revision.json         — commit hash, branch, dirty flag
  - seed.txt                  — random seed used
  - command.txt               — exact command that launched the experiment
  - manifest.json             — summary of all files written

Usage::

    exp = ExperimentManifest(output_dir, config_dict, args_command)
    exp.save_manifest()
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, Optional


def _get_git_info() -> Dict[str, Any]:
    """Attempt to capture git commit, branch, and dirty status."""
    info: Dict[str, Any] = {"available": False}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        info = {
            "available":  True,
            "commit":     commit,
            "branch":     branch,
            "dirty":      bool(status),
            "status_lines": len(status.splitlines()) if status else 0,
        }
    except Exception as e:
        info["error"] = str(e)
    return info


def _get_hardware_info() -> Dict[str, Any]:
    hw: Dict[str, Any] = {
        "cpu": platform.processor(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    try:
        import torch
        hw["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hw["cuda_device_count"] = torch.cuda.device_count()
            hw["cuda_device_name"]  = torch.cuda.get_device_name(0)
        else:
            hw["cuda_device_count"] = 0
            hw["cuda_device_name"]  = "N/A"
    except ImportError:
        hw["cuda_available"] = False
    return hw


def _get_software_versions() -> Dict[str, str]:
    pkgs = ["torch", "transformers", "datasets", "numpy", "scipy", "sklearn"]
    versions: Dict[str, str] = {"python": sys.version}
    for pkg in pkgs:
        try:
            import importlib
            m = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


class ExperimentManifest:
    """
    Creates and populates a reproducible experiment directory.

    Args:
        output_dir:   Path to the experiment output directory.
        config_dict:  Resolved configuration dictionary.
        command:      sys.argv string or equivalent command string.
    """

    def __init__(
        self,
        output_dir: str,
        config_dict: Dict[str, Any],
        command: Optional[str] = None,
    ) -> None:
        self.output_dir  = output_dir
        self.config_dict = config_dict
        self.command     = command or " ".join(sys.argv)
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def save_manifest(self) -> str:
        """Write all provenance files and return the manifest path."""
        files_written = []

        def _write(filename: str, content: Any) -> str:
            path = os.path.join(self.output_dir, filename)
            if isinstance(content, dict) or isinstance(content, list):
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(content, fh, indent=2)
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(str(content))
            files_written.append(filename)
            return path

        # Resolved configuration
        _write("resolved_config.json", self.config_dict)

        # Environment
        env = {
            "python_version": sys.version,
            "platform":       platform.platform(),
        }
        _write("environment.json", env)

        # Hardware
        _write("hardware.json", _get_hardware_info())

        # Software versions
        _write("software_versions.json", _get_software_versions())

        # Git revision
        _write("git_revision.json", _get_git_info())

        # Seed
        seed = self.config_dict.get("seed", "unknown")
        _write("seed.txt", seed)

        # Command
        _write("command.txt", self.command)

        # Master manifest
        manifest = {
            "experiment_name": self.config_dict.get("experiment_name", "unknown"),
            "created_at":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "output_dir":      self.output_dir,
            "files":           files_written,
            "config":          self.config_dict,
            "environment":     env,
            "hardware":        _get_hardware_info(),
            "git":             _get_git_info(),
        }
        manifest_path = _write("manifest.json", manifest)

        return manifest_path
