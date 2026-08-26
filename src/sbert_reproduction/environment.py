import os
import platform
import sys
from typing import Dict, Any

def get_environment_info() -> Dict[str, Any]:
    """Collects system, hardware, and python package versions."""
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = None
        info["cuda_available"] = False

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = None

    return info
