import random
import os
import platform
import sys
from typing import Dict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if HAS_NUMPY:
        np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_software_versions() -> Dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda"] = torch.version.cuda or "N/A"
    except ImportError:
        versions["torch"] = "not installed"

    try:
        import torchvision
        versions["torchvision"] = torchvision.__version__
    except ImportError:
        versions["torchvision"] = "not installed"

    try:
        import PySide6
        versions["PySide6"] = PySide6.__version__
    except ImportError:
        versions["PySide6"] = "not installed"

    try:
        import PIL
        versions["Pillow"] = PIL.__version__
    except ImportError:
        versions["Pillow"] = "not installed"

    try:
        import cv2
        versions["opencv"] = cv2.__version__
    except ImportError:
        versions["opencv"] = "not installed"

    return versions
