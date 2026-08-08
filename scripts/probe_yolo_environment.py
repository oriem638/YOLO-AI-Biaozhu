"""Verify the exact supported ML development environment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE_RUNTIME_ROOT = PROJECT_ROOT / "build" / "probe-runtime"
YOLO_CONFIG_ROOT = PROBE_RUNTIME_ROOT / "ultralytics"
MATPLOTLIB_CONFIG_ROOT = PROBE_RUNTIME_ROOT / "matplotlib"
YOLO_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
MATPLOTLIB_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CONFIG_ROOT))

EXPECTED = {
    "python": (3, 11),
    "torch": "2.11.0+cu128",
    "torchvision": "0.26.0+cu128",
    "ultralytics": "8.4.82",
}


def main() -> int:
    import torch
    import torchvision
    import ultralytics

    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    actual = {
        "python": tuple(sys.version_info[:2]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
    }
    if actual != EXPECTED:
        raise SystemExit(
            f"Version lock mismatch: expected={EXPECTED!r}, actual={actual!r}"
        )

    cuda_available = torch.cuda.is_available()
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    print(json.dumps(payload, ensure_ascii=False))
    if args.require_gpu and not cuda_available:
        raise SystemExit("A CUDA GPU was required but torch.cuda.is_available() is false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
