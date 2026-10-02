"""
TruthLens — Device manager for ML inference.

Handles auto-detection of CPU/GPU/MPS, VRAM tracking,
and model placement.
"""

from __future__ import annotations

import torch
from backend.app.core.logging import get_logger

log = get_logger("device")


def resolve_device(requested: str = "auto") -> torch.device:
    """
    Resolve the compute device from a string.

    Supported values:
    - "auto": prefer CUDA > MPS > CPU
    - "cpu": force CPU
    - "cuda" / "cuda:0": specific CUDA device
    - "mps": Apple Metal (if available)
    """
    req = requested.strip().lower()

    if req == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            log.info("Auto-detected CUDA device: %s (%d MB)", name, mem)
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("Auto-detected Apple MPS device")
            return torch.device("mps")
        else:
            log.info("No GPU found — using CPU")
            return torch.device("cpu")

    if req.startswith("cuda"):
        if not torch.cuda.is_available():
            log.warning("CUDA requested but not available — falling back to CPU")
            return torch.device("cpu")
        return torch.device(req)

    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("MPS requested but not available — falling back to CPU")
        return torch.device("cpu")

    return torch.device("cpu")


def get_gpu_memory_info() -> dict:
    """Return current GPU memory usage (MB), or empty dict if no GPU."""
    if not torch.cuda.is_available():
        return {}
    return {
        "total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 ** 2),
        "allocated_mb": torch.cuda.memory_allocated(0) // (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(0) // (1024 ** 2),
        "free_mb": (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved(0)
        ) // (1024 ** 2),
    }


def clear_gpu_memory() -> None:
    """Free up cached GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.info("GPU cache cleared. %s", get_gpu_memory_info())
