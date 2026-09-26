"""Opt-in per-process VRAM cap for GPU jobs that share a device.

``OMNIVGGT_VRAM_LIMIT_GB`` (GiB, 2**30 bytes) caps the CUDA caching allocator of the process on its device with
``torch.cuda.set_per_process_memory_fraction``; an allocation beyond the cap raises an out-of-memory error in this
process instead of taking memory from the other jobs on the GPU. Unset or empty: no cap. The training entry points
and the stream tools call ``apply_vram_limit`` at startup, before the model is built, and record what it returns
with ``memory_peaks`` at the end.
"""

import logging
import math
import os
from typing import Optional

import torch

ENV = "OMNIVGGT_VRAM_LIMIT_GB"
GIB = 2**30
MIB = 2**20

logger = logging.getLogger(__name__)


def vram_limit_gb() -> Optional[float]:
    """The cap in GiB from ``OMNIVGGT_VRAM_LIMIT_GB`` (None when unset or empty); anything but a positive finite
    number is an error."""
    text = os.environ.get(ENV, "").strip()
    if not text:
        return None
    try:
        limit = float(text)
    except ValueError:
        limit = math.nan
    if not (math.isfinite(limit) and limit > 0):
        raise ValueError(f"{ENV} must be a positive number of GiB, got {text!r}")
    return limit


def apply_vram_limit(device) -> Optional[dict]:
    """Cap the memory of this process on the CUDA ``device`` at ``OMNIVGGT_VRAM_LIMIT_GB`` GiB; call it before the
    model is built. Returns None without the variable, else ``{"limit_gb", "fraction", "total_gb", "device"}``. A cap
    above the device memory, or for a device that is not CUDA, is an error (it is never clamped or skipped)."""
    limit = vram_limit_gb()
    if limit is None:
        return None
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"{ENV}={limit:g} caps the memory of a CUDA device, but the device is {device}")
    if not torch.cuda.is_available():
        raise ValueError(f"{ENV}={limit:g} caps the memory of a CUDA device, but CUDA is not available")
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    if limit * GIB > total:
        raise ValueError(f"{ENV}={limit:g} GiB exceeds the {total / GIB:.2f} GiB of cuda:{index}")
    fraction = limit * GIB / total
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    record = {"limit_gb": limit, "fraction": fraction, "total_gb": total / GIB, "device": f"cuda:{index}"}
    logger.info(f"VRAM limit: {limit:g} GiB of {total / GIB:.2f} GiB on cuda:{index} (fraction {fraction:.4f})")
    return record


def memory_peaks(device) -> Optional[dict]:
    """The peak memory allocated and reserved by this process on the CUDA ``device`` (MiB); None off CUDA."""
    device = torch.device(device)
    if device.type != "cuda":
        return None
    return {"max_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
            "max_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB}
