# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Utility modules for hardware detection, metrics, and helpers."""

from .hardware import pick_hw_backend, set_windows_timer_resolution
from .metrics import RateMeter, PerformanceTracker
from .helpers import (
    is_youtube_url, is_http_url, resolve_local_path, truthy,
    compute_spacing_and_group
)

__all__ = [
    # Hardware
    "pick_hw_backend", "set_windows_timer_resolution",
    # Metrics
    "RateMeter", "PerformanceTracker",
    # Helpers
    "is_youtube_url", "is_http_url", "resolve_local_path", "truthy",
    "compute_spacing_and_group"
]