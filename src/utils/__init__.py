# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Utility modules for hardware detection, metrics, and helpers."""

from .hardware import pick_hw_backend, set_windows_timer_resolution
from .helpers import compute_spacing_and_group, is_http_url, is_youtube_url, resolve_local_path
from .metrics import PerformanceTracker, RateMeter


__all__ = [
    "PerformanceTracker",
    # Metrics
    "RateMeter",
    "compute_spacing_and_group",
    "is_http_url",
    # Helpers
    "is_youtube_url",
    # Hardware
    "pick_hw_backend",
    "resolve_local_path",
    "set_windows_timer_resolution",
]
