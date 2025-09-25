# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Streaming module for media-proxy.

This module handles streaming operations including:
- Stream options configuration and validation
- Media source resolution and frame processing
- Output protocol coordination
"""

from .options import StreamOptions
from .core import create_streaming_task, stream_frames

__all__ = [
    "StreamOptions",
    "create_streaming_task",
    "stream_frames"
]