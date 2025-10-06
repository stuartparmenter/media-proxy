# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Streaming module for media-proxy.

This module handles streaming operations including:
- Stream options configuration and validation
- Media source resolution and frame processing
- Output protocol coordination
"""

from .core import create_streaming_task, stream_frames
from .options import StreamOptions


__all__ = ["StreamOptions", "create_streaming_task", "stream_frames"]
