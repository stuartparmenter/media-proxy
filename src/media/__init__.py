# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Media handling modules for sources, video, images, and processing."""

from .sources import resolve_media_source, MediaSource, StreamUrlExpiredError
# Protocol-based frame iteration
from .protocol import FrameIterator, FrameIteratorConfig, FrameIteratorFactory
from .images import PilFrameIterator
from .video import PyAvFrameIterator
from .processing import resize_pad_to_rgb_bytes, rgb888_to_565_bytes

__all__ = [
    # Sources
    "resolve_media_source", "MediaSource", "StreamUrlExpiredError",
    # Frame iteration (protocol-based)
    "FrameIterator", "FrameIteratorConfig", "FrameIteratorFactory",
    "PilFrameIterator", "PyAvFrameIterator",
    # Processing
    "resize_pad_to_rgb_bytes", "rgb888_to_565_bytes"
]