# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Media handling modules for sources, video, images, and processing."""

from .images import PilFrameIterator
from .processing import resize_pad_to_rgb_bytes, rgb888_to_565_bytes

# Protocol-based frame iteration
from .protocol import FrameIterator, FrameIteratorFactory
from .sources import MediaSource, MediaUnavailableError, resolve_media_source
from .video import PyAvFrameIterator


__all__ = [
    # Frame iteration (protocol-based)
    "FrameIterator",
    "FrameIteratorFactory",
    "MediaSource",
    "MediaUnavailableError",
    "PilFrameIterator",
    "PyAvFrameIterator",
    # Processing
    "resize_pad_to_rgb_bytes",
    # Sources
    "resolve_media_source",
    "rgb888_to_565_bytes",
]
