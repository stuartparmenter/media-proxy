# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Media handling modules for sources, video, images, and processing."""

# Media-layer exceptions (clean abstraction from library-specific errors)
from .exceptions import (
    MediaDecodeError,
    MediaFormatError,
    MediaNetworkError,
    MediaNotFoundError,
    MediaSourceError,
)
from .images import PilFrameIterator
from .processing import resize_pad_to_rgb_bytes, rgb888_to_565_bytes

# Protocol-based frame iteration
from .protocol import FrameIterator, FrameIteratorFactory
from .sources import MediaSource, MediaUnavailableError, resolve_media_source
from .video import PyAvFrameIterator


__all__ = [
    "FrameIterator",
    "FrameIteratorFactory",
    "MediaDecodeError",
    "MediaFormatError",
    "MediaNetworkError",
    "MediaNotFoundError",
    "MediaSource",
    "MediaSourceError",
    "MediaUnavailableError",
    "PilFrameIterator",
    "PyAvFrameIterator",
    "resize_pad_to_rgb_bytes",
    "resolve_media_source",
    "rgb888_to_565_bytes",
]
