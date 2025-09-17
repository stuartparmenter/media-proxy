# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Media handling modules for sources, video, images, and processing."""

from .sources import resolve_media_source, MediaSource, StreamUrlExpiredError
from .images import iter_frames_pil
from .video import iter_frames_pyav
from .processing import resize_pad_to_rgb_bytes, rgb888_to_565_bytes

__all__ = [
    # Sources
    "resolve_media_source", "MediaSource", "StreamUrlExpiredError",
    # Frame iteration
    "iter_frames_pil", "iter_frames_pyav",
    # Processing
    "resize_pad_to_rgb_bytes", "rgb888_to_565_bytes"
]