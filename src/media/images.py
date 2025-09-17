# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import numpy as np
from PIL import Image
from typing import Iterator, Tuple
import urllib.request
from io import BytesIO
import tempfile
import os
import weakref
import atexit
from abc import ABC, abstractmethod

from ..config import Config
from ..utils.helpers import is_http_url

MIN_DELAY_MS = 10.0  # clamp very small frame delays

# Image size limits
MAX_SIZE_LIMIT = 50 * 1024 * 1024    # 50MB - reject larger images

# Global registry for cleanup tracking
_active_image_sources = weakref.WeakSet()

def _cleanup_all_sources():
    """Emergency cleanup function called on exit."""
    for source in list(_active_image_sources):
        try:
            source.cleanup()
        except Exception:
            pass

# Register emergency cleanup
atexit.register(_cleanup_all_sources)

def cleanup_active_image_sources():
    """Clean up all active image sources. Called when streams are stopped."""
    active_count = len(_active_image_sources)
    if active_count > 0:
        print(f"[cache] cleaning up {active_count} active image sources")
        _cleanup_all_sources()
        return active_count
    return 0


class ImageSource(ABC):
    """Abstract base for image data sources."""

    @abstractmethod
    def open_image(self) -> Image.Image:
        """Open and return PIL Image object."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources."""
        pass


# Removed BytesIOSource - using temp files for all images for simplicity


class TempFileSource(ImageSource):
    """Image source that uses a temporary file."""

    def __init__(self, temp_path: str):
        self.temp_path = temp_path
        self._cleaned = False
        # Register for cleanup tracking
        _active_image_sources.add(self)

    def open_image(self) -> Image.Image:
        if self._cleaned:
            raise RuntimeError("Attempted to use cleaned up ImageSource")
        return Image.open(self.temp_path)

    def cleanup(self) -> None:
        if self._cleaned:
            return

        try:
            if os.path.exists(self.temp_path):
                os.unlink(self.temp_path)
                print(f"[cache] cleaned up temp file: {os.path.basename(self.temp_path)}")
        except Exception as e:
            print(f"[cache] temp file cleanup warning: {e}")
        finally:
            self._cleaned = True
            # Remove from tracking (weakref will handle this automatically too)
            try:
                _active_image_sources.discard(self)
            except:
                pass


def _create_image_source(src_url: str) -> ImageSource:
    """Create temp file image source (simplified approach)."""
    try:
        if is_http_url(src_url):
            with urllib.request.urlopen(src_url) as response:
                # Check content length if available
                content_length = response.headers.get('Content-Length')
                if content_length:
                    size = int(content_length)
                    if size > MAX_SIZE_LIMIT:
                        raise ValueError(f"Image too large: {size / 1024 / 1024:.1f}MB (max {MAX_SIZE_LIMIT / 1024 / 1024:.1f}MB)")

                data = response.read()
        else:
            # Local file
            file_size = os.path.getsize(src_url)
            if file_size > MAX_SIZE_LIMIT:
                raise ValueError(f"Image too large: {file_size / 1024 / 1024:.1f}MB (max {MAX_SIZE_LIMIT / 1024 / 1024:.1f}MB)")

            with open(src_url, 'rb') as f:
                data = f.read()

        # Check actual size after download
        actual_size = len(data)
        if actual_size > MAX_SIZE_LIMIT:
            raise ValueError(f"Image too large: {actual_size / 1024 / 1024:.1f}MB (max {MAX_SIZE_LIMIT / 1024 / 1024:.1f}MB)")

        # Always use temp file - let OS/Docker handle optimization (tmpfs, caching, etc.)
        with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        return TempFileSource(temp_path)

    except Exception as e:
        raise RuntimeError(f"Failed to create image source: {e}") from e


def iter_frames_pil(src_url: str, size: Tuple[int, int], loop_video: bool) -> Iterator[Tuple[bytes, float]]:
    """Iterate frames from static images or animated GIFs using PIL directly."""
    config = Config()

    # Default fallback delay (for images without timing info)
    default_delay_ms = 1000.0 / 10.0

    # Create optimized image source (memory vs temp file)
    img_source = _create_image_source(src_url)

    try:
        from .processing import resize_pad_to_rgb_bytes

        # Determine if animated and get frame info
        with img_source.open_image() as pil_img:
            is_animated = getattr(pil_img, 'is_animated', False)
            n_frames = getattr(pil_img, 'n_frames', 1) if is_animated else 1

            # Pre-load frame timings for animated GIFs
            frame_timings = []
            if is_animated:
                for i in range(n_frames):
                    pil_img.seek(i)
                    duration = pil_img.info.get('duration', default_delay_ms)
                    frame_timings.append(max(MIN_DELAY_MS, float(duration)))

                if frame_timings and not hasattr(iter_frames_pil, '_logged_frame_timing'):
                    print(f"[gif] loaded per-frame timing: {len(frame_timings)} frames")
                    iter_frames_pil._logged_frame_timing = True

        # Main iteration loop
        while True:
            saw_frame = False

            if is_animated:
                # Iterate through all frames
                for frame_idx in range(n_frames):
                    with img_source.open_image() as pil_img:
                        pil_img.seek(frame_idx)
                        frame_img = pil_img.convert("RGB")
                        rgb888 = resize_pad_to_rgb_bytes(frame_img, size, config)

                        # Use pre-loaded timing
                        delay_ms = frame_timings[frame_idx] if frame_timings else default_delay_ms
                        final_delay = max(MIN_DELAY_MS, float(delay_ms))

                        yield rgb888, final_delay
                        saw_frame = True
            else:
                # Static image
                with img_source.open_image() as pil_img:
                    frame_img = pil_img.convert("RGB")
                    rgb888 = resize_pad_to_rgb_bytes(frame_img, size, config)
                    yield rgb888, default_delay_ms
                    saw_frame = True

            if not loop_video:
                break
            if not saw_frame:
                raise FileNotFoundError(f"cannot decode frames: {src_url}")

    except Exception as e2:
        msg = str(e2).lower()
        if isinstance(e2, FileNotFoundError) or "no such file" in msg or "not found" in msg:
            raise FileNotFoundError(f"cannot open source: {src_url}") from e2
        raise RuntimeError(f"PIL error: {e2}") from e2
    finally:
        # Always cleanup resources
        img_source.cleanup()