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
from .protocol import FrameIterator, FrameIteratorConfig
from .processing import resize_pad_to_rgb_bytes

MIN_DELAY_MS = 10.0  # clamp very small frame delays

# Image size limits
MAX_SIZE_LIMIT = 50 * 1024 * 1024    # 50MB - reject larger images

def _format_size_mb(size_bytes: int) -> str:
    """Format size in bytes as MB string."""
    return f"{size_bytes / 1024 / 1024:.1f}MB"

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
                        raise ValueError(f"Image too large: {_format_size_mb(size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})")

                data = response.read()
        else:
            # Local file
            file_size = os.path.getsize(src_url)
            if file_size > MAX_SIZE_LIMIT:
                raise ValueError(f"Image too large: {_format_size_mb(file_size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})")

            with open(src_url, 'rb') as f:
                data = f.read()

        # Check actual size after download
        actual_size = len(data)
        if actual_size > MAX_SIZE_LIMIT:
            raise ValueError(f"Image too large: {_format_size_mb(actual_size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})")

        # Always use temp file - let OS/Docker handle optimization (tmpfs, caching, etc.)
        with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name

        return TempFileSource(temp_path)

    except Exception as e:
        raise RuntimeError(f"Failed to create image source: {e}") from e




class PilFrameIterator(FrameIterator):
    """Frame iterator for static images and animated GIFs using PIL."""

    def __init__(self, src_url: str, config: FrameIteratorConfig):
        super().__init__(src_url, config)
        self.img_source = None

    @classmethod
    def can_handle(cls, src_url: str) -> bool:
        """Check if this is an image file we can handle."""
        try:
            # Quick check based on URL/path extension
            lower_url = src_url.lower()
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            return any(lower_url.endswith(ext) for ext in image_extensions)
        except Exception:
            return False

    def __iter__(self) -> Iterator[Tuple[bytes, float]]:
        """Iterate frames from static images or animated GIFs."""
        config = Config()
        size = self.config.size
        loop_video = self.config.loop_video

        # Default fallback delay (for images without timing info)
        default_delay_ms = 100.0  # 10 FPS

        # Create optimized image source (memory vs temp file)
        self.img_source = _create_image_source(self.src_url)

        try:
            # Open image once and keep it open for the iteration
            pil_img = self.img_source.open_image()
            try:
                is_animated = getattr(pil_img, 'is_animated', False)
                n_frames = getattr(pil_img, 'n_frames', 1) if is_animated else 1

                # Main iteration loop
                while True:
                    saw_frame = False

                    if is_animated:
                        # Iterate through all frames - collect timing during iteration
                        for frame_idx in range(n_frames):
                            pil_img.seek(frame_idx)
                            pil_img.load()  # REQUIRED: Pillow only extracts WebP duration after load(), not seek()

                            # Get timing info from current frame
                            duration = pil_img.info.get('duration', default_delay_ms)
                            duration = max(MIN_DELAY_MS, float(duration))

                            frame_img = pil_img.convert("RGB")
                            rgb888 = resize_pad_to_rgb_bytes(frame_img, size, config)

                            yield rgb888, duration
                            saw_frame = True
                    else:
                        # Static image
                        frame_img = pil_img.convert("RGB")
                        rgb888 = resize_pad_to_rgb_bytes(frame_img, size, config)
                        yield rgb888, default_delay_ms
                        saw_frame = True

                    if not loop_video:
                        break
                    if not saw_frame:
                        raise FileNotFoundError(f"cannot decode frames: {self.src_url}")
            finally:
                # Ensure PIL image is closed
                pil_img.close()

        except Exception as e:
            msg = str(e).lower()
            if isinstance(e, FileNotFoundError) or "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(f"cannot open source: {self.src_url}") from e
            raise RuntimeError(f"PIL error: {e}") from e
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Clean up any resources used by the iterator."""
        if self.img_source:
            self.img_source.cleanup()
            self.img_source = None