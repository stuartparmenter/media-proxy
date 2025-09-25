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
import logging
from abc import ABC, abstractmethod

from ..config import Config
from ..utils.helpers import is_http_url
from .protocol import FrameIterator
from .processing import resize_pad_to_rgb_bytes

MIN_DELAY_MS = 10.0  # clamp very small frame delays

# Image size limits
MAX_SIZE_LIMIT = 50 * 1024 * 1024    # 50MB - reject larger images
MEMORY_THRESHOLD = 500 * 1024        # 500KB - use BytesIO for smaller images

# Normalized disposal method constants
class DisposalMethod:
    """Normalized disposal methods for animated images."""
    NONE = 0        # Do not dispose - leave frame for next to composite over
    BACKGROUND = 1  # Restore to background color before next frame
    PREVIOUS = 2    # Restore to previous frame state before next frame

# Normalized blend mode constants
class BlendMode:
    """Normalized blend modes for animated images."""
    OVER = 0        # Alpha blend over existing pixels (default)
    SOURCE = 1      # Replace pixels entirely (no alpha blending)

def _format_size_mb(size_bytes: int) -> str:
    """Format size in bytes as MB string."""
    return f"{size_bytes / 1024 / 1024:.1f}MB"


def _get_normalized_disposal_method(pil_img: Image.Image) -> int:
    """Get normalized disposal method from PIL image.

    Returns DisposalMethod constant regardless of format (GIF/APNG/WebP).
    """
    # Try GIF format first (has disposal_method attribute)
    if hasattr(pil_img, 'disposal_method'):
        gif_disposal = pil_img.disposal_method
        # GIF: 0/1=none, 2=background, 3=previous → normalize to 0,1,2
        if gif_disposal in (0, 1):
            return DisposalMethod.NONE
        elif gif_disposal == 2:
            return DisposalMethod.BACKGROUND
        elif gif_disposal == 3:
            return DisposalMethod.PREVIOUS
        else:
            return DisposalMethod.NONE  # fallback

    # Try APNG format (uses info['disposal'])
    elif 'disposal' in pil_img.info:
        apng_disposal = pil_img.info['disposal']
        # APNG: 0=none, 1=background, 2=previous → already normalized
        if apng_disposal == 0:
            return DisposalMethod.NONE
        elif apng_disposal == 1:
            return DisposalMethod.BACKGROUND
        elif apng_disposal == 2:
            return DisposalMethod.PREVIOUS
        else:
            return DisposalMethod.NONE  # fallback

    # Default for WebP or unknown formats
    return DisposalMethod.NONE


def _get_normalized_blend_mode(pil_img: Image.Image) -> int:
    """Get normalized blend mode from PIL image.

    Returns BlendMode constant regardless of format.
    """
    # Try APNG format (has blend_op attribute)
    if hasattr(pil_img, 'blend_op'):
        apng_blend = pil_img.blend_op
        # APNG: 0=source, 1=over → reverse to match our constants
        if apng_blend == 0:
            return BlendMode.SOURCE
        else:
            return BlendMode.OVER

    # Default for GIF, WebP, or unknown formats (always alpha blend)
    return BlendMode.OVER

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
        logging.getLogger('images').info(f"cleaning up {active_count} active image sources")
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


class BytesIOSource(ImageSource):
    """Image source that uses in-memory BytesIO for small images."""

    def __init__(self, data: bytes):
        self.data = data
        self._cleaned = False
        # Register for cleanup tracking
        _active_image_sources.add(self)

    def open_image(self) -> Image.Image:
        if self._cleaned:
            raise RuntimeError("Attempted to use cleaned up ImageSource")
        return Image.open(BytesIO(self.data))

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self.data = None
        self._cleaned = True
        # Remove from tracking (weakref will handle this automatically too)
        try:
            _active_image_sources.discard(self)
        except:
            pass


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
                logging.getLogger('images').debug(f"cleaned up temp file: {os.path.basename(self.temp_path)}")
        except Exception as e:
            logging.getLogger('images').warning(f"temp file cleanup warning: {e}")
        finally:
            self._cleaned = True
            # Remove from tracking (weakref will handle this automatically too)
            try:
                _active_image_sources.discard(self)
            except:
                pass


def _create_image_source(src_url: str) -> ImageSource:
    """Create image source - BytesIO for small images, temp file for large ones."""
    try:
        if is_http_url(src_url):
            try:
                # Create request with proper headers to avoid 403 errors
                req = urllib.request.Request(src_url)
                req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36')
                with urllib.request.urlopen(req) as response:
                    # Check content length if available
                    content_length = response.headers.get('Content-Length')
                    if content_length:
                        size = int(content_length)
                        if size > MAX_SIZE_LIMIT:
                            raise ValueError(f"Image too large: {_format_size_mb(size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})")

                    data = response.read()
            except (urllib.error.HTTPError, urllib.error.URLError):
                # Re-raise HTTP/network errors for upstream handling
                raise
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

        # Use BytesIO for small images, temp file for large ones
        if actual_size <= MEMORY_THRESHOLD:
            return BytesIOSource(data)

        # Use temp file for larger images
        with tempfile.NamedTemporaryFile(suffix='.img', delete=False) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name
        return TempFileSource(temp_path)

    except (urllib.error.HTTPError, urllib.error.URLError, FileNotFoundError):
        # Re-raise HTTP/network errors and file not found for upstream handling
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create image source: {e}") from e




class PilFrameIterator(FrameIterator):
    """Frame iterator for static images and animated GIFs using PIL."""

    def __init__(self, src_url: str, stream_options):
        super().__init__(src_url, stream_options)
        self.img_source = None

    @classmethod
    def can_handle(cls, src_url: str) -> bool:
        """Check if this is an image file we can handle."""
        try:
            from urllib.parse import urlparse
            # Parse URL to get path without query parameters
            parsed = urlparse(src_url.lower())
            path = parsed.path if parsed.path else src_url.lower()

            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            return any(path.endswith(ext) for ext in image_extensions)
        except Exception:
            return False

    def __iter__(self) -> Iterator[Tuple[bytes, float]]:
        """Iterate frames from static images or animated GIFs with proper disposal handling."""
        config = Config()
        size = self.stream_options.size
        loop_video = self.stream_options.loop

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

                # Extract palette and background color once for animated images (same for all frames)
                background_color = None
                global_palette = None
                bg_rgb = (0, 0, 0)  # Default black background for LEDs

                if is_animated:
                    background_color = pil_img.info.get('background', 0)  # Background color index
                    if hasattr(pil_img, 'palette') and pil_img.palette:
                        try:
                            palette_data = pil_img.palette.getdata()[1]
                            if len(palette_data) >= 3:
                                palette_rgb = np.frombuffer(palette_data, dtype=np.uint8)
                                if len(palette_rgb) % 3 == 0:
                                    global_palette = palette_rgb.reshape(-1, 3)
                        except:
                            pass

                    # Determine background color (default to black for LEDs)
                    if global_palette is not None and background_color < len(global_palette):
                        bg_rgb = tuple(global_palette[background_color])

                # Main iteration loop
                while True:
                    saw_frame = False

                    if is_animated:
                        # For animated GIFs, we need to handle frame compositing properly
                        canvas = None  # Current composite state
                        previous_canvas = None  # For disposal method 3

                        # Iterate through all frames with proper disposal handling
                        for frame_idx in range(n_frames):
                            pil_img.seek(frame_idx)
                            pil_img.load()  # REQUIRED: Pillow only extracts WebP duration after load(), not seek()

                            # Get timing info from current frame
                            duration = pil_img.info.get('duration', default_delay_ms)
                            duration = max(MIN_DELAY_MS, float(duration))

                            # Get normalized disposal method and blend mode
                            disposal_method = _get_normalized_disposal_method(pil_img)
                            blend_mode = _get_normalized_blend_mode(pil_img)

                            # Initialize canvas on first frame
                            if canvas is None:
                                canvas = Image.new("RGBA", pil_img.size, bg_rgb + (255,))

                            # Save current canvas state before applying frame (needed for PREVIOUS disposal)
                            if disposal_method == DisposalMethod.PREVIOUS:
                                previous_canvas = canvas.copy()

                            # Get current frame as RGBA
                            frame = pil_img.convert("RGBA")

                            # Composite current frame onto canvas based on blend mode
                            if blend_mode == BlendMode.SOURCE:
                                # SOURCE mode: replace pixels entirely (no alpha blending)
                                canvas.paste(frame, (0, 0))
                            else:
                                # OVER mode: alpha blend over existing pixels
                                canvas.paste(frame, (0, 0), frame)

                            # Process the composite canvas for output
                            rgb888 = resize_pad_to_rgb_bytes(canvas, size, config)
                            yield rgb888, duration
                            saw_frame = True

                            # Apply disposal method for next frame
                            if disposal_method == DisposalMethod.BACKGROUND:
                                # Restore to background color
                                canvas = Image.new("RGBA", pil_img.size, bg_rgb + (255,))
                            elif disposal_method == DisposalMethod.PREVIOUS:
                                # Restore to previous frame state
                                if previous_canvas is not None:
                                    canvas = previous_canvas.copy()
                            # For DisposalMethod.NONE: do nothing (leave canvas as-is)
                    else:
                        # Static image - pass as-is to handle transparency
                        rgb888 = resize_pad_to_rgb_bytes(pil_img, size, config)
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