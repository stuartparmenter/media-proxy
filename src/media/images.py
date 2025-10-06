# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import atexit
import contextlib
import hashlib
import logging
import tempfile
import urllib.error
import urllib.request
import weakref
from abc import ABC, abstractmethod
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import numpy as np
from PIL import Image

from ..config import Config
from ..utils.helpers import resolve_local_path
from .processing import resize_pad_to_rgb_bytes
from .protocol import FrameIterator


MIN_DELAY_MS = 10.0  # clamp very small frame delays

# Image size limits
MAX_SIZE_LIMIT = 50 * 1024 * 1024  # 50MB - reject larger images
MEMORY_THRESHOLD = 500 * 1024  # 500KB - use BytesIO for smaller images


# Normalized disposal method constants
class DisposalMethod:
    """Normalized disposal methods for animated images."""

    NONE = 0  # Do not dispose - leave frame for next to composite over
    BACKGROUND = 1  # Restore to background color before next frame
    PREVIOUS = 2  # Restore to previous frame state before next frame


# Normalized blend mode constants
class BlendMode:
    """Normalized blend modes for animated images."""

    OVER = 0  # Alpha blend over existing pixels (default)
    SOURCE = 1  # Replace pixels entirely (no alpha blending)


def _format_size_mb(size_bytes: int) -> str:
    """Format size in bytes as MB string."""
    return f"{size_bytes / 1024 / 1024:.1f}MB"


def _get_normalized_disposal_method(pil_img: Image.Image) -> int:
    """Get normalized disposal method from PIL image.

    Returns DisposalMethod constant regardless of format (GIF/APNG/WebP).
    """
    # Try GIF format first (has disposal_method attribute)
    if hasattr(pil_img, "disposal_method"):
        gif_disposal = pil_img.disposal_method  # type: ignore[attr-defined]
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
    elif "disposal" in pil_img.info:
        apng_disposal = pil_img.info["disposal"]
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
    if hasattr(pil_img, "blend_op"):
        apng_blend = pil_img.blend_op  # type: ignore[attr-defined]
        # APNG: 0=source, 1=over → reverse to match our constants
        if apng_blend == 0:
            return BlendMode.SOURCE
        else:
            return BlendMode.OVER

    # Default for GIF, WebP, or unknown formats (always alpha blend)
    return BlendMode.OVER


# Global registry for cleanup tracking
_active_image_sources: weakref.WeakSet[Any] = weakref.WeakSet()


def _cleanup_all_sources():
    """Emergency cleanup function called on exit."""
    for source in list(_active_image_sources):
        with contextlib.suppress(Exception):  # atexit cleanup - logging/raising would be problematic
            source.cleanup()


# Register emergency cleanup
atexit.register(_cleanup_all_sources)


def cleanup_active_image_sources():
    """Clean up all active image sources. Called when streams are stopped."""
    active_count = len(_active_image_sources)
    if active_count > 0:
        logging.getLogger("images").info(f"cleaning up {active_count} active image sources")
        _cleanup_all_sources()
    return active_count


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
        self._data: bytes = data
        self._cleaned = False
        # Register for cleanup tracking
        _active_image_sources.add(self)

    def open_image(self) -> Image.Image:
        if self._cleaned:
            raise RuntimeError("Attempted to use cleaned up ImageSource")
        return Image.open(BytesIO(self._data))

    def cleanup(self) -> None:
        if self._cleaned:
            return
        del self._data
        self._cleaned = True
        # Remove from tracking (weakref will handle this automatically too)
        with contextlib.suppress(BaseException):  # Destructor cleanup - must not raise
            _active_image_sources.discard(self)


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
            temp_path_obj = Path(self.temp_path)
            if temp_path_obj.exists():
                temp_path_obj.unlink()
                logging.getLogger("images").debug(f"cleaned up temp file: {temp_path_obj.name}")
        except Exception as e:
            logging.getLogger("images").warning(f"temp file cleanup warning: {e}")
        finally:
            self._cleaned = True
            # Remove from tracking (weakref will handle this automatically too)
            with contextlib.suppress(BaseException):  # Destructor cleanup - must not raise
                _active_image_sources.discard(self)


class LocalFileSource(ImageSource):
    """Image source that directly uses a local file without copying."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._cleaned = False
        # Register for cleanup tracking
        _active_image_sources.add(self)

    def open_image(self) -> Image.Image:
        if self._cleaned:
            raise RuntimeError("Attempted to use cleaned up ImageSource")
        return Image.open(self.file_path)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        # Remove from tracking (weakref will handle this automatically too)
        with contextlib.suppress(BaseException):  # Destructor cleanup - must not raise
            _active_image_sources.discard(self)


async def _create_image_source(src_url: str) -> ImageSource:
    """Create image source using async HTTP fetching (aiohttp) to avoid blocking event loop."""
    try:
        # For file:// URLs, use local file directly (no copying needed)
        local_path = resolve_local_path(src_url)
        if local_path:
            # Check file size limit
            file_size = Path(local_path).stat().st_size
            if file_size > MAX_SIZE_LIMIT:
                raise ValueError(
                    f"Image too large: {_format_size_mb(file_size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})"
                )

            return LocalFileSource(local_path)

        # For HTTP/HTTPS URLs, fetch with aiohttp (non-blocking)
        async with aiohttp.ClientSession() as session:
            headers = {"User-Agent": Config().get("net.user_agent")}
            timeout = aiohttp.ClientTimeout(total=30)

            async with session.get(src_url, headers=headers, timeout=timeout) as response:
                response.raise_for_status()

                # Check content length if available
                content_length = response.headers.get("Content-Length")
                if content_length:
                    size = int(content_length)
                    if size > MAX_SIZE_LIMIT:
                        raise ValueError(
                            f"Image too large: {_format_size_mb(size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})"
                        )

                # Read data asynchronously
                data = await response.read()

        # Check actual size after download
        actual_size = len(data)
        if actual_size > MAX_SIZE_LIMIT:
            raise ValueError(f"Image too large: {_format_size_mb(actual_size)} (max {_format_size_mb(MAX_SIZE_LIMIT)})")

        # Use BytesIO for small images, temp file for large ones
        if actual_size <= MEMORY_THRESHOLD:
            return BytesIOSource(data)

        # Use temp file for larger downloaded images
        with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as temp_file:
            temp_file.write(data)
            temp_path = temp_file.name
        return TempFileSource(temp_path)

    except aiohttp.ClientError as e:
        # Convert to URLError for compatibility with existing error handling
        raise urllib.error.URLError(str(e)) from e
    except FileNotFoundError:
        # Re-raise file not found for upstream handling
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to create image source: {e}") from e


def _get_display_name(src_url: str) -> str:
    """Get a display name for a URL for logging."""
    try:
        parsed = urlparse(src_url)
        path = parsed.path or "/"
        filename = path.split("/")[-1]
        return filename or parsed.netloc or src_url
    except Exception:
        return src_url[-50:] if len(src_url) > 50 else src_url


class PilFrameIterator(FrameIterator):
    """Frame iterator for static images and animated GIFs using PIL."""

    def __init__(self, src_url: str, stream_options):
        super().__init__(src_url, stream_options)
        self._img_source: ImageSource | None = None
        self._frame_cache: dict[str, list[tuple[bytes, float]]] = {}
        self._cache_memory_usage = 0
        self._cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._first_loop_complete = False

    async def async_init(self):
        """Async initialization to fetch image source without blocking event loop."""
        self._img_source = await _create_image_source(self.src_url)

    @classmethod
    def can_handle(cls, src_url: str) -> bool:
        """Check if this is an image file we can handle."""
        try:
            from urllib.parse import urlparse

            # Parse URL to get path without query parameters
            parsed = urlparse(src_url.lower())
            path = parsed.path if parsed.path else src_url.lower()

            image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
            return any(path.endswith(ext) for ext in image_extensions)
        except Exception:
            return False

    def _get_cache_key(self) -> str:
        """Generate cache key based on source URL and stream options."""
        options_hash = hashlib.md5()  # noqa: S324  # Not used for security, just cache deduplication
        options_hash.update(self.src_url.encode())
        options_hash.update(str(self.stream_options.__dict__).encode())
        return f"cache_{options_hash.hexdigest()[:16]}"

    def _should_cache_frames(self, n_frames: int, loop_video: bool) -> bool:
        """Determine if frames should be cached based on config and animation properties."""
        if not loop_video:
            return False

        config = Config()
        cache_mb = config.get("image.frame_cache_mb")
        if cache_mb <= 0:
            return False

        min_frames = config.get("image.frame_cache_min_frames")
        return n_frames >= min_frames

    def _estimate_frame_size(self, size: tuple[int, int]) -> int:
        """Estimate memory usage per frame in bytes (RGB format)."""
        width, height = size
        return width * height * 3  # RGB = 3 bytes per pixel

    def _evict_cache_if_needed(self, estimated_new_size: int) -> None:
        """Evict cache entries if adding new frames would exceed memory limit."""
        cache_limit_bytes = Config().get("image.frame_cache_mb") * 1024 * 1024

        if self._cache_memory_usage + estimated_new_size <= cache_limit_bytes:
            return

        # Simple LRU: remove all entries for this source if we would exceed limit
        # In a more sophisticated implementation, we could track access times
        cache_key = self._get_cache_key()
        if cache_key in self._frame_cache:
            old_frames = self._frame_cache[cache_key]
            old_size = len(old_frames) * self._estimate_frame_size(self.stream_options.size)
            del self._frame_cache[cache_key]
            self._cache_memory_usage -= old_size
            self._cache_stats["evictions"] += 1
            logging.getLogger("images").debug(f"evicted cache entry for {self.src_url} (saved {old_size} bytes)")

    def __iter__(self) -> Iterator[tuple[bytes, float]]:
        """Iterate frames - image source must be initialized via async_init() first."""
        assert self._img_source is not None, "Image source not initialized. Call async_init() before iteration."

        size = self.stream_options.size
        loop_video = self.stream_options.loop

        # Default fallback delay (for images without timing info)
        default_delay_ms = 100.0  # 10 FPS

        # Use pre-fetched image source
        img_source = self._img_source  # Local reference for type narrowing

        try:
            # Open image once and keep it open for the iteration
            pil_img = img_source.open_image()
            try:
                is_animated = getattr(pil_img, "is_animated", False)
                n_frames = getattr(pil_img, "n_frames", 1) if is_animated else 1

                # Extract palette and background color once for animated images (same for all frames)
                background_color = None
                global_palette = None
                bg_rgb = (0, 0, 0)  # Default black background for LEDs

                if is_animated:
                    background_color = pil_img.info.get("background", 0)  # Background color index
                    if hasattr(pil_img, "palette") and pil_img.palette:
                        try:
                            palette_data = pil_img.palette.getdata()[1]
                            if len(palette_data) >= 3:
                                palette_rgb = np.frombuffer(palette_data, dtype=np.uint8)  # type: ignore[call-overload]  # numpy stubs limitation with bytes/buffer input
                                if len(palette_rgb) % 3 == 0:
                                    global_palette = palette_rgb.reshape(-1, 3)
                        except Exception:  # noqa: S110  # GIF palette parsing fallback
                            pass

                    # Determine background color (default to black for LEDs)
                    if global_palette is not None and background_color < len(global_palette):
                        bg_rgb = tuple(global_palette[background_color])

                # Cache setup
                cache_key = self._get_cache_key()
                should_cache = self._should_cache_frames(n_frames, loop_video)

                # Main iteration loop
                loop_count = 0
                while True:
                    loop_count += 1

                    # Check cache on each loop iteration
                    cached_frames = None
                    if should_cache and cache_key in self._frame_cache:
                        cached_frames = self._frame_cache[cache_key]
                        self._cache_stats["hits"] += 1

                    saw_frame = False
                    current_loop_frames = []  # Collect frames for caching on first loop

                    if is_animated:
                        if cached_frames:
                            # Serve from cache
                            for frame_data, delay in cached_frames:
                                yield frame_data, delay
                                saw_frame = True

                            if not loop_video:
                                break
                            if not saw_frame:
                                raise FileNotFoundError(f"cannot decode frames: {self.src_url}")
                            continue  # Skip to next loop iteration

                        # Process frames normally and cache if needed
                        canvas = None  # Current composite state
                        previous_canvas = None  # For disposal method 3
                        self._cache_stats["misses"] += 1

                        # Iterate through all frames with proper disposal handling
                        for frame_idx in range(n_frames):
                            pil_img.seek(frame_idx)
                            pil_img.load()  # REQUIRED: Pillow only extracts WebP duration after load(), not seek()

                            # Get timing info from current frame
                            duration = pil_img.info.get("duration", default_delay_ms)
                            duration = max(MIN_DELAY_MS, float(duration))

                            # Get normalized disposal method and blend mode
                            disposal_method = _get_normalized_disposal_method(pil_img)
                            blend_mode = _get_normalized_blend_mode(pil_img)

                            # Initialize canvas on first frame
                            if canvas is None:
                                canvas = Image.new("RGBA", pil_img.size, (*bg_rgb, 255))

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
                            rgb888 = resize_pad_to_rgb_bytes(canvas, size, self.stream_options.fit)

                            # Cache frame if we're on first loop and caching is enabled
                            if should_cache and not self._first_loop_complete:
                                current_loop_frames.append((rgb888, duration))

                            yield rgb888, duration
                            saw_frame = True

                            # Apply disposal method for next frame
                            if disposal_method == DisposalMethod.BACKGROUND:
                                # Restore to background color
                                canvas = Image.new("RGBA", pil_img.size, (*bg_rgb, 255))
                            elif disposal_method == DisposalMethod.PREVIOUS:
                                # Restore to previous frame state
                                if previous_canvas is not None:
                                    canvas = previous_canvas.copy()
                            # For DisposalMethod.NONE: do nothing (leave canvas as-is)

                            # Cache frames after first complete loop
                            if should_cache and not self._first_loop_complete and len(current_loop_frames) == n_frames:
                                estimated_size = len(current_loop_frames) * self._estimate_frame_size(size)
                                self._evict_cache_if_needed(estimated_size)

                                self._frame_cache[cache_key] = current_loop_frames.copy()
                                self._cache_memory_usage += estimated_size
                                self._first_loop_complete = True

                                logging.getLogger("images").info(
                                    f"cached {len(current_loop_frames)} frames for {_get_display_name(self.src_url)} "
                                    f"(~{estimated_size // 1024}KB, total cache: {self._cache_memory_usage // 1024}KB)"
                                )
                    else:
                        # Static image - pass as-is to handle transparency
                        rgb888 = resize_pad_to_rgb_bytes(pil_img, size, self.stream_options.fit)
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
        if self._img_source:
            self._img_source.cleanup()
            self._img_source = None

        # Log cache statistics
        stats = self._cache_stats
        if any(stats.values()):
            logging.getLogger("images").debug(
                f"cache stats for {_get_display_name(self.src_url)}: "
                f"hits={stats['hits']} misses={stats['misses']} evictions={stats['evictions']}"
            )
