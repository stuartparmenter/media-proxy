# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import numpy as np
from PIL import Image
from typing import Iterator, Tuple
import imageio.v3 as iio

from ..config import Config

MIN_DELAY_MS = 10.0  # clamp very small frame delays


def iter_frames_imageio(src_url: str, size: Tuple[int, int], loop_video: bool, fps_override: float = None) -> Iterator[Tuple[bytes, float]]:
    """Iterate frames from static images or animated GIFs using imageio."""
    config = Config()
    
    # Try PIL directly first for GIF timing (most reliable)
    default_delay_ms = 1000.0 / 10.0

    if src_url.lower().endswith(".gif"):
        try:
            from PIL import Image
            from ..utils.helpers import is_http_url

            if is_http_url(src_url):
                # For URLs, download to a BytesIO buffer
                import urllib.request
                from io import BytesIO

                with urllib.request.urlopen(src_url) as response:
                    img_data = BytesIO(response.read())
                    with Image.open(img_data) as pil_img:
                        if pil_img.is_animated and 'duration' in pil_img.info:
                            duration_ms = pil_img.info['duration']
                            default_delay_ms = max(MIN_DELAY_MS, float(duration_ms))
                            print(f"[gif] using PIL duration={duration_ms}ms (from URL)")
            else:
                # Local file path
                with Image.open(src_url) as pil_img:
                    if pil_img.is_animated and 'duration' in pil_img.info:
                        duration_ms = pil_img.info['duration']
                        default_delay_ms = max(MIN_DELAY_MS, float(duration_ms))
                        print(f"[gif] using PIL duration={duration_ms}ms (local file)")

        except Exception as e:
            print(f"[gif] PIL detection failed: {e!r}")

    # Fallback to imageio FPS detection
    if default_delay_ms == 1000.0 / 10.0:
        try:
            props = iio.improps(src_url)
            fps = getattr(props, "fps", None)
            if fps and fps > 0:
                default_delay_ms = max(MIN_DELAY_MS, 1000.0 / float(fps))
                print(f"[gif] using imageio fps={fps}, delay={default_delay_ms:.1f}ms")
        except Exception:
            pass

    # Final fallback to PyAV FPS
    if default_delay_ms == 1000.0 / 10.0 and fps_override and fps_override > 0:
        default_delay_ms = max(MIN_DELAY_MS, 1000.0 / float(fps_override))
        print(f"[gif] using PyAV fps={fps_override}, delay={default_delay_ms:.1f}ms")

    plugin = "pillow" if src_url.lower().endswith(".gif") else None

    while True:
        try:
            reader = iio.imiter(src_url, plugin=plugin) if plugin else iio.imiter(src_url)
            saw_frame = False
            
            for frame in reader:
                saw_frame = True

                im = Image.fromarray(np.asarray(frame)).convert("RGB")
                from .processing import resize_pad_to_rgb_bytes
                rgb888 = resize_pad_to_rgb_bytes(im, size, config)
                
                # Start with default delay, try to get per-frame timing
                delay_ms = default_delay_ms
                frame_has_timing = False

                # Try imageio frame metadata first
                try:
                    if hasattr(frame, "meta"):
                        d = frame.meta.get("duration")
                        if d is not None:
                            if isinstance(d, (int, float)):
                                delay_ms = float(d) * (1000.0 if float(d) <= 10.0 else 1.0)
                                frame_has_timing = True
                                if saw_frame and not hasattr(iter_frames_imageio, '_logged_frame_timing'):
                                    print(f"[gif] found imageio per-frame timing: {delay_ms:.1f}ms")
                                    iter_frames_imageio._logged_frame_timing = True
                except Exception:
                    pass

                # If imageio has no timing, try to get per-frame timing from cached PIL data
                if not frame_has_timing and src_url.lower().endswith(".gif"):
                    # Use a function-level cache to avoid re-downloading
                    if not hasattr(iter_frames_imageio, '_pil_timing_cache'):
                        iter_frames_imageio._pil_timing_cache = {}

                    if src_url not in iter_frames_imageio._pil_timing_cache:
                        # Load timing data once per GIF
                        try:
                            from PIL import Image
                            from ..utils.helpers import is_http_url

                            timing_data = []
                            if is_http_url(src_url):
                                import urllib.request
                                from io import BytesIO

                                with urllib.request.urlopen(src_url) as response:
                                    img_data = BytesIO(response.read())
                                    with Image.open(img_data) as pil_img:
                                        if pil_img.is_animated:
                                            for i in range(pil_img.n_frames):
                                                pil_img.seek(i)
                                                duration = pil_img.info.get('duration', default_delay_ms)
                                                timing_data.append(max(MIN_DELAY_MS, float(duration)))
                            else:
                                # Local file
                                with Image.open(src_url) as pil_img:
                                    if pil_img.is_animated:
                                        for i in range(pil_img.n_frames):
                                            pil_img.seek(i)
                                            duration = pil_img.info.get('duration', default_delay_ms)
                                            timing_data.append(max(MIN_DELAY_MS, float(duration)))

                            iter_frames_imageio._pil_timing_cache[src_url] = timing_data
                            if timing_data:
                                print(f"[gif] cached PIL per-frame timing: {len(timing_data)} frames")

                        except Exception as e:
                            iter_frames_imageio._pil_timing_cache[src_url] = []

                    # Use cached timing data for this frame
                    cached_timing = iter_frames_imageio._pil_timing_cache.get(src_url, [])
                    frame_idx = (saw_frame - 1) % len(cached_timing) if cached_timing else -1
                    if frame_idx >= 0:
                        delay_ms = cached_timing[frame_idx]
                        frame_has_timing = True

                # Log if we're using default timing
                if saw_frame and not frame_has_timing and not hasattr(iter_frames_imageio, '_logged_default_timing'):
                    print(f"[gif] using default timing: {delay_ms:.1f}ms per frame")
                    iter_frames_imageio._logged_default_timing = True

                final_delay = max(MIN_DELAY_MS, float(delay_ms))
                yield rgb888, final_delay
                
            if not loop_video:
                break
            if not saw_frame:
                raise FileNotFoundError(f"cannot decode frames: {src_url}")
                
        except Exception as e2:
            msg = str(e2).lower()
            if isinstance(e2, FileNotFoundError) or "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(f"cannot open source: {src_url}") from e2
            raise RuntimeError(f"imageio error: {e2}") from e2
            
        if not loop_video:
            break