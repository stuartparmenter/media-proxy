# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import av
import numpy as np
from av.codec.hwaccel import HWAccel
from av.error import FFmpegError, HTTPClientError, HTTPError, HTTPServerError
from av.filter import Graph as AvFilterGraph
from av.video.frame import VideoFrame

from ..config import Config
from ..utils.helpers import resolve_local_path
from .protocol import FrameIterator


MIN_DELAY_MS = 10.0


def open_stream(src_url: str, hw_backend: str | None, options: dict[str, str] | None = None):
    """Open media container with resolved hardware acceleration backend."""
    options = options or {}

    # Convert file:// URLs to local paths for PyAV compatibility
    local_path = resolve_local_path(src_url)
    pyav_url = local_path if local_path else src_url

    # For local files, check existence first to avoid misleading hwaccel errors
    if local_path and not Path(local_path).exists():
        raise FileNotFoundError(f"cannot open media file: {src_url}")

    try:
        hwaccel = HWAccel(device_type=hw_backend) if hw_backend else None
        container = av.open(pyav_url, mode="r", hwaccel=hwaccel, options=options)
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise RuntimeError("no video stream")
        return container, vstream

    except OSError as e:
        # Re-raise OS errors with original URL for better error messages
        raise OSError(f"cannot open media file: {src_url}") from e
    except FFmpegError as e:
        # Re-raise FFmpeg errors with original URL
        raise RuntimeError(f"FFmpeg error opening {src_url}: {e}") from e


def rotation_from_stream_and_frame(vstream, frame) -> int:
    """Extract rotation metadata from stream or frame."""
    rot = getattr(frame, "rotation", None)
    if isinstance(rot, int):
        return ((rot % 360) + 360) % 360

    md = getattr(vstream, "metadata", {})
    if md:
        r = md.get("rotate")
        if r is not None:
            return ((int(str(r)) % 360) + 360) % 360

    return 0


def tb_num_den(tb) -> tuple[int, int]:
    """Extract numerator/denominator from time base."""
    if tb is None:
        return (1, 1000)
    for a, b in (("num", "den"), ("numerator", "denominator")):
        n = getattr(tb, a, None)
        d = getattr(tb, b, None)
        if n is not None and d is not None:
            return int(n), int(d)
    try:
        n, d = tb
        return int(n), int(d)
    except Exception:
        return (1, 1000)


def sar_of(obj, vstream) -> tuple[int, int]:
    """Extract sample aspect ratio from frame or stream."""
    # Try frame SAR first; fall back to codec_context SAR; else 1:1
    sar = getattr(obj, "sample_aspect_ratio", None)
    n = getattr(sar, "num", getattr(sar, "numerator", None))
    d = getattr(sar, "den", getattr(sar, "denominator", None))
    if n and d and n > 0 and d > 0:
        return int(n), int(d)

    cc = getattr(vstream, "codec_context", None)
    sar = getattr(cc, "sample_aspect_ratio", None)
    n = getattr(sar, "num", getattr(sar, "numerator", None))
    d = getattr(sar, "den", getattr(sar, "denominator", None))
    if n and d and n > 0 and d > 0:
        return int(n), int(d)

    return (1, 1)


def has_unreliable_pts(container) -> bool:
    """Check if container format has synthetic/unreliable PTS timestamps.

    Some streaming formats (like mpjpeg) provide PTS values that increment at
    constant intervals but don't reflect actual frame arrival timing. This causes
    burst behavior when frames arrive irregularly from the network.

    Args:
        container: PyAV container object

    Returns:
        True if the format is known to have unreliable PTS timing
    """
    format_name = container.format.name if container.format else ""
    # MIME multipart JPEG (IP camera HTTP streams) and piped JPEG sequences
    # typically have synthetic PTS that doesn't match real frame timing
    return format_name in ("mpjpeg", "jpeg_pipe")


def estimate_black_bars(
    frame_w: int, frame_h: int, gray: np.ndarray, thresh: int, min_px: int, max_ratio: float
) -> dict[str, int]:
    """Return {'l','r','t','b'} estimated black bar widths in pixels for one frame."""
    h, w = gray.shape  # (H, W)

    def walk_left():
        cap = int(w * max_ratio)
        x = 0
        while x < min(cap, w - 1):
            if int(np.median(gray[:, x : min(x + 4, w)])) > thresh:
                break
            x += 1
        return max(min_px if x >= min_px else 0, min(x, cap))

    def walk_right():
        cap = int(w * max_ratio)
        x = 0
        while x < min(cap, w - 1):
            if int(np.median(gray[:, max(0, w - 1 - (x + 3)) : w - x])) > thresh:
                break
            x += 1
        return max(min_px if x >= min_px else 0, min(x, cap))

    def walk_top():
        cap = int(h * max_ratio)
        y = 0
        while y < min(cap, h - 1):
            if int(np.median(gray[y : min(y + 4, h), :])) > thresh:
                break
            y += 1
        return max(min_px if y >= min_px else 0, min(y, cap))

    def walk_bot():
        cap = int(h * max_ratio)
        y = 0
        while y < min(cap, h - 1):
            if int(np.median(gray[max(0, h - 1 - (y + 3)) : h - y, :])) > thresh:
                break
            y += 1
        return max(min_px if y >= min_px else 0, min(y, cap))

    return {"l": walk_left(), "r": walk_right(), "t": walk_top(), "b": walk_bot()}


class PyAvFrameIterator(FrameIterator):
    """Frame iterator for video files using PyAV."""

    def __init__(self, src_url: str, stream_options):
        super().__init__(src_url, stream_options)
        self.real_src_url = src_url  # May be updated for YouTube URLs
        self.http_opts: dict[str, Any] = {}  # May be updated for YouTube URLs
        self._container = None  # Will be set by async_init if called
        self._vstream = None
        self._unreliable_pts = False  # Will be set after container is opened

    async def async_init(self):
        """Async initialization to open stream without blocking event loop."""
        import asyncio

        loop = asyncio.get_event_loop()

        # Determine URL to open
        pyav_url = self.real_src_url
        if self.stream_options.enable_cache:
            pyav_url = f"cache:{pyav_url}"

        # Run av.open in thread executor to avoid blocking
        self._container, self._vstream = await loop.run_in_executor(
            None, open_stream, pyav_url, self.stream_options.hw, self.http_opts
        )

        # Detect if this format has unreliable/synthetic PTS
        self._unreliable_pts = has_unreliable_pts(self._container)

    @classmethod
    def can_handle(cls, src_url: str, content_type: str | None = None) -> bool:
        """Check if this is a video file or stream we can handle.

        Args:
            src_url: Source URL to check
            content_type: Optional Content-Type header from HTTP HEAD request

        Returns:
            True if this iterator can handle the source
        """
        try:
            from urllib.parse import urlparse

            # If content type is available, use it for detection
            if content_type:
                content_type_lower = content_type.lower().split(";")[0].strip()
                # Reject images - let PIL handle them
                if content_type_lower.startswith("image/"):
                    return False
                # Accept video types
                if content_type_lower.startswith("video/") or content_type_lower.startswith("application/"):
                    return True

            # Parse URL to get path and scheme
            parsed = urlparse(src_url)
            path = parsed.path.lower() if parsed.path else ""
            scheme = parsed.scheme.lower()

            # First, check if this looks like an image that PIL should handle
            image_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp")
            if any(path.endswith(ext) for ext in image_extensions):
                return False  # Let PIL handle these

            # Video file extensions
            video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm", ".m4v", ".3gp")
            if any(path.endswith(ext) for ext in video_extensions):
                return True

            # Streaming protocols
            if scheme in ("rtmp", "rtsp", "udp", "tcp"):
                return True

            # HTTP/HTTPS URLs without content-type or extension info - accept as fallback
            # (PIL will get first chance via content-type detection)
            if scheme in ("http", "https"):
                return True

            # Local file without clear extension - let PyAV try to handle it as video
            # (but only if it's not http/https, which we already checked above)
            return scheme not in ("http", "https")
        except Exception:
            return False

    def __iter__(self) -> Iterator[tuple[bytes, float]]:
        """Iterate frames from video sources using PyAV filter graph."""
        from ..utils.helpers import is_youtube_url

        if is_youtube_url(self.src_url) and self.real_src_url == self.src_url:
            # If real_src_url wasn't updated, resolution may have failed
            # Try to proceed with original URL - if it fails, we'll get HTTP errors
            logging.getLogger("video").warning(
                f"YouTube URL resolution may have failed, trying original URL: {self.src_url}"
            )

        # Implement the PyAV iteration directly in the protocol class
        config = Config()
        target_width, target_height = self.stream_options.size

        # Auto-crop (black bar) config/state
        ac_cfg = config.get("video.autocrop")
        ac_enabled = bool(ac_cfg.get("enabled"))
        ac_probe_frames = int(ac_cfg.get("probe_frames"))
        ac_thresh = int(ac_cfg.get("luma_thresh"))
        ac_max_ratio = float(ac_cfg.get("max_bar_ratio"))
        ac_min_px = int(ac_cfg.get("min_bar_px"))
        ac_samples: dict[str, list[int]] = {"l": [], "r": [], "t": [], "b": []}
        ac_decided = False
        ac_crop = {"l": 0, "r": 0, "t": 0, "b": 0}  # in source pixel coords
        ac_seen = 0

        try:
            from av.error import (
                BlockingIOError as AvBlockingIOError,  # type: ignore[import]  # PyAV stubs may not include this
            )
        except Exception:
            AvBlockingIOError = None  # type: ignore[misc,assignment]  # noqa: N806  # Fallback when PyAV doesn't have BlockingIOError

        first_graph_log_done = False
        frames_decoded = 0  # Initialize outside try block for error logging
        container = None  # Initialize for error handling scope
        vstream = None  # Initialize for error handling scope

        try:
            # Determine URL to open (with optional cache: protocol)
            pyav_url = self.real_src_url

            # Enable FFmpeg cache protocol if requested
            if self.stream_options.enable_cache:
                logging.getLogger("video").info("Using FFmpeg cache: protocol")
                pyav_url = f"cache:{pyav_url}"

            # Log HTTP options if present (debug only, don't log auth tokens)
            if self.http_opts and logging.getLogger("video").isEnabledFor(logging.DEBUG):
                debug_opts = {k: (f"{len(v)} chars" if k == "headers" else v) for k, v in self.http_opts.items()}
                logging.getLogger("video").debug(f"HTTP options: {debug_opts}")

            # Use pre-opened container from async_init
            assert self._container is not None and self._vstream is not None, (
                "Container not initialized. Must call async_init() before iteration."
            )

            container = self._container
            vstream = self._vstream

            # Detect if this format has unreliable/synthetic PTS
            self._unreliable_pts = has_unreliable_pts(container)

            # Default frame delay if timestamps are missing
            avg_ms: float | None = None
            if vstream.average_rate:
                fps = float(vstream.average_rate)
                if fps > 0:
                    avg_ms = max(MIN_DELAY_MS, 1000.0 / fps)

            if self._unreliable_pts:
                format_name = container.format.name if container.format else "unknown"
                if avg_ms is not None:
                    logging.getLogger("video").info(
                        f"Format {format_name} has unreliable PTS - using fixed interval ({avg_ms:.1f}ms)"
                    )
                else:
                    # For streams without metadata, use MIN_DELAY_MS to output at natural arrival rate
                    # (demuxer paces frames from network, don't add artificial delay)
                    logging.getLogger("video").info(
                        f"Format {format_name} has unreliable PTS - no average_rate, using natural timing ({MIN_DELAY_MS}ms)"
                    )

            # Rebuildable filter graph state
            graph = None
            src_in = sink_out = None
            g_props: dict[str, Any] = {"w": None, "h": None, "fmt": None, "sar": (1, 1), "rot": 0}

            def ensure_graph_for(frame) -> None:
                """(Re)build the filter graph if geometry/SAR/format/rotation changed."""
                nonlocal graph, src_in, sink_out, first_graph_log_done, g_props

                w, h = int(frame.width), int(frame.height)
                fmt_name = getattr(frame.format, "name", "rgb24")
                sar_n, sar_d = sar_of(frame, vstream)
                rot = rotation_from_stream_and_frame(vstream, frame)

                # Track whether we've already applied the autocrop in the current graph
                applied_ac = g_props.get("ac_applied", False)
                want_ac = bool(ac_enabled and ac_decided and any(v > 0 for v in ac_crop.values()))

                need_rebuild = (
                    graph is None
                    or g_props["w"] != w
                    or g_props["h"] != h
                    or g_props["fmt"] != fmt_name
                    or g_props["sar"] != (sar_n, sar_d)
                    or g_props["rot"] != rot
                    or (want_ac and not applied_ac)
                )
                if not need_rebuild:
                    return

                old = g_props.copy()
                g_props.update(
                    {
                        "w": w,
                        "h": h,
                        "fmt": fmt_name,
                        "sar": (sar_n, sar_d),
                        "rot": rot,
                        "ac_applied": want_ac,
                    }
                )
                logging.getLogger("video").debug(
                    f"rebuild: {old} -> {g_props} (ac={ac_crop if (ac_enabled and ac_decided) else 'pending'})"
                )

                if not first_graph_log_done:
                    cc = getattr(vstream, "codec_context", None)
                    in_sar = getattr(cc, "sample_aspect_ratio", None)
                    logging.getLogger("video").info(f"input codec SAR={in_sar} size={w}x{h}")
                    first_graph_log_done = True

                g = AvFilterGraph()

                tb_n, tb_d = tb_num_den(frame.time_base or vstream.time_base)
                fr_n, fr_d = tb_num_den(getattr(vstream, "average_rate", None))
                rate_arg = f":frame_rate={fr_n}/{fr_d}" if (fr_n and fr_d) else ""

                # buffersrc with input SAR (we normalize later)
                src = g.add(
                    "buffer",
                    args=(
                        f"video_size={w}x{h}:"
                        f"pix_fmt={fmt_name}:"
                        f"time_base={tb_n}/{tb_d}:"
                        f"pixel_aspect={sar_n}/{sar_d}" + rate_arg
                    ),
                )
                last = src

                # (A) autocrop baked bars in source pixel coords, if decided
                if want_ac:
                    left, right, top, bottom = ac_crop["l"], ac_crop["r"], ac_crop["t"], ac_crop["b"]
                    cw = max(1, w - (left + right))
                    ch = max(1, h - (top + bottom))
                    n = g.add("crop", args=f"{cw}:{ch}:{left}:{top}")
                    last.link_to(n)
                    last = n

                # unsqueeze PAR -> setsar=1
                n = g.add("scale", args="iw*sar:ih")
                last.link_to(n)
                last = n
                n = g.add("setsar", args="1")
                last.link_to(n)
                last = n

                # rotate (metadata) if needed
                if rot in (90, 180, 270):
                    if rot == 90:
                        n = g.add("transpose", args="clock")
                        last.link_to(n)
                        last = n
                    elif rot == 270:
                        n = g.add("transpose", args="cclock")
                        last.link_to(n)
                        last = n
                    else:  # 180
                        t1 = g.add("transpose", args="clock")
                        t2 = g.add("transpose", args="clock")
                        last.link_to(t1)
                        t1.link_to(t2)
                        last = t2

                # expand TV->PC range before final downscale if requested
                expand_args = ""
                if self.stream_options.expand == 2:
                    expand_args = ":in_range=tv:out_range=pc"
                elif self.stream_options.expand == 1:
                    expand_args = ":in_range=auto:out_range=pc"

                # Fit selection
                fit_mode = self.stream_options.fit
                if fit_mode == "cover":
                    n = g.add(
                        "scale",
                        args=f"{target_width}:{target_height}:flags=bilinear:force_original_aspect_ratio=increase"
                        + expand_args,
                    )
                    last.link_to(n)
                    last = n
                    n = g.add(
                        "crop", args=f"{target_width}:{target_height}:(in_w-{target_width})/2:(in_h-{target_height})/2"
                    )
                    last.link_to(n)
                    last = n
                elif fit_mode == "auto":
                    # Smart fit: check if aspect ratios match
                    src_ratio = (w * sar_n) / (h * sar_d) if sar_d != 0 else w / h
                    target_ratio = target_width / target_height

                    if abs(src_ratio - target_ratio) < 0.01:  # Aspect ratios match
                        # Just scale directly - no padding or cropping needed
                        n = g.add("scale", args=f"{target_width}:{target_height}:flags=bilinear" + expand_args)
                        last.link_to(n)
                        last = n
                    else:
                        # Fall back to pad behavior for mismatched ratios
                        n = g.add(
                            "scale",
                            args=f"{target_width}:{target_height}:flags=bilinear:force_original_aspect_ratio=decrease"
                            + expand_args,
                        )
                        last.link_to(n)
                        last = n
                        n = g.add("pad", args=f"{target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black")
                        last.link_to(n)
                        last = n
                else:  # "pad" mode
                    n = g.add(
                        "scale",
                        args=f"{target_width}:{target_height}:flags=bilinear:force_original_aspect_ratio=decrease"
                        + expand_args,
                    )
                    last.link_to(n)
                    last = n
                    n = g.add("pad", args=f"{target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black")
                    last.link_to(n)
                    last = n

                n = g.add("setdar", args="1")
                last.link_to(n)
                last = n
                n = g.add("format", args="rgb24")
                last.link_to(n)
                last = n

                sink = g.add("buffersink")
                last.link_to(sink)
                g.configure()

                graph = g
                src_in = src
                sink_out = sink

            # Main iteration loop
            loop_count = 0
            loop_video = self.stream_options.loop

            while True:
                loop_count += 1
                saw_frame = False

                # Seek to beginning for 2nd+ loops when cache is enabled
                if loop_count > 1 and self.stream_options.enable_cache:
                    try:
                        logging.getLogger("video").debug(f"Seeking to start for loop {loop_count} (cache enabled)")
                        container.seek(0)
                    except Exception as e:
                        logging.getLogger("video").warning(f"Failed to seek to start: {e}, will continue without seek")

                # Reset per-loop state
                last_pts_s: float | None = None

                for packet in container.demux(vstream):
                    # decode() may return 0..N frames (depending on codec & bottom-frames)
                    frames = packet.decode()
                    for frame in frames:
                        # PyAV packet.decode() stub is incomplete - returns VideoFrame for video streams
                        if not isinstance(frame, VideoFrame):
                            continue  # Skip non-video frames (shouldn't happen with video stream)
                        saw_frame = True
                        # Auto-crop sampling on early frames (before building graph)
                        if ac_enabled and not ac_decided and ac_seen < ac_probe_frames:
                            try:
                                gray = frame.to_ndarray(format="gray")
                                cand = estimate_black_bars(
                                    frame.width, frame.height, gray, ac_thresh, ac_min_px, ac_max_ratio
                                )
                                # Adjust for rotation metadata
                                r = rotation_from_stream_and_frame(vstream, frame)
                                if r == 90:
                                    cand = {"l": cand["t"], "r": cand["b"], "t": cand["r"], "b": cand["l"]}
                                elif r == 270:
                                    cand = {"l": cand["b"], "r": cand["t"], "t": cand["l"], "b": cand["r"]}
                                elif r == 180:
                                    cand = {"l": cand["r"], "r": cand["l"], "t": cand["b"], "b": cand["t"]}
                                for k in ("l", "r", "t", "b"):
                                    ac_samples[k].append(int(cand[k]))
                            except Exception:  # noqa: S110  # Skip invalid autocrop samples
                                pass
                            finally:
                                ac_seen += 1
                                if ac_seen >= ac_probe_frames:
                                    import statistics as _st

                                    ac_crop = {k: int(_st.median(v) if v else 0) for k, v in ac_samples.items()}
                                    ac_decided = True
                                    # Force a rebuild next frame to apply crop
                                    graph = None

                        ensure_graph_for(frame)
                        assert src_in is not None, "Filter graph must be initialized"
                        src_in.push(frame)  # type: ignore[name-defined]  # src_in defined conditionally via ensure_graph_for

                        # Pull 0..N filtered frames
                        out_frames = []
                        while True:
                            try:
                                assert sink_out is not None, "Filter graph must be initialized"
                                of = sink_out.pull()  # type: ignore[name-defined]  # sink_out defined conditionally via ensure_graph_for
                                out_frames.append(of)
                            except Exception as pe:
                                # Check for PyAV blocking IO errors
                                if (
                                    (AvBlockingIOError is not None and isinstance(pe, AvBlockingIOError))
                                    or getattr(pe, "errno", None) in (11, 35)
                                    or "resource temporarily unavailable" in str(pe).lower()
                                    or "eagain" in str(pe).lower()
                                ):
                                    break
                                raise

                        for of in out_frames:
                            rgb888 = of.to_ndarray(format="rgb24").tobytes()  # type: ignore[attr-defined]  # PyAV Frame.to_ndarray returns array-like with tobytes
                            frames_decoded += 1  # Increment frame counter

                            # For unreliable PTS formats (MJPEG), use fixed interval from avg_ms if available,
                            # otherwise use MIN_DELAY_MS to output at natural network arrival rate
                            delay_ms: float = float(avg_ms) if (avg_ms is not None) else MIN_DELAY_MS

                            # Only use PTS-based timing for formats with reliable timestamps
                            if not self._unreliable_pts:
                                pts_s = None
                                if of.pts is not None:
                                    tb_n, tb_d = tb_num_den(of.time_base or vstream.time_base)
                                    pts_s = float(of.pts) * (tb_n / tb_d)
                                if pts_s is not None:
                                    if last_pts_s is None:
                                        delay_ms = float(avg_ms) if (avg_ms is not None) else 33.33
                                    else:
                                        delta_ms = (pts_s - last_pts_s) * 1000.0
                                        if avg_ms is not None:
                                            low = 0.75 * avg_ms
                                            high = 1.25 * avg_ms
                                            delta_ms = max(low, min(high, delta_ms))
                                        elif delta_ms <= 0:
                                            delta_ms = 33.33
                                        delay_ms = max(MIN_DELAY_MS, float(delta_ms))
                                    last_pts_s = pts_s

                            yield rgb888, float(delay_ms)

                # End of single iteration - check if we should loop
                if not saw_frame:
                    # No frames decoded - likely end of stream or error
                    raise RuntimeError(f"no frames decoded from source: {self.src_url}")

                if not loop_video:
                    # Not looping - exit after first iteration
                    break

                # For non-cached streams, exit after one iteration
                # and let streaming core create fresh iterator for each loop
                if not self.stream_options.enable_cache:
                    logging.getLogger("video").debug(
                        "Non-cached stream exhausted after one iteration, "
                        "returning to streaming core for fresh connection"
                    )
                    break

        except (HTTPError, HTTPClientError, HTTPServerError):
            # HTTP errors - re-raise for upstream handling (streaming layer will decide if retry needed)
            raise
        except FileNotFoundError as e:
            # File not found - re-raise with clearer message
            raise FileNotFoundError(f"cannot open source: {self.src_url}") from e
        except Exception as e:
            # Check for file-not-found-like errors in the message
            msg = str(e).lower()
            if "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(f"cannot open source: {self.src_url}") from e

            # Enhanced error logging for I/O errors and other failures
            errno_val = getattr(e, "errno", None)

            # Get codec information if available
            try:
                if vstream and hasattr(vstream, "codec_context"):
                    codec_name = getattr(vstream.codec_context, "name", "unknown")
                else:
                    codec_name = "unknown"
            except Exception:
                codec_name = "unknown"

            # Special handling for I/O errors (errno 5)
            if errno_val == 5 or "i/o error" in msg:
                logging.getLogger("video").error(
                    f"I/O error during decode: errno={errno_val} codec={codec_name} "
                    f"hw={self.stream_options.hw} frames_decoded={frames_decoded} "
                    f"http_opts={list(self.http_opts.keys()) if self.http_opts else 'none'}"
                )
            else:
                # Log other errors with diagnostic context
                logging.getLogger("video").error(
                    f"Decode error: {type(e).__name__} errno={errno_val} codec={codec_name} "
                    f"hw={self.stream_options.hw} frames_decoded={frames_decoded}"
                )

            # Re-raise all other errors for upstream handling
            raise RuntimeError(f"av error: {e}") from e
        finally:
            # Always close container, even on errors
            if container:
                with contextlib.suppress(Exception):
                    container.close()

    def cleanup(self) -> None:
        """Clean up any resources used by the iterator."""
        # PyAV handles its own cleanup via context managers
        pass
