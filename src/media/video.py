# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import contextlib
import logging
import numpy as np
from typing import Iterator, Tuple, Dict, Optional
import av
from av.filter import Graph as AvFilterGraph

from ..config import Config
from ..utils.hardware import pick_hw_backend
from .protocol import FrameIterator, FrameIteratorConfig

MIN_DELAY_MS = 10.0


def open_with_hwaccel(src_url: str, prefer: Optional[str], options: Optional[Dict[str, str]] = None):
    """Open media container with optional hardware acceleration."""
    kind, _kw = pick_hw_backend(prefer)
    options = options or {}
    
    try:
        if kind:
            try:
                from av.codec.hwaccel import HWAccel
            except Exception as ie:
                raise RuntimeError(f"hwaccel API unavailable: {ie}")
            container = av.open(src_url, mode="r", hwaccel=HWAccel(device_type=kind), options=options)
            logging.getLogger('video').info(f"selected {kind} for decode")
        else:
            logging.getLogger('video').info("using CPU decode (no HW accel selected)")
            container = av.open(src_url, mode="r", options=options)
        
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise RuntimeError("no video stream")
        return container, vstream
        
    except Exception as e:
        logging.getLogger('video').info(f"hwaccel disabled: {kind or 'auto'} not available: {e}")
        container = av.open(src_url, mode="r", options=options)
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise RuntimeError("no video stream")
        return container, vstream


def rotation_from_stream_and_frame(vstream, frame) -> int:
    """Extract rotation metadata from stream or frame."""
    try:
        rot = getattr(frame, "rotation", None)
        if isinstance(rot, int):
            return ((rot % 360) + 360) % 360
    except Exception:
        pass
    try:
        md = getattr(vstream, "metadata", {})
        if md:
            r = md.get("rotate")
            if r is not None:
                return ((int(str(r)) % 360) + 360) % 360
    except Exception:
        pass
    return 0


def tb_num_den(tb) -> Tuple[int, int]:
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


def sar_of(obj, vstream) -> Tuple[int, int]:
    """Extract sample aspect ratio from frame or stream."""
    # Try frame SAR first; fall back to codec_context SAR; else 1:1
    try:
        sar = getattr(obj, "sample_aspect_ratio", None)
        n = getattr(sar, "num", getattr(sar, "numerator", None))
        d = getattr(sar, "den", getattr(sar, "denominator", None))
        if n and d and n > 0 and d > 0:
            return int(n), int(d)
    except Exception:
        pass
    try:
        cc = getattr(vstream, "codec_context", None)
        sar = getattr(cc, "sample_aspect_ratio", None)
        n = getattr(sar, "num", getattr(sar, "numerator", None))
        d = getattr(sar, "den", getattr(sar, "denominator", None))
        if n and d and n > 0 and d > 0:
            return int(n), int(d)
    except Exception:
        pass
    return (1, 1)


def estimate_black_bars(frame_w: int, frame_h: int, gray: np.ndarray,
                       thresh: int, min_px: int, max_ratio: float) -> Dict[str, int]:
    """Return {'l','r','t','b'} estimated black bar widths in pixels for one frame."""
    h, w = gray.shape  # (H, W)
    
    def walk_left():
        cap = int(w * max_ratio)
        x = 0
        while x < min(cap, w - 1):
            if int(np.median(gray[:, x:min(x+4, w)])) > thresh: 
                break
            x += 1
        return max(min_px if x >= min_px else 0, min(x, cap))
    
    def walk_right():
        cap = int(w * max_ratio)
        x = 0
        while x < min(cap, w - 1):
            if int(np.median(gray[:, max(0, w-1-(x+3)):w-x])) > thresh: 
                break
            x += 1
        return max(min_px if x >= min_px else 0, min(x, cap))
    
    def walk_top():
        cap = int(h * max_ratio)
        y = 0
        while y < min(cap, h - 1):
            if int(np.median(gray[y:min(y+4, h), :])) > thresh: 
                break
            y += 1
        return max(min_px if y >= min_px else 0, min(y, cap))
    
    def walk_bot():
        cap = int(h * max_ratio)
        y = 0
        while y < min(cap, h - 1):
            if int(np.median(gray[max(0, h-1-(y+3)):h-y, :])) > thresh: 
                break
            y += 1
        return max(min_px if y >= min_px else 0, min(y, cap))
    
    return {"l": walk_left(), "r": walk_right(), "t": walk_top(), "b": walk_bot()}




class PyAvFrameIterator(FrameIterator):
    """Frame iterator for video files using PyAV."""

    def __init__(self, src_url: str, config: FrameIteratorConfig):
        super().__init__(src_url, config)
        self.real_src_url = src_url  # May be updated for YouTube URLs
        self.http_opts = {}  # May be updated for YouTube URLs

    @classmethod
    def can_handle(cls, src_url: str) -> bool:
        """Check if this is a video file or stream we can handle."""
        try:
            from urllib.parse import urlparse
            # Parse URL to get path without query parameters
            parsed = urlparse(src_url.lower())
            path = parsed.path if parsed.path else src_url.lower()
            lower_url = src_url.lower()

            # First, check if this looks like an image that PIL should handle
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')
            if any(path.endswith(ext) for ext in image_extensions):
                return False  # Let PIL handle these

            # Video file extensions
            video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.3gp')
            if any(path.endswith(ext) for ext in video_extensions):
                return True

            # Streaming protocols (but be careful with HTTP images)
            streaming_prefixes = ('rtmp://', 'rtsp://', 'udp://', 'tcp://')
            if any(lower_url.startswith(prefix) for prefix in streaming_prefixes):
                return True

            # HTTP/HTTPS URLs - accept these as potential video streams
            # (but PIL will get first chance at obvious image URLs)
            if lower_url.startswith(('http://', 'https://')):
                return True

            # Local file without clear extension - let PyAV try to handle it as video
            # (but only if it's not an obvious image extension)
            if not lower_url.startswith(('http://', 'https://')):
                return True

            return False
        except Exception:
            return False

    def __iter__(self) -> Iterator[Tuple[bytes, float]]:
        """Iterate frames from video sources using PyAV filter graph."""
        from ..utils.helpers import is_youtube_url
        import asyncio

        # For YouTube URLs, we need to resolve synchronously here
        # This is a limitation of the current design - the protocol is sync but resolution is async
        # For now, use the real_src_url that should be set by the streaming layer
        http_opts = {}

        if is_youtube_url(self.src_url) and self.real_src_url == self.src_url:
            # If real_src_url wasn't updated, resolution may have failed
            # Try to proceed with original URL - if it fails, we'll get HTTP errors
            logging.getLogger('video').warning(f"YouTube URL resolution may have failed, trying original URL: {self.src_url}")

        # Implement the PyAV iteration directly in the protocol class
        config = Config()
        TW, TH = self.config.size

        # Auto-crop (black bar) config/state
        ac_cfg = config.get("video.autocrop", {})
        ac_enabled = bool(ac_cfg.get("enabled", True))
        ac_probe_frames = int(ac_cfg.get("probe_frames", 8))
        ac_thresh = int(ac_cfg.get("luma_thresh", 22))
        ac_max_ratio = float(ac_cfg.get("max_bar_ratio", 0.20))
        ac_min_px = int(ac_cfg.get("min_bar_px", 2))
        ac_samples = {"l": [], "r": [], "t": [], "b": []}
        ac_decided = False
        ac_crop = {"l": 0, "r": 0, "t": 0, "b": 0}  # in source pixel coords
        ac_seen = 0

        try:
            from av.error import BlockingIOError as AvBlockingIOError  # type: ignore
        except Exception:
            AvBlockingIOError = None  # type: ignore

        first_graph_log_done = False

        try:
            container, vstream = open_with_hwaccel(self.real_src_url, self.config.hw_prefer, options=self.http_opts)
            if vstream is None:
                raise RuntimeError("no video stream")

            # Default frame delay if timestamps are missing
            avg_ms: Optional[float] = None
            if vstream.average_rate:
                try:
                    fps = float(vstream.average_rate)
                    if fps > 0:
                        avg_ms = max(MIN_DELAY_MS, 1000.0 / fps)
                except Exception:
                    pass

            # Rebuildable filter graph state
            graph = None
            src_in = sink_out = None
            g_props = {"w": None, "h": None, "fmt": None, "sar": (1, 1), "rot": 0}

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
                    graph is None or
                    g_props["w"] != w or
                    g_props["h"] != h or
                    g_props["fmt"] != fmt_name or
                    g_props["sar"] != (sar_n, sar_d) or
                    g_props["rot"] != rot or
                    (want_ac and not applied_ac)
                )
                if not need_rebuild:
                    return

                old = g_props.copy()
                g_props.update({"w": w, "h": h, "fmt": fmt_name, "sar": (sar_n, sar_d), "rot": rot, "ac_applied": want_ac})
                logging.getLogger('video').debug(f"rebuild: {old} -> {g_props} (ac={ac_crop if (ac_enabled and ac_decided) else 'pending'})")

                if not first_graph_log_done:
                    try:
                        cc = getattr(vstream, "codec_context", None)
                        in_sar = getattr(cc, "sample_aspect_ratio", None)
                        logging.getLogger('video').info(f"input codec SAR={in_sar} size={w}x{h}")
                    except Exception:
                        pass
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
                    L, R, T, B = ac_crop["l"], ac_crop["r"], ac_crop["t"], ac_crop["b"]
                    cw = max(1, w - (L + R))
                    ch = max(1, h - (T + B))
                    n = g.add("crop", args=f"{cw}:{ch}:{L}:{T}")
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
                if self.config.expand_mode == 2:
                    expand_args = ":in_range=tv:out_range=pc"
                elif self.config.expand_mode == 1:
                    expand_args = ":in_range=auto:out_range=pc"

                # Fit selection
                fit_mode = str(config.get("video.fit")).lower()
                if fit_mode == "cover":
                    n = g.add("scale", args=f"{TW}:{TH}:flags=bilinear:force_original_aspect_ratio=increase" + expand_args)
                    last.link_to(n)
                    last = n
                    n = g.add("crop", args=f"{TW}:{TH}:(in_w-{TW})/2:(in_h-{TH})/2")
                    last.link_to(n)
                    last = n
                else:
                    n = g.add("scale", args=f"{TW}:{TH}:flags=bilinear:force_original_aspect_ratio=decrease" + expand_args)
                    last.link_to(n)
                    last = n
                    n = g.add("pad", args=f"{TW}:{TH}:(ow-iw)/2:(oh-ih)/2:color=black")
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

            last_pts_s: Optional[float] = None

            for packet in container.demux(vstream):
                # decode() may return 0..N frames (depending on codec & B-frames)
                frames = packet.decode()
                for frame in frames:
                    # Auto-crop sampling on early frames (before building graph)
                    if ac_enabled and not ac_decided and ac_seen < ac_probe_frames:
                        try:
                            gray = frame.to_ndarray(format="gray")
                            cand = estimate_black_bars(frame.width, frame.height, gray,
                                                     ac_thresh, ac_min_px, ac_max_ratio)
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
                        except Exception:
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
                    src_in.push(frame)  # type: ignore[name-defined]

                    # Pull 0..N filtered frames
                    out_frames = []
                    while True:
                        try:
                            of = sink_out.pull()  # type: ignore[name-defined]
                            out_frames.append(of)
                        except Exception as pe:
                            if (AvBlockingIOError and isinstance(pe, AvBlockingIOError)) \
                               or getattr(pe, "errno", None) in (11, 35) \
                               or "resource temporarily unavailable" in str(pe).lower() \
                               or "eagain" in str(pe).lower():
                                break
                            raise

                    for of in out_frames:
                        rgb888 = of.to_ndarray(format="rgb24").tobytes()

                        # Compute inter-frame delay using PTS if available; otherwise avg_ms fallback
                        delay_ms: float = float(avg_ms) if (avg_ms is not None) else 1000.0 / 10.0
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

            container.close()

        except (av.error.HTTPError, av.error.HTTPClientError, av.error.HTTPServerError) as e:
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

            # Re-raise all other errors for upstream handling
            raise RuntimeError(f"av error: {e}") from e

    def cleanup(self) -> None:
        """Clean up any resources used by the iterator."""
        # PyAV handles its own cleanup via context managers
        pass