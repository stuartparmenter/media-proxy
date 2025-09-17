# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import os
import platform
import shutil
import subprocess
from typing import Optional, Tuple, Dict, Any


def get_ffmpeg_exe_path() -> Optional[str]:
    """Find FFmpeg executable path."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def ffmpeg_has_hwaccel(name: str) -> bool:
    """Check if FFmpeg supports a specific hardware acceleration."""
    exe = get_ffmpeg_exe_path()
    if not exe:
        return False
    try:
        out = subprocess.check_output([exe, "-hide_banner", "-hwaccels"], text=True, stderr=subprocess.STDOUT)
        return any(line.strip().lower() == name.lower() for line in out.splitlines())
    except Exception:
        return False


def pick_hw_backend(prefer: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    """Pick the best hardware acceleration backend for this system."""
    sys = platform.system().lower()
    prefer = (prefer or "auto").lower()

    ALIASES = {"d3d11": "d3d11va", "gpu": "cuda"}

    def ok(n): 
        return ffmpeg_has_hwaccel(n)
    
    def norm(n): 
        return ALIASES.get(n, n)

    if prefer not in ("", "auto", "none"):
        pn = norm(prefer)
        return (pn if ok(pn) else None, {})

    if sys == "windows":
        for cand in ("cuda", "d3d11va", "qsv"):
            if ok(cand):
                return cand, {}
        return (None, {})
    
    if sys == "darwin":
        if ok("videotoolbox"):
            return "videotoolbox", {}
        return (None, {})
    
    for cand in ("vaapi", "qsv", "cuda"):
        if ok(cand):
            return cand, {}
    return (None, {})


def choose_decode_preference(
    source_url: str,
    prefer: Optional[str],
    output_size: Tuple[int, int],
    expand_mode: int
) -> Optional[str]:
    """
    Decide CPU vs HW decode based on input properties and output requirements.
    Returns preference string to pass to pick_hw_backend().
    """
    if prefer and str(prefer).lower() not in ("", "auto"):
        print(f"[decode] prefer={prefer!r} explicitly requested -> honoring")
        return prefer

    # For YouTube URLs, skip probing and choose auto
    from .helpers import is_youtube_url
    if is_youtube_url(source_url):
        kind, _ = pick_hw_backend("auto")
        print(f"[decode] source=YouTube page -> choosing HW 'auto'; auto maps to {kind or 'none'}")
        return "auto"

    TW, TH = output_size

    # Probe input to get dimensions/fps/codec
    in_w = in_h = 0
    in_fps: Optional[float] = None
    in_codec = ""
    
    try:
        import av
        import contextlib
        
        pc = av.open(source_url, mode="r")
        try:
            pvs = next((s for s in pc.streams if s.type == "video"), None)
            if pvs is not None:
                in_w = int(getattr(getattr(pvs, "codec_context", None), "width", 0) or getattr(pvs, "width", 0) or 0)
                in_h = int(getattr(getattr(pvs, "codec_context", None), "height", 0) or getattr(pvs, "height", 0) or 0)
                try:
                    in_fps = float(pvs.average_rate) if pvs.average_rate else None
                except Exception:
                    in_fps = None
                in_codec = (getattr(getattr(pvs, "codec", None), "name", "") or "").lower()
        finally:
            with contextlib.suppress(Exception):
                pc.close()
    except Exception as e:
        print(f"[decode] probe failed ({e!r}); defaulting to auto")
        return "auto"

    # Heuristics
    BIG_PIXELS = 1920 * 1080
    HIGH_FPS = 50.0
    SMALL_OUT = 128 * 128
    HARD_CODECS = ("hevc", "h265", "av1", "vp9")

    input_big = (in_w * in_h) >= BIG_PIXELS
    input_fast = (in_fps or 0.0) >= HIGH_FPS
    hard_codec = any(k in in_codec for k in HARD_CODECS)
    output_tiny = (TW * TH) <= SMALL_OUT
    expand_needed = expand_mode in (1, 2)

    reasons = []
    if input_big:
        reasons.append(">=1080p input")
    if input_fast:
        reasons.append(">=50 fps input")
    if hard_codec:
        reasons.append(f"hard codec ({in_codec or 'unknown'})")
    if output_tiny and expand_needed:
        reasons.append("tiny output + range-expand")

    if input_big or input_fast or hard_codec:
        kind, _ = pick_hw_backend("auto")
        why = ", ".join(reasons) or "input suggests HW"
        print(
            f"[decode] in={in_w}x{in_h}@{(in_fps or 0):.2f}fps codec={in_codec or 'unknown'} "
            f"out={TW}x{TH} expand={expand_mode} -> choosing HW ('auto') "
            f"because {why}; auto maps to {kind or 'none'}"
        )
        return "auto"

    why = ", ".join(reasons) or "small/slow input; HW adds overhead here"
    print(
        f"[decode] in={in_w}x{in_h}@{(in_fps or 0):.2f}fps codec={in_codec or 'unknown'} "
        f"out={TW}x{TH} expand={expand_mode} -> choosing CPU because {why}"
    )
    return "cpu"


def set_windows_timer_resolution(enable: bool = True) -> None:
    """Set Windows timer resolution for better timing accuracy."""
    if os.name == "nt":
        try:
            import ctypes
            if enable:
                ctypes.windll.winmm.timeBeginPeriod(1)
            else:
                ctypes.windll.winmm.timeEndPeriod(1)
        except Exception as e:
            print(f"[warn] timeBeginPeriod/timeEndPeriod failed: {e}")