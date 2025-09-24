# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import logging
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
    Choose hardware decode preference. Always prefer hardware acceleration when available.
    Returns preference string to pass to pick_hw_backend().
    """
    if prefer and str(prefer).lower() not in ("", "auto"):
        logging.getLogger('hardware').info(f"prefer={prefer!r} explicitly requested -> honoring")
        return prefer

    # Always use hardware acceleration if available
    kind, _ = pick_hw_backend("auto")
    logging.getLogger('hardware').info(f"choosing HW 'auto'; auto maps to {kind or 'none'}")
    return "auto"


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
            logging.getLogger('hardware').warning(f"timeBeginPeriod/timeEndPeriod failed: {e}")