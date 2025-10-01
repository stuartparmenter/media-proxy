# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import logging
import os
import platform
import shutil
import subprocess
from typing import Optional

# Cache for hardware acceleration detection
_hw_backend_cache = None


def get_ffmpeg_exe_path() -> Optional[str]:
    """Find FFmpeg executable path."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # type: ignore[import]  # imageio-ffmpeg has no type stubs
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _build_hw_backend_cache():
    """Build cache of available hardware accelerations."""
    exe = get_ffmpeg_exe_path()
    if not exe:
        return set()

    try:
        out = subprocess.check_output([exe, "-hide_banner", "-hwaccels"], text=True, stderr=subprocess.STDOUT)
        return {line.strip().lower() for line in out.splitlines() if line.strip()}
    except Exception:
        return set()


def pick_hw_backend(prefer: Optional[str] = None) -> Optional[str]:
    """Pick the best hardware acceleration backend for this system."""
    global _hw_backend_cache
    if _hw_backend_cache is None:
        _hw_backend_cache = _build_hw_backend_cache()

    sys = platform.system().lower()
    prefer = (prefer or "auto").lower()
    ALIASES = {"d3d11": "d3d11va"}

    def norm(n):
        return ALIASES.get(n, n)

    if prefer not in ("", "auto", "none"):
        pn = norm(prefer)
        return pn if pn in _hw_backend_cache else None

    # Auto selection based on platform
    candidates: tuple[str, ...]
    if sys == "windows":
        candidates = ("cuda", "d3d11va", "qsv")
    elif sys == "darwin":
        candidates = ("videotoolbox",)
    else:
        candidates = ("vaapi", "qsv", "cuda")

    for cand in candidates:
        if cand in _hw_backend_cache:
            return cand

    return None


def set_windows_timer_resolution(enable: bool = True) -> None:
    """Set Windows timer resolution for better timing accuracy."""
    if os.name == "nt":
        try:
            import ctypes
            winmm = ctypes.windll.winmm  # type: ignore[attr-defined]  # windll only exists on Windows
            if enable:
                winmm.timeBeginPeriod(1)
            else:
                winmm.timeEndPeriod(1)
        except Exception as e:
            logging.getLogger('hardware').warning(f"timeBeginPeriod/timeEndPeriod failed: {e}")