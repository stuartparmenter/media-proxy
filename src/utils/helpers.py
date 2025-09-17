# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import os
from urllib.parse import urlparse, unquote
from typing import Optional, Dict, Tuple
import urllib.request as _urlreq

try:
    from urllib.request import url2pathname
except Exception:  # pragma: no cover
    def url2pathname(p): return p  # type: ignore[misc]


def is_youtube_url(url: str) -> bool:
    """Check if a URL is a YouTube URL."""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        return any(h in host for h in (
            "youtube.com", "youtu.be", "youtube-nocookie.com"
        ))
    except Exception:
        return False


def is_http_url(url: str) -> bool:
    """Check if a URL is HTTP/HTTPS."""
    try:
        s = (urlparse(url).scheme or "").lower()
        return s in ("http", "https")
    except Exception:
        return False


def resolve_local_path(src_url: str) -> Optional[str]:
    """Convert a file:// URL or local path to an absolute file path."""
    u = urlparse(src_url)
    if u.scheme in ("", "file"):
        p = u.path if u.scheme == "file" else src_url
        if os.name == "nt" and len(p) >= 3 and p[0] == "/" and p[2] == ":":
            p = p[1:]
        return url2pathname(p)
    return None


def truthy(s: str) -> bool:
    """Check if a string represents a truthy value."""
    return s.lower() not in ("0", "false", "no", "off", "")


def probe_http_content_type(url: str, timeout: float = 3.0) -> Optional[str]:
    """Try HEAD to get Content-Type. If server rejects HEAD, try a tiny GET with Range."""
    if not is_http_url(url):
        return None
    try:
        req = _urlreq.Request(url, method="HEAD")
        with _urlreq.urlopen(req, timeout=timeout) as resp:
            ct = resp.headers.get("Content-Type")
            return (ct or "").split(";")[0].strip().lower() or None
    except Exception:
        try:
            req = _urlreq.Request(url, method="GET", headers={"Range": "bytes=0-0"})
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                ct = resp.headers.get("Content-Type")
                return (ct or "").split(";")[0].strip().lower() or None
        except Exception:
            return None


def headers_dict_to_ffmpeg_opt(headers: Dict[str, str]) -> str:
    """
    FFmpeg's libavformat expects a single CRLF-delimited string in the `headers`
    option. Must end with a trailing CRLF.
    """
    if not headers:
        return ""
    lines = []
    for k, v in headers.items():
        if not k or v is None:
            continue
        k = str(k).strip()
        v = str(v).strip()
        if k and v:
            lines.append(f"{k}: {v}")
    return "\r\n".join(lines) + "\r\n"


def parse_expand_mode(val, default) -> int:
    """Parse expand mode configuration value."""
    s = str(val if val is not None else default).lower()
    if s in ("2", "force"):
        return 2
    if s in ("0", "false", "never"):
        return 0
    return 1  # auto


def parse_hw_preference(val, default) -> Optional[str]:
    """Parse hardware acceleration preference."""
    s = (val if val is not None else default)
    return None if str(s).lower() in ("none", "off", "cpu") else str(s)


def parse_pace_hz(val) -> int:
    """Parse and validate pace Hz value."""
    try:
        n = int(val or 0)
    except Exception:
        raise ValueError(f"pace must be integer Hz (got {val!r})")
    if n < 0:
        raise ValueError("pace must be >= 0")
    return n


def compute_spacing_and_group(pkt_count: int, frame_interval_s: float, config: Dict) -> Tuple[Optional[float], int]:
    """
    Compute (spacing_s, group_n) for packet spreading.
    
    spacing_s: sleep between packet *groups* (None => no spreading)
    group_n:   number of packets sent per timeslot (then sleep once)
    """
    import math

    if pkt_count <= 0 or frame_interval_s <= 0.0:
        return (None, 1)

    net_cfg = config.get("net", {})
    min_s = float(net_cfg.get("spread_min_ms")) / 1000.0
    max_sleeps = int(net_cfg.get("spread_max_sleeps", 6))

    # Ideal per-packet spacing if we slept once per packet
    ideal = frame_interval_s / float(pkt_count)

    # Start by grouping to satisfy minimum spacing
    group_n = 1
    if 0.0 < ideal < min_s:
        group_n = max(1, int(math.ceil(min_s / ideal)))

    # Then cap total sleeps per frame
    if max_sleeps > 0:
        per_sleep = max(1, int(math.ceil(pkt_count / max_sleeps)))
        group_n = max(group_n, per_sleep)

    spacing = ideal * group_n
    if spacing > frame_interval_s:
        spacing = frame_interval_s

    return (spacing, group_n)


def normalize_pixel_format(fmt_str: str) -> str:
    """Normalize pixel format string."""
    s = (fmt_str or "rgb888").strip().lower()
    if s in ("rgb565", "565"):  # default to LE if unspecified
        return "rgb565le"
    return s