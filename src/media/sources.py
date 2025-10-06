# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote, urlparse, urlunparse

import yt_dlp

from ..config import Config
from ..utils.helpers import is_youtube_url, headers_dict_to_ffmpeg_opt


def is_internal_url(url: str) -> bool:
    """Check if URL uses internal: protocol"""
    return url.startswith("internal:")


def rewrite_internal_url(url: str, server_host: str) -> str:
    """Rewrite internal: URL to localhost HTTP URL using urllib

    Args:
        url: internal: URL to rewrite (e.g., internal:placeholder/64x64)
        server_host: Host from request.host (includes port, e.g., '192.168.1.1:8788')

    Examples:
        internal:placeholder/64x64 -> http://192.168.1.1:8788/api/internal/placeholder/64x64
        internal:placeholder/64x64/ff0000?text=Hi -> http://192.168.1.1:8788/api/internal/placeholder/64x64/ff0000?text=Hi
    """
    parsed = urlparse(url)
    new_path = f"/api/internal/{parsed.path}"

    rewritten = urlunparse((
        'http',
        server_host,
        new_path,
        '',
        parsed.query,
        ''
    ))

    return rewritten


def build_yt_dlp_format(W: int, H: int, mode: Optional[str] = None, video_only: bool = True) -> Tuple[str, None]:
    """
    Build optimized yt-dlp format selector based on target resolution and hardware.

    Args:
        W, H: Target minimum resolution (width, height)
        mode: Hardware acceleration mode (None, "vaapi", "qsv", "cuda", "videotoolbox", "d3d11va")
        video_only: Prefer video-only streams, fallback to combined if needed

    Returns:
        (format_expr, None) tuple - ordering in format_expr handles priority
    """
    # Hardware-optimized codec preferences
    codec_preferences = {
        "vaapi": ["av01", "vp9", "vp09", "h265", "hevc", "hev1", "h264", "avc1", "avc3"],  # Intel/AMD Linux
        "qsv": ["h265", "hevc", "hev1", "h264", "avc1", "avc3", "av01", "vp9"],          # Intel Quick Sync
        "cuda": ["av01", "h265", "hevc", "hev1", "h264", "avc1", "avc3", "vp9"],         # NVIDIA NVDEC (RTX 30+ has AV1 decode)
        "videotoolbox": ["h264", "avc1", "avc3", "h265", "hevc", "hev1", "av01", "vp9"], # macOS
        "d3d11va": ["h264", "avc1", "avc3", "h265", "hevc", "hev1", "av01", "vp9"],      # Windows D3D11
        None: ["h264", "avc1", "avc3", "vp9", "vp09", "h265", "hevc", "hev1", "av01"]    # CPU fallback
    }

    codecs = codec_preferences.get(mode, codec_preferences[None])
    vcodec_regex = "^(" + "|".join(codecs) + ")$"

    # Calculate reasonable max resolution to avoid extreme over-fetching
    max_H = min(H * 4, 1080)  # Cap at 1080p unless display is >270px tall

    # For very small displays, be much more aggressive about resolution capping
    if H <= 64:  # For displays 64px or smaller (like 64x64)
        max_H = min(max_H, 480)  # Cap at 480p max for tiny displays
    elif H <= 128:  # For displays 65-128px
        max_H = min(max_H, 720)  # Cap at 720p

    # Determine optimal resolution tiers based on target size
    # Start with closest match to target, then progressively larger
    resolutions = []
    if H <= 144:
        resolutions = [144, 240, 360, 480]  # For very small displays, start with 144p
    elif H <= 240:
        resolutions = [240, 144, 360, 480, 720]
    elif H <= 360:
        resolutions = [360, 240, 480, 720, 1080]
    elif H <= 480:
        resolutions = [480, 360, 240, 720, 1080]  # Include smaller fallbacks
    elif H <= 720:
        resolutions = [720, 1080, 480, 360, 240]  # Prefer going up to 1080p before falling back
    else:
        resolutions = [1080, 720, 480, 360, 240, 144]  # Include all smaller fallbacks

    # Filter resolutions that exceed our max_H cap
    resolutions = [r for r in resolutions if r <= max_H]

    # Ensure we have at least one fallback
    if not resolutions:
        resolutions = [240]

    components = []

    # Check config for 60fps preference
    config = Config()
    try_60fps = config.get("youtube.60fps")

    # First, try 60fps at 720p regardless of target size (since 60fps is only available at 720p+)
    # IMPORTANT: Only allow https protocol - avoid playlists (m3u8, dash) and streaming protocols
    if try_60fps:
        for codec in codecs:
            components.append(f'bv*[fps>=60][vcodec*={codec}][height>=720][height<=720][protocol=https]')
            if not video_only:
                components.append(f'b[fps>=60][vcodec*={codec}][height>=720][height<=720][protocol=https]')

    # Build codec-specific selectors for each resolution
    # This ensures explicit codec priority ordering
    # IMPORTANT: Only allow https protocol - avoid playlists (m3u8, dash) and streaming protocols
    for res in resolutions:
        # For each codec in preference order, add specific selectors
        for i, codec in enumerate(codecs):
            # Then any fps - prioritize video-only
            components.append(f'bv*[vcodec*={codec}][height>={res}][height<={res}][protocol=https]')
            if not video_only:
                components.append(f'b[vcodec*={codec}][height>={res}][height<={res}][protocol=https]')

    # Fallback to any codec at each resolution - prioritize video-only
    for res in resolutions:
        components.append(f'bv*[height>={res}][height<={res}][protocol=https]')
        if not video_only:
            components.append(f'b[height>={res}][height<={res}][protocol=https]')

    # Final fallbacks for edge cases - prioritize video-only
    components.append(f'bv*[vcodec~="{vcodec_regex}"][height>={H}][protocol=https]')  # Preferred codec, minimum height
    components.append(f'bv*[height>={H}][protocol=https]')  # Any codec, minimum height
    components.append(f'bv*[vcodec~="{vcodec_regex}"][protocol=https]')  # Preferred codec, any resolution
    components.append('bv*[protocol=https]')  # Any video-only stream (https only)

    if not video_only:
        components.append(f'b[vcodec~="{vcodec_regex}"][height>={H}][protocol=https]')  # Preferred codec, minimum height
        components.append(f'b[height>={H}][protocol=https]')  # Any codec, minimum height
        components.append(f'b[vcodec~="{vcodec_regex}"][protocol=https]')  # Preferred codec, any resolution
        components.append('b[protocol=https]')  # Any combined stream (https only)

    format_expr = "/".join(components)

    # Don't use format_sort - let the format selector handle priority via ordering
    return format_expr, None



class MediaUnavailableError(Exception):
    """Raised when a media source is temporarily unavailable (HTTP errors, network issues, etc.)."""
    def __init__(self, message: str, url: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.url = url
        self.original_error = original_error


async def resolve_stream_url_async(src_url: str, target_size: Tuple[int, int], hw_mode: Optional[str] = None) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """
    Async version of YouTube URL resolution.
    Resolve YouTube (and similar) page URLs into a direct media URL + HTTP headers
    suitable for av.open(..., options={ 'headers': 'K: V\\r\\n...' }).
    Non-YouTube URLs are returned unchanged.

    Args:
        src_url: Source URL to resolve
        target_size: Target (width, height) for optimal format selection
        hw_mode: Hardware acceleration mode for codec preference

    Returns:
        Tuple of (media_url, http_options, info_dict)
    """
    if not is_youtube_url(src_url):
        return src_url, {}, {}

    logger = logging.getLogger('sources')

    # Build optimized format selector based on target size and hardware
    if not target_size or len(target_size) != 2:
        raise ValueError("target_size must be provided as (width, height) tuple")

    format_expr, _ = build_yt_dlp_format(target_size[0], target_size[1], hw_mode, video_only=True)
    logger.info(f"YouTube format selection for {target_size[0]}x{target_size[1]}, hw={hw_mode}")
    logger.debug(f"Format expression: {format_expr}")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "extract_flat": False,
        "format": format_expr,
    }

    # Debug: List available formats if debug logging is enabled
    if logger.isEnabledFor(logging.DEBUG) and target_size:
        logger.debug("Listing available formats for debugging...")
        debug_opts = {
            "quiet": False,
            "listformats": True,
        }
        try:
            with yt_dlp.YoutubeDL(debug_opts) as debug_ydl:  # type: ignore[arg-type]
                debug_ydl.extract_info(src_url, download=False)
        except Exception:
            pass  # Don't fail if debug listing fails

    # Run the blocking yt-dlp operation in a thread pool
    loop = asyncio.get_event_loop()
    
    def _extract_info():

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # type: ignore[arg-type]
            info = ydl.extract_info(src_url, download=False)
            if info is None:
                return src_url, {}, {}
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            media_url = info.get("url") or src_url

            # Log selected format details for debugging
            if target_size:
                format_id = info.get("format_id", "unknown")
                resolution = f"{info.get('width', '?')}x{info.get('height', '?')}"
                fps = info.get("fps", "?")
                vcodec = info.get("vcodec", "?")
                filesize = info.get("filesize") or info.get("filesize_approx")
                size_mb = f"{filesize / 1024 / 1024:.1f}MB" if filesize else "?MB"
                logger.info(f"Selected format {format_id}: {resolution} @ {fps}fps, {vcodec}, ~{size_mb}")

                # Show codec optimization results if debug logging enabled
                if logger.isEnabledFor(logging.DEBUG):
                    if "formats" in info and isinstance(info["formats"], list):
                        matching_res = [f for f in info["formats"]
                                      if f.get("height") == info.get("height")]
                        if matching_res:
                            logger.debug(f"Alternative codecs at {info.get('height')}p:")
                            for f in matching_res[:3]:
                                codec = f.get("vcodec", "unknown")
                                logger.debug(f"  {f.get('format_id')}: {codec}")

                        av1_available = any(f.get("vcodec", "").find("av01") >= 0
                                          for f in info["formats"])
                        logger.debug(f"AV1 available: {av1_available}, Selected: {vcodec}")

            headers = {}
            rh = info.get("http_headers") or {}
            if not rh and "formats" in info and isinstance(info["formats"], list):
                for f in info["formats"]:
                    if f.get("url") == media_url and f.get("http_headers"):
                        rh = f["http_headers"]
                        break
            if rh:
                headers_str = headers_dict_to_ffmpeg_opt({k: v for k, v in rh.items()})
                if headers_str:
                    headers["headers"] = headers_str
            return media_url, headers, info
    
    try:
        # Run in thread pool to avoid blocking the event loop
        return await loop.run_in_executor(None, _extract_info)  # type: ignore[return-value]
    except Exception as e:
        logger = logging.getLogger('sources')
        if isinstance(e, yt_dlp.DownloadError):  # type: ignore[attr-defined]
            # Re-raise DownloadError for proper handling upstream
            logger.warning(f"URL resolution failed for {src_url}: {e!r}")
            raise
        else:
            # Other exceptions - log and return original URL
            logger.warning(f"URL resolution failed for {src_url}: {e!r}")
            return src_url, {}, {}


class MediaSource:
    """Represents a resolved media source."""

    def __init__(self, original_url: str, resolved_url: str, options: Optional[Dict[str, str]] = None, info: Optional[Dict[str, Any]] = None):
        self.original_url = original_url
        self.resolved_url = resolved_url
        self.options = options or {}
        self.info = info or {}  # yt-dlp metadata
        self.is_youtube = is_youtube_url(original_url)

    def __repr__(self):
        return f"MediaSource(original={self.original_url!r}, resolved={self.resolved_url!r})"

    def should_enable_cache(self, loop: bool) -> bool:
        """Determine if FFmpeg cache: protocol should be enabled for this source.

        Enables caching when:
        - YouTube source (benefits from caching)
        - Config enables caching
        - Content will loop (reuse cache multiple times)
        - Video size under threshold

        Args:
            loop: Whether content will loop

        Returns:
            True if cache should be enabled, False otherwise
        """
        if not self.is_youtube:
            return False

        config = Config()
        if not config.get('youtube.cache.enabled'):
            return False

        if not loop:
            return False

        filesize = self.info.get('filesize') or self.info.get('filesize_approx')
        if not filesize:
            return False

        max_size = config.get('youtube.cache.max_size')
        return filesize < max_size


async def resolve_media_source(src: str, stream_options) -> MediaSource:
    """Resolve a media source URL to a MediaSource object."""
    src_url = unquote(src)
    resolved_url, options, info = await resolve_stream_url_async(src_url, stream_options.size, stream_options.hw)
    return MediaSource(src_url, resolved_url, options, info)