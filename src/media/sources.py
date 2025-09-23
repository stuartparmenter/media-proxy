# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
from typing import Dict, Optional, Tuple
from urllib.parse import unquote

import yt_dlp

from ..utils.helpers import is_youtube_url, headers_dict_to_ffmpeg_opt



class MediaUnavailableError(Exception):
    """Raised when a media source is temporarily unavailable (HTTP errors, network issues, etc.)."""
    def __init__(self, message: str, url: str, original_error: Exception = None):
        super().__init__(message)
        self.url = url
        self.original_error = original_error


async def resolve_stream_url_async(src_url: str) -> Tuple[str, Dict[str, str]]:
    """
    Async version of YouTube URL resolution.
    Resolve YouTube (and similar) page URLs into a direct media URL + HTTP headers
    suitable for av.open(..., options={ 'headers': 'K: V\\r\\n...' }).
    Non-YouTube URLs are returned unchanged.
    """
    if not is_youtube_url(src_url):
        return src_url, {}

    # Accept any available format - FFmpeg can process almost anything
    # Prefer video-only streams up to 1080p, fallback to combined streams
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "extract_flat": False,
        "format": (
            "bv[height<=720][vcodec!=none]/"
            "best[height<=720][vcodec!=none]/"
            "bv[vcodec!=none]/best[vcodec!=none]"
        ),
    }

    # Run the blocking yt-dlp operation in a thread pool
    loop = asyncio.get_event_loop()
    
    def _extract_info():
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(src_url, download=False)
            if info is None:
                return src_url, {}
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            media_url = info.get("url") or src_url
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
            return media_url, headers
    
    try:
        # Run in thread pool to avoid blocking the event loop
        return await loop.run_in_executor(None, _extract_info)
    except Exception as e:
        if isinstance(e, yt_dlp.DownloadError):
            # Re-raise DownloadError for proper handling upstream
            print(f"[warn] YouTube URL resolution failed for {src_url}: {e!r}")
            raise
        else:
            # Other exceptions - log and return original URL
            print(f"[warn] YouTube URL resolution failed for {src_url}: {e!r}")
            return src_url, {}


class MediaSource:
    """Represents a resolved media source."""
    
    def __init__(self, original_url: str, resolved_url: str, options: Dict[str, str] = None):
        self.original_url = original_url
        self.resolved_url = resolved_url
        self.options = options or {}
        self.is_youtube = is_youtube_url(original_url)
        
    def __repr__(self):
        return f"MediaSource(original={self.original_url!r}, resolved={self.resolved_url!r})"


async def resolve_media_source(src: str) -> MediaSource:
    """Resolve a media source URL to a MediaSource object."""
    src_url = unquote(src)
    resolved_url, options = await resolve_stream_url_async(src_url)
    return MediaSource(src_url, resolved_url, options)