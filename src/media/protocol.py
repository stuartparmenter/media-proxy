# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, ClassVar

import aiohttp


if TYPE_CHECKING:
    from ..streaming.options import StreamOptions


class FrameIterator(ABC):
    """Abstract base class for frame iteration from various media sources.

    All iterators must implement async_init() for non-blocking resource initialization.
    """

    def __init__(self, src_url: str, stream_options: "StreamOptions"):
        self.src_url = src_url
        self.stream_options = stream_options

    @abstractmethod
    async def async_init(self):
        """Initialize async resources (HTTP fetches, file I/O, etc.).

        This method is called by the factory before the iterator is returned.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[bytes, float]]:
        """Iterate over frames, yielding (rgb888_bytes, delay_ms) tuples."""
        pass

    @classmethod
    @abstractmethod
    def can_handle(cls, src_url: str, content_type: str | None = None) -> bool:
        """Check if this iterator can handle the given source URL.

        Args:
            src_url: Source URL to check
            content_type: Optional Content-Type header from HTTP HEAD request
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the iterator."""
        pass


class FrameIteratorFactory:
    """Factory for creating appropriate frame iterators based on source type."""

    _iterators: ClassVar[dict[str, type[FrameIterator]]] = {}

    @classmethod
    def register(cls, name: str, iterator_class: type[FrameIterator]) -> None:
        """Register a frame iterator implementation."""
        cls._iterators[name] = iterator_class

    @classmethod
    async def _detect_content_type(cls, url: str) -> str | None:
        """Detect content type of URL via HEAD request (HTTP/HTTPS) or file magic (file:// or local path).

        Args:
            url: URL to check (HTTP/HTTPS/file:// or local file path)

        Returns:
            Content-Type string or None if detection fails
        """
        from pathlib import Path
        from urllib.parse import urlparse

        from ..utils.helpers import resolve_local_path

        # Parse URL to get scheme
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()

        # HTTP/HTTPS: Use HEAD request
        if scheme in ("http", "https"):
            try:
                async with (
                    aiohttp.ClientSession() as session,
                    session.head(url, timeout=aiohttp.ClientTimeout(total=5), allow_redirects=True) as resp,
                ):
                    return resp.headers.get("Content-Type")
            except Exception:
                return None

        # file:// URLs or no scheme (local paths): Use file magic detection
        if scheme in ("file", ""):
            file_path = resolve_local_path(url)
            # If resolve_local_path returns None, check if url is a plain file path
            if not file_path and Path(url).exists():
                file_path = url

            if file_path:
                try:
                    import filetype  # type: ignore[import-untyped]

                    # Detect file type from first 261 bytes
                    kind = filetype.guess(file_path)
                    if kind is not None:
                        return kind.mime
                except Exception:
                    return None

        return None

    @classmethod
    async def create(
        cls,
        src_url: str,
        stream_options: "StreamOptions",
        resolved_url: str | None = None,
        http_opts: dict | None = None,
    ) -> FrameIterator:
        """Create and initialize the appropriate frame iterator for the given source.

        Args:
            src_url: Source URL to create iterator for
            stream_options: Streaming configuration options
            resolved_url: Optional resolved URL (for YouTube, etc.)
            http_opts: Optional HTTP options for the iterator
        """
        _ensure_iterators_registered()

        # For HTTP/HTTPS URLs, try to detect content type
        content_type = await cls._detect_content_type(src_url)

        for _name, iterator_class in cls._iterators.items():
            if iterator_class.can_handle(src_url, content_type):
                iterator = iterator_class(src_url, stream_options)

                # Set optional properties before initialization (for PyAV video iterators)
                from ..media.video import PyAvFrameIterator

                if isinstance(iterator, PyAvFrameIterator):
                    if resolved_url:
                        iterator.real_src_url = resolved_url
                    if http_opts:
                        iterator.http_opts = http_opts

                # Initialize the iterator
                await iterator.async_init()
                return iterator

        raise ValueError(f"No frame iterator available for source: {src_url}")

    @classmethod
    def list_iterators(cls) -> list[str]:
        """List available iterator names."""
        _ensure_iterators_registered()
        return list(cls._iterators.keys())


# Lazy registration of iterators (done when factory is first used)
def _ensure_iterators_registered():
    """Ensure default frame iterators are registered."""
    if FrameIteratorFactory._iterators:
        return  # Already registered

    try:
        from .images import PilFrameIterator

        FrameIteratorFactory.register("pil", PilFrameIterator)
    except ImportError:
        pass

    try:
        from .video import PyAvFrameIterator

        FrameIteratorFactory.register("pyav", PyAvFrameIterator)
    except ImportError:
        pass
