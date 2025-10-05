# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..streaming.options import StreamOptions


class FrameIterator(ABC):
    """Abstract base class for frame iteration from various media sources.

    All iterators must implement async_init() for non-blocking resource initialization.
    """

    def __init__(self, src_url: str, stream_options: 'StreamOptions'):
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
    def __iter__(self) -> Iterator[Tuple[bytes, float]]:
        """Iterate over frames, yielding (rgb888_bytes, delay_ms) tuples."""
        pass

    @classmethod
    @abstractmethod
    def can_handle(cls, src_url: str) -> bool:
        """Check if this iterator can handle the given source URL."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources used by the iterator."""
        pass


class FrameIteratorFactory:
    """Factory for creating appropriate frame iterators based on source type."""

    _iterators: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, iterator_class: type) -> None:
        """Register a frame iterator implementation."""
        cls._iterators[name] = iterator_class

    @classmethod
    async def create(cls, src_url: str, stream_options: 'StreamOptions',
                     resolved_url: Optional[str] = None, http_opts: Optional[dict] = None) -> FrameIterator:
        """Create and initialize the appropriate frame iterator for the given source.

        Args:
            src_url: Source URL to create iterator for
            stream_options: Streaming configuration options
            resolved_url: Optional resolved URL (for YouTube, etc.)
            http_opts: Optional HTTP options for the iterator
        """
        _ensure_iterators_registered()

        for name, iterator_class in cls._iterators.items():
            if iterator_class.can_handle(src_url):  # type: ignore[attr-defined]  # can_handle is a class method on all registered iterators
                iterator = iterator_class(src_url, stream_options)

                # Set optional properties before initialization (for PyAV iterators)
                if resolved_url and hasattr(iterator, 'real_src_url'):
                    iterator.real_src_url = resolved_url
                if http_opts and hasattr(iterator, 'http_opts'):
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