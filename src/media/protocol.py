# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Iterator, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FrameIteratorConfig:
    """Configuration for frame iteration."""
    size: Tuple[int, int]  # (width, height)
    loop_video: bool = False
    expand_mode: int = 0  # 0=auto, 1=auto_to_pc, 2=tv_to_pc
    hw_prefer: Optional[str] = None  # Hardware acceleration preference


class FrameIterator(ABC):
    """Abstract base class for frame iteration from various media sources."""

    def __init__(self, src_url: str, config: FrameIteratorConfig):
        self.src_url = src_url
        self.config = config

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
    def create(cls, src_url: str, config: FrameIteratorConfig) -> FrameIterator:
        """Create the appropriate frame iterator for the given source."""
        _ensure_iterators_registered()

        for name, iterator_class in cls._iterators.items():
            # Check compatibility without creating an instance
            if iterator_class.can_handle(src_url):
                return iterator_class(src_url, config)

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