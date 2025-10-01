# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""
Shared caching infrastructure for media sources.

Provides common cache management for both images and videos:
- TempCachedFile: Cached media in temporary files
- LocalCachedFile: Direct access to local files
- Global cleanup tracking with weakref
"""

import logging
import os
import weakref
import atexit
from abc import ABC, abstractmethod
from typing import Any
from weakref import WeakSet


# Global registry for cleanup tracking
_active_sources: WeakSet[Any] = weakref.WeakSet()


def _cleanup_all_sources():
    """Emergency cleanup function called on exit."""
    for source in list(_active_sources):
        try:
            source.cleanup()
        except Exception:
            pass


# Register emergency cleanup
atexit.register(_cleanup_all_sources)


def cleanup_active_sources():
    """Clean up all active sources (called when streams stop)."""
    active_count = len(_active_sources)
    if active_count > 0:
        logging.getLogger('cache').debug(f"Cleaning up {active_count} active sources")
        _cleanup_all_sources()


class CachedMediaFile(ABC):
    """Abstract base for cached media files."""

    @abstractmethod
    def get_path(self) -> str:
        """Get the file path for this media source."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources."""
        pass


class TempCachedFile(CachedMediaFile):
    """Cached media file stored in a temporary file."""

    def __init__(self, temp_path: str):
        self.temp_path = temp_path
        self._cleaned = False
        # Register for cleanup tracking
        _active_sources.add(self)

    def get_path(self) -> str:
        if self._cleaned:
            raise RuntimeError("Attempted to use cleaned up CachedMediaFile")
        return self.temp_path

    def cleanup(self) -> None:
        if self._cleaned:
            return

        try:
            if os.path.exists(self.temp_path):
                os.unlink(self.temp_path)
                logging.getLogger('cache').debug(f"Cleaned up temp file: {os.path.basename(self.temp_path)}")
        except Exception as e:
            logging.getLogger('cache').warning(f"Temp file cleanup warning: {e}")
        finally:
            self._cleaned = True
            # Remove from tracking (weakref will handle this automatically too)
            try:
                _active_sources.discard(self)
            except:
                pass


class LocalCachedFile(CachedMediaFile):
    """Cached media file that directly references a local file without copying."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._cleaned = False
        # Register for cleanup tracking
        _active_sources.add(self)

    def get_path(self) -> str:
        if self._cleaned:
            raise RuntimeError("Attempted to use cleaned up CachedMediaFile")
        return self.file_path

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        # Remove from tracking (weakref will handle this automatically too)
        try:
            _active_sources.discard(self)
        except:
            pass
