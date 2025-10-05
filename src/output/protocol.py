# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from contextlib import asynccontextmanager
import asyncio
import logging
import time

from ..streaming.options import StreamOptions


@dataclass
class FrameMetadata:
    """Metadata associated with a frame."""
    sequence: int
    timestamp_ms: float
    delay_ms: float
    size: Tuple[int, int]  # (width, height)
    format: str  # "rgb888", "rgb565le", etc.
    is_still: bool = False
    is_last_frame: bool = False


@dataclass
class OutputTarget:
    """Represents an output destination."""
    host: str
    port: int
    output_id: int
    protocol: str  # "ddp", "mjpeg", "h264", etc.
    

@dataclass
class OutputMetrics:
    """Performance metrics for an output stream."""
    frames_sent: int = 0
    packets_sent: int = 0
    bytes_sent: int = 0
    queue_drops: int = 0
    last_fps: float = 0.0
    last_pps: float = 0.0  # packets per second
    avg_queue_size: float = 0.0
    max_queue_size: int = 0
    last_update: float = 0.0
    
    def reset(self):
        self.frames_sent = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.queue_drops = 0
        self.last_update = time.perf_counter()


class OutputProtocol(ABC):
    """Abstract base class for output protocols (DDP, MJPEG, H264, etc.)."""

    def __init__(self, target: OutputTarget, stream_options: StreamOptions):
        self.target = target
        self.stream_options = stream_options
        self.metrics = OutputMetrics()
        self._running = False

    @abstractmethod
    async def start(self) -> None:
        """Initialize the output protocol and start any background tasks."""
        self._running = True

    @abstractmethod
    async def stop(self) -> None:
        """Stop the output protocol and clean up resources."""
        self._running = False

    @abstractmethod
    async def send_frame(self, frame_data: bytes, metadata: FrameMetadata) -> None:
        """Send a frame to the output destination."""
        pass

    # Stream management interface (optional for protocols that need conflict resolution)
    def get_stream_key(self, session, params: Dict[str, Any]) -> Any:
        """Generate stream key for conflict detection. Return None if no conflicts possible."""
        return None  # Default: no conflicts

    async def create_and_register_stream(self, stream_key: Any, task_factory: Callable[[], 'asyncio.Task']) -> 'asyncio.Task':
        """Atomically ensure exclusive access, create task, and register stream.

        Args:
            stream_key: Key identifying the stream target
            task_factory: Callable that creates and returns the stream task

        Returns:
            The created task

        Default implementation: just create the task (no conflict resolution).
        """
        return task_factory()

    async def cleanup_stream(self, stream_key: Any, task: 'asyncio.Task') -> None:
        """Clean up stream registration. Default: no-op."""
        pass
        
        
    @property
    def is_running(self) -> bool:
        """Check if the output protocol is currently running."""
        return self._running
        
    def get_metrics(self) -> OutputMetrics:
        """Get current performance metrics."""
        return self.metrics
        
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics.reset()


class BufferedOutputProtocol(OutputProtocol):
    """Base class for output protocols that use internal buffering/queuing."""
    
    def __init__(self, target: OutputTarget, stream_options: StreamOptions):
        super().__init__(target, stream_options)
        self._queue: asyncio.Queue[Tuple[bytes, FrameMetadata]] = asyncio.Queue(maxsize=4096)
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the buffered output with worker task."""
        await super().start()
        self._worker_task = asyncio.create_task(self._worker_loop())
        
    async def stop(self) -> None:
        """Stop the buffered output and clean up."""
        await super().stop()

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.getLogger('output').error(f"worker cleanup error: {e!r}")

        # Try to flush remaining items quickly
        try:
            await asyncio.wait_for(self._queue.join(), timeout=0.5)
        except asyncio.TimeoutError:
            pass

    async def flush_and_stop(self) -> None:
        """Stop the buffered output and wait for complete queue drain."""
        # Wait for all queued items to be processed completely
        await self._queue.join()

        # Now stop the protocol and clean up
        await super().stop()

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.getLogger('output').error(f"worker cleanup error: {e!r}")
                
    async def send_frame(self, frame_data: bytes, metadata: FrameMetadata) -> None:
        """Queue a frame for sending."""
        if not self._running:
            return
            
        if self._queue.full():
            # Drop oldest frame to make room
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self.metrics.queue_drops += 1
            except asyncio.QueueEmpty:
                pass
                
        await self._queue.put((frame_data, metadata))
        
    @abstractmethod
    async def _send_frame_internal(self, frame_data: bytes, metadata: FrameMetadata) -> None:
        """Internal frame sending implementation."""
        pass
        
    async def _worker_loop(self) -> None:
        """Worker loop that processes the frame queue."""
        try:
            while self._running:
                frame_data, metadata = await self._queue.get()
                try:
                    await self._send_frame_internal(frame_data, metadata)
                    self.metrics.frames_sent += 1
                    self.metrics.bytes_sent += len(frame_data)
                except Exception as e:
                    logging.getLogger('output').error(f"frame send error: {e!r}")
                finally:
                    self._queue.task_done()
                    
                # Update queue metrics
                current_size = self._queue.qsize()
                self.metrics.max_queue_size = max(self.metrics.max_queue_size, current_size)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.getLogger('output').error(f"worker loop error: {e!r}")




class OutputProtocolFactory:
    """Factory for creating output protocol instances."""
    
    _protocols: Dict[str, type] = {}
    
    @classmethod
    def register(cls, protocol_name: str, protocol_class: type) -> None:
        """Register a new output protocol."""
        cls._protocols[protocol_name] = protocol_class
        
    @classmethod
    def create(cls, target: OutputTarget, stream_options: StreamOptions) -> OutputProtocol:
        """Create an output protocol instance."""
        protocol_class = cls._protocols.get(target.protocol)
        if not protocol_class:
            raise ValueError(f"Unknown output protocol: {target.protocol}")
            
        return protocol_class(target, stream_options)
        
    @classmethod
    def list_protocols(cls) -> list[str]:
        """List available protocol names."""
        return list(cls._protocols.keys())