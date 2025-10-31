# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import contextlib
import logging
import socket
import struct
from collections.abc import Callable, Iterator
from typing import ClassVar

from ..config import Config
from ..media.processing import rgb888_to_565_bytes
from ..utils.helpers import compute_spacing_and_group
from ..utils.metrics import PerformanceTracker
from .protocol import BufferedOutputProtocol, FrameMetadata, OutputProtocolFactory, OutputTarget


# Max payload per DDP packet. 1440 keeps UDP datagrams < 1500B MTU
# (IP+UDP+DDP header overhead), reducing fragmentation on typical links.
DDP_MAX_DATA = 1440

# DDP header layout (big-endian):
#   flags: 0x40 => header present, 0x01 => PUSH (end-of-frame)
#   seq:   0..255 sequence number (low 8 bits used)
#   cfg:   pixel config (data type field per DDP spec byte 2)
#          RGB888: 0x0B (bits C=0, R=0, TTT=001[RGB], SSS=011[8-bit])
#          EXT: 0x61 = RGB565(BE), 0x62 = RGB565(LE)
#   out_id: destination output/canvas id (0..255)
#   offset: byte offset within the frame buffer
#   length: payload bytes in this packet
DDP_HDR = struct.Struct("!BBB B I H")  # flags, seq, cfg, out_id, offset, length (network byte order)
DDP_PIXEL_CFG_RGB888 = 0x0B
DDP_PIXEL_CFG_RGB565_BE = 0x61
DDP_PIXEL_CFG_RGB565_LE = 0x62


class DDPSender(asyncio.DatagramProtocol):
    """UDP protocol for sending DDP packets."""

    def __init__(self) -> None:
        self.transport: asyncio.DatagramTransport | None = None
        self.packets_sent = 0

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]  # asyncio DatagramTransport compatibility

    def error_received(self, exc: BaseException) -> None:
        logging.getLogger("ddp").error(f"error_received: {exc!r}")

    def connection_lost(self, exc: BaseException | None) -> None:
        if exc:
            logging.getLogger("ddp").error(f"connection_lost: {exc!r}")

    def sendto(self, data: bytes, addr):
        if self.transport is not None:
            self.transport.sendto(data, addr)  # type: ignore[union-attr]  # transport is always DatagramTransport when initialized
            self.packets_sent += 1


def ddp_iter_packets(rgb_bytes: bytes, output_id: int, seq: int, *, fmt: str) -> Iterator[tuple[bytes, int]]:
    """Generate DDP packets for a frame with incrementing sequence numbers.

    Yields tuples of (packet_bytes, next_sequence_number).
    Sequence numbers wrap in range 1-15 per DDP spec.
    """

    if fmt == "rgb888":
        pixcfg = DDP_PIXEL_CFG_RGB888
        payload = memoryview(rgb_bytes)
    elif fmt == "rgb565le":
        pixcfg = DDP_PIXEL_CFG_RGB565_LE
        payload = memoryview(rgb888_to_565_bytes(rgb_bytes, "le"))
    elif fmt == "rgb565be":
        pixcfg = DDP_PIXEL_CFG_RGB565_BE
        payload = memoryview(rgb888_to_565_bytes(rgb_bytes, "be"))
    else:
        # Fallback to 888 if unknown token
        pixcfg = DDP_PIXEL_CFG_RGB888
        payload = memoryview(rgb_bytes)

    total = len(payload)
    off = 0
    push_mask = 0x01
    ddp_base_flags = 0x40  # header present (per DDP spec)
    current_seq = seq

    while off < total:
        end = min(off + DDP_MAX_DATA, total)
        chunk = payload[off:end]
        is_last = end >= total
        flags = ddp_base_flags | (push_mask if is_last else 0)
        payload_len = len(chunk)

        pkt = bytearray(DDP_HDR.size + payload_len)
        DDP_HDR.pack_into(pkt, 0, flags, current_seq, pixcfg, output_id & 0xFF, off, payload_len)
        pkt[DDP_HDR.size :] = chunk.tobytes()

        # Increment sequence for next packet (wrap 1-15 per DDP spec)
        next_seq = (current_seq % 15) + 1
        yield (bytes(pkt), next_seq)

        current_seq = next_seq
        off = end


class DDPOutput(BufferedOutputProtocol):
    """DDP output protocol implementation."""

    # Global stream registry - shared across all DDP instances for conflict resolution
    _global_streams: ClassVar[dict[tuple[str, int], asyncio.Task]] = {}  # (target_ip, out_id) -> task
    _stream_locks: ClassVar[dict[tuple[str, int], asyncio.Lock]] = {}  # Prevent race conditions

    def __init__(self, target: OutputTarget, stream_options):
        super().__init__(target, stream_options)

        self.sender: DDPSender  # Initialized in start()
        self.transport: asyncio.DatagramTransport | None = None
        self.socket: socket.socket | None = None
        self.seq = 1  # DDP spec: sequence 1-15 (0 = not used)

        # Performance tracking and logging configuration from app config
        app_config = Config()
        self.log_metrics = app_config.get("log.metrics")
        self.tracker = PerformanceTracker(log_interval_s=app_config.get("log.rate_ms") / 1000.0)

        # Frame format configuration
        self.pixel_format = stream_options.fmt

        # Packet spreading configuration
        self.spread_enabled = bool(app_config.get("net.spread_packets"))
        self.spread_max_fps = int(app_config.get("net.spread_max_fps"))

        # Still frame redundancy configuration
        self.still_redundancy = app_config.get("playback_still.redundancy")

        # Mode and pacing information for logging
        self.mode = "pace" if stream_options.pace > 0 else "native"
        self.pace_hz = stream_options.pace

        # Track target FPS for native mode
        self.last_delay_ms: float | None = None

        # Packet counters for logging
        self._packets_enqueued = 0

    async def create_and_register_stream(
        self, stream_key: tuple[str, int], task_factory: Callable[[], asyncio.Task]
    ) -> asyncio.Task:
        """Atomically ensure exclusive access, create task, and register stream.

        This method holds the lock during conflict resolution, task creation, and registration
        to prevent race conditions.

        Args:
            stream_key: Tuple of (target_ip, output_id)
            task_factory: Callable that creates and returns the stream task

        Returns:
            The created and registered task
        """
        target_ip, out_id = stream_key

        # Get or create lock for this stream key
        if stream_key not in DDPOutput._stream_locks:
            DDPOutput._stream_locks[stream_key] = asyncio.Lock()
            logging.getLogger("ddp").debug(f"created new lock for {target_ip}:{out_id}")

        lock = DDPOutput._stream_locks[stream_key]
        logging.getLogger("ddp").debug(f"acquiring lock for {target_ip}:{out_id}")
        async with lock:
            logging.getLogger("ddp").debug(f"lock acquired for {target_ip}:{out_id}, checking conflicts...")
            # Cancel any conflicting streams while holding lock
            await self._ensure_no_global_conflict_locked(stream_key)

            # Create task while holding lock (no conflicts exist)
            logging.getLogger("ddp").debug(f"creating new stream task for {target_ip}:{out_id}")
            task = task_factory()

            # Register stream while still holding lock
            DDPOutput._global_streams[stream_key] = task
            logging.getLogger("ddp").debug(f"registered new stream task for {target_ip}:{out_id}")
        logging.getLogger("ddp").debug(f"lock released for {target_ip}:{out_id}")

        return task

    async def cleanup_stream(self, stream_key: tuple[str, int], task: asyncio.Task) -> None:
        """Clean up DDP stream from global registry"""
        target_ip, out_id = stream_key
        # Only remove from global registry if we actually owned this stream
        global_task = DDPOutput._global_streams.get(stream_key)
        if global_task is task:
            DDPOutput._global_streams.pop(stream_key, None)
            logging.getLogger("ddp").debug(f"removed stream from global registry for {target_ip}:{out_id}")

    async def _ensure_no_global_conflict_locked(self, stream_key: tuple[str, int]) -> None:
        """Ensure no conflicting streams exist globally. Assumes lock is already held."""
        target_ip, out_id = stream_key

        # Check for existing global stream
        existing_task = DDPOutput._global_streams.get(stream_key)
        logging.getLogger("ddp").debug(
            f"global conflict check for {target_ip}:{out_id}: existing_task={existing_task is not None}, done={existing_task.done() if existing_task else 'N/A'}"
        )

        if existing_task and not existing_task.done():
            logging.getLogger("ddp").info(f"stopping conflicting stream for {target_ip}:{out_id}")
            existing_task.cancel()
            try:
                await existing_task
                logging.getLogger("ddp").debug(f"conflicting stream cancelled successfully for {target_ip}:{out_id}")
            except asyncio.CancelledError:
                logging.getLogger("ddp").debug(f"conflicting stream cancellation completed for {target_ip}:{out_id}")
            except Exception as e:
                logging.getLogger("ddp").error(f"global stream cleanup error {target_ip}:{out_id}: {e!r}")
            # Remove from global registry
            DDPOutput._global_streams.pop(stream_key, None)
            logging.getLogger("ddp").debug(f"removed conflicting stream from global registry for {target_ip}:{out_id}")
        else:
            logging.getLogger("ddp").debug(f"no conflicting stream found for {target_ip}:{out_id}")

    async def start(self) -> None:
        """Initialize UDP transport."""
        # Initialize sender and transport BEFORE starting worker loop to prevent race condition
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)

        sock.bind(("0.0.0.0", 0))

        self.sender = DDPSender()
        self.transport, _ = await loop.create_datagram_endpoint(  # type: ignore[type-var]  # asyncio transport/protocol tuple unpacking
            lambda: self.sender, sock=sock
        )
        self.socket = sock

        logging.getLogger("ddp").info(
            f"out={self.target.output_id} started output to {self.target.host}:{self.target.port}"
        )

        await super().start()

    async def stop(self) -> None:
        """Clean up UDP transport."""
        await super().stop()

        if self.transport:
            self.transport.close()
            self.transport = None

        if self.socket:
            with contextlib.suppress(Exception):
                self.socket.close()
            self.socket = None

        logging.getLogger("ddp").info(f"out={self.target.output_id} stopped output")

    async def flush_and_stop(self) -> None:
        """Stop DDP output and wait for complete queue drain."""
        # FIRST: Wait for all queued items to be processed completely
        await self._queue.join()

        # SECOND: Stop the buffered protocol (sets _running=False and cancels worker)
        await super().stop()

        # DDP-specific cleanup
        if self.transport:
            self.transport.close()
            self.transport = None

        if self.socket:
            with contextlib.suppress(Exception):
                self.socket.close()
            self.socket = None

        logging.getLogger("ddp").info(f"out={self.target.output_id} stopped output")

    async def _send_frame_internal(self, frame_data: bytes, metadata: FrameMetadata) -> None:
        """Send a single frame via DDP."""
        if not self.sender:
            return

        # Determine if packet spreading should be used
        spacing_s = None
        group_n = 1

        if self.spread_enabled:
            frame_rate = 1000.0 / max(metadata.delay_ms, 1.0)
            if frame_rate <= self.spread_max_fps:
                # Calculate effective payload length for spacing
                payload_len = len(frame_data) if self.pixel_format == "rgb888" else (len(frame_data) // 3) * 2

                pkt_count = (payload_len + DDP_MAX_DATA - 1) // DDP_MAX_DATA
                spacing_s, group_n = compute_spacing_and_group(pkt_count, metadata.delay_ms / 1000.0)

        # Send packets with optional spreading and redundancy (updates self.seq internally)
        await self._send_frame_packets(frame_data, metadata, spacing_s=spacing_s, group_n=group_n)

        # Update tracking
        self.tracker.record_frame()

        # Track delay for target FPS calculation
        self.last_delay_ms = metadata.delay_ms

        # Note: self.seq is already updated by _send_frame_packets()
        # Note: redundancy for still frames is handled inside _send_frame_packets()

        # Log metrics if needed
        # For single-frame/still content, log once per frame since they're rare
        # For looping content, respect the minimum 5-second interval
        if self.log_metrics and self.tracker.should_log(is_loop_end=metadata.is_last_frame):
            await self._log_metrics()

        # Track loop starts for very short content
        if metadata.is_last_frame:
            self.tracker.record_loop_start()

    async def _send_frame_packets(
        self, frame_data: bytes, metadata: FrameMetadata, *, spacing_s: float | None = None, group_n: int = 1
    ) -> None:
        """Send all packets for a frame with optional spreading and redundancy.

        For still frames, each packet is sent 'redundancy' times immediately
        before moving to the next packet (per DDP spec: duplicate packets for redundancy).
        """
        if not self.sender:
            return

        addr = (self.target.host, self.target.port)

        # Determine redundancy count for still frames
        redundancy = 1  # Default: send each packet once
        if metadata.is_still and metadata.is_last_frame:
            redundancy = self.still_redundancy
            if redundancy > 1:
                logging.getLogger("ddp").info(f"out={self.target.output_id} still frame redundancy={redundancy}")

        # Generate packets with incrementing sequences
        packet_iter = ddp_iter_packets(frame_data, self.target.output_id, self.seq, fmt=self.pixel_format)

        if not spacing_s or spacing_s <= 0:
            # Send all packets immediately
            for pkt, next_seq in packet_iter:
                # Send same packet 'redundancy' times before moving to next
                for i in range(redundancy):
                    self.sender.sendto(pkt, addr)
                    self.tracker.record_packet()  # Count all physical sends
                    self.tracker.record_physical_send(time_offset=i * 0.000001)  # Track with micro-offsets
                    self._packets_enqueued += 1
                self.tracker.record_unique_packet()  # Count unique packet once
                self.seq = next_seq
        else:
            # Send with spreading (apply spreading between unique packets, not duplicates)
            loop = asyncio.get_running_loop()
            start_time = loop.time()
            slot_idx = 0
            group_left = group_n

            for pkt, next_seq in packet_iter:
                # Send same packet 'redundancy' times with no delay between duplicates
                for i in range(redundancy):
                    self.sender.sendto(pkt, addr)
                    self.tracker.record_packet()  # Count all physical sends
                    self.tracker.record_physical_send(time_offset=i * 0.000001)  # Track with micro-offsets
                    self._packets_enqueued += 1
                self.tracker.record_unique_packet()  # Count unique packet once
                self.seq = next_seq

                # Apply spreading delay only between unique packets
                group_left -= 1
                if group_left <= 0:
                    slot_idx += 1
                    target_time = start_time + slot_idx * spacing_s
                    sleep_time = max(0.0, target_time - loop.time())
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    group_left = group_n

    async def _log_metrics(self) -> None:
        """Log performance metrics."""
        metrics = self.tracker.get_metrics_and_reset()

        spread_tag = " (spread)" if self.spread_enabled else ""

        # Format pps: show unique, with physical and redundancy multiplier if different
        unique_pps = metrics["unique_pps"]
        physical_pps = metrics["physical_pps"]
        jitter_ms = metrics["unique_packet_jitter_ms"]  # Always use unique packet jitter

        if abs(unique_pps - physical_pps) > 0.5:  # Redundancy is active
            redundancy_factor = physical_pps / unique_pps if unique_pps > 0 else 1.0
            pps_str = f"pps={unique_pps:.0f} ({physical_pps:.0f}phy, {redundancy_factor:.1f}x)"
        else:
            pps_str = f"pps={unique_pps:.0f}"

        if self.mode == "pace":
            # Paced mode logging
            logging.getLogger("ddp").info(
                f"out={self.target.output_id} pace={self.pace_hz}Hz "
                f"fps={metrics['fps']:.2f} {pps_str} "
                f"pkt_jit={jitter_ms:.1f}ms frm_jit={metrics['frame_jitter_ms']:.1f}ms "
                f"q_avg={metrics['queue_avg']:.0f}/{self._queue.maxsize} "
                f"q_max={metrics['queue_max']} "
                f"enq={self._packets_enqueued} tx={self.sender.packets_sent if self.sender else 0} "
                f"drops={metrics['queue_drops']}{spread_tag}"
            )
        else:
            # Native mode logging with target FPS
            target_fps = 1000.0 / max(self.last_delay_ms or 33.33, 1.0)
            logging.getLogger("ddp").info(
                f"out={self.target.output_id} native "
                f"fps={metrics['fps']:.2f} (~{target_fps:.1f} tgt) {pps_str} "
                f"pkt_jit={jitter_ms:.1f}ms frm_jit={metrics['frame_jitter_ms']:.1f}ms "
                f"q_avg={metrics['queue_avg']:.0f}/{self._queue.maxsize} "
                f"q_max={metrics['queue_max']} "
                f"enq={self._packets_enqueued} tx={self.sender.packets_sent if self.sender else 0} "
                f"drops={metrics['queue_drops']}{spread_tag}"
            )


# Register the DDP protocol
OutputProtocolFactory.register("ddp", DDPOutput)
