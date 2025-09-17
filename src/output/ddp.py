# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import socket
import struct
import time
from collections import deque
from typing import Dict, Any, Optional, Iterator

from .protocol import BufferedOutputProtocol, OutputTarget, FrameMetadata, OutputProtocolFactory
from ..media.processing import rgb888_to_565_bytes
from ..utils.helpers import normalize_pixel_format, compute_spacing_and_group
from ..utils.metrics import PerformanceTracker
from ..config import Config


# Max payload per DDP packet. 1440 keeps UDP datagrams < 1500B MTU
# (IP+UDP+DDP header overhead), reducing fragmentation on typical links.
DDP_MAX_DATA = 1440

# DDP header layout (big-endian):
#   flags: 0x40 => header present, 0x01 => PUSH (end-of-frame)
#   seq:   0..255 sequence number (low 8 bits used)
#   cfg:   pixel config; 0x2C = RGB888 (per 3waylabs DDP spec)
#          EXT: 0x61 = RGB565(BE), 0x62 = RGB565(LE)
#   out_id: destination output/canvas id (0..255)
#   offset: byte offset within the frame buffer
#   length: payload bytes in this packet
DDP_HDR = struct.Struct("!BBB B I H")  # flags, seq, cfg, out_id, offset, length (network byte order)
DDP_PIXEL_CFG_RGB888 = 0x2C
DDP_PIXEL_CFG_RGB565_BE = 0x61
DDP_PIXEL_CFG_RGB565_LE = 0x62


class DDPSender(asyncio.DatagramProtocol):
    """UDP protocol for sending DDP packets."""
    
    def __init__(self):
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.packets_sent = 0

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = transport  # type: ignore[assignment]

    def error_received(self, exc: BaseException) -> None:
        print(f"[udp] error_received: {exc!r}")

    def connection_lost(self, exc: BaseException | None) -> None:
        if exc:
            print(f"[udp] connection_lost: {exc!r}")

    def sendto(self, data: bytes, addr):
        if self.transport is not None:
            self.transport.sendto(data, addr)  # type: ignore[union-attr]
            self.packets_sent += 1


def ddp_iter_packets(rgb_bytes: bytes, output_id: int, seq: int, *, fmt: str = "rgb888") -> Iterator[bytes]:
    """Generate DDP packets for a frame."""
    fmt = normalize_pixel_format(fmt)
    
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
    
    while off < total:
        end = min(off + DDP_MAX_DATA, total)
        chunk = payload[off:end]
        is_last = end >= total
        flags = ddp_base_flags | (push_mask if is_last else 0)
        payload_len = len(chunk)
        
        pkt = bytearray(DDP_HDR.size + payload_len)
        DDP_HDR.pack_into(
            pkt, 0,
            flags,
            seq & 0xFF,
            pixcfg,
            output_id & 0xFF,
            off,
            payload_len
        )
        pkt[DDP_HDR.size:] = chunk.tobytes()
        yield bytes(pkt)
        off = end


class DDPOutput(BufferedOutputProtocol):
    """DDP output protocol implementation."""
    
    def __init__(self, target: OutputTarget, config: Dict[str, Any]):
        # Override max queue size from config
        config_copy = config.copy()
        config_copy["max_queue_size"] = config.get("max_queue_size", 4096)
        super().__init__(target, config_copy)
        
        self.config_obj = Config()
        self.sender: Optional[DDPSender] = None
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.socket: Optional[socket.socket] = None
        self.seq = 0
        
        # Performance tracking
        self.tracker = PerformanceTracker(
            log_interval_s=config.get("log_interval_s", 1.0)
        )
        
        # Frame format configuration
        self.pixel_format = normalize_pixel_format(config.get("fmt", "rgb888"))
        
        # Packet spreading configuration
        self.spread_enabled = bool(self.config_obj.get("net.spread_packets", True))
        self.spread_max_fps = int(self.config_obj.get("net.spread_max_fps", 60))
        
        # Still frame resend configuration
        self.still_resend_config = self.config_obj.get("playback_still", {})
        self.last_frame_data: Optional[bytes] = None
        self.last_frame_seq: Optional[int] = None
        self.last_frame_was_still = False

        # Mode and pacing information for logging
        self.mode = config.get("mode", "unknown")
        self.pace_hz = config.get("pace_hz", 0)

        # Track target FPS for native mode
        self.last_delay_ms: Optional[float] = None

        # Packet counters for logging
        self._packets_enqueued = 0
        
    async def start(self) -> None:
        """Initialize UDP transport."""
        await super().start()
        
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        except OSError:
            pass
            
        sock.bind(("0.0.0.0", 0))
        
        self.sender = DDPSender()
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: self.sender, sock=sock
        )
        self.socket = sock
        
        print(f"[ddp] started output to {self.target.host}:{self.target.port} (id={self.target.output_id})")
        
    async def stop(self) -> None:
        """Clean up UDP transport."""
        await super().stop()
        
        if self.transport:
            self.transport.close()
            self.transport = None
            
        if self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
            self.socket = None
            
        print(f"[ddp] stopped output {self.target.output_id}")
        
    async def configure(self, new_config: Dict[str, Any]) -> None:
        """Update configuration."""
        if "fmt" in new_config:
            self.pixel_format = normalize_pixel_format(new_config["fmt"])
            
        # Update other configuration as needed
        self.config.update(new_config)
        
    async def _send_frame_internal(self, frame_data: bytes, metadata: FrameMetadata) -> None:
        """Send a single frame via DDP."""
        if not self.sender:
            return
            
        # Determine if packet spreading should be used
        use_spreading = False
        spacing_s = None
        group_n = 1
        
        if self.spread_enabled:
            frame_rate = 1000.0 / max(metadata.delay_ms, 1.0)
            if frame_rate <= self.spread_max_fps:
                # Calculate effective payload length for spacing
                if self.pixel_format == "rgb888":
                    payload_len = len(frame_data)
                else:  # RGB565
                    payload_len = (len(frame_data) // 3) * 2
                    
                pkt_count = (payload_len + DDP_MAX_DATA - 1) // DDP_MAX_DATA
                spacing_s, group_n = compute_spacing_and_group(
                    pkt_count, metadata.delay_ms / 1000.0, self.config_obj.get()
                )
                use_spreading = spacing_s is not None
        
        # Send packets with optional spreading
        await self._send_frame_packets(
            frame_data, metadata, 
            spacing_s=spacing_s, group_n=group_n
        )
        
        # Update tracking
        self.tracker.record_frame()

        # Track delay for target FPS calculation
        self.last_delay_ms = metadata.delay_ms

        # Cache for potential still resends
        if metadata.is_still or metadata.is_last_frame:
            self.last_frame_data = frame_data
            self.last_frame_seq = self.seq
            self.last_frame_was_still = metadata.is_still

        self.seq = (self.seq + 1) & 0xFF
        
        # Handle still frame resends
        if metadata.is_still and metadata.is_last_frame:
            await self._handle_still_resends(frame_data)
            
        # Log metrics if needed
        # For single-frame/still content, log once per frame since they're rare
        # For looping content, respect the minimum 5-second interval
        should_log = self.tracker.should_log(is_loop_end=metadata.is_last_frame)

        if should_log:
            await self._log_metrics()

        # Track loop starts for very short content
        if metadata.is_last_frame:
            self.tracker.record_loop_start()
            
    async def _send_frame_packets(self, frame_data: bytes, metadata: FrameMetadata,
                                *, spacing_s: Optional[float] = None, group_n: int = 1) -> None:
        """Send all packets for a frame with optional spreading."""
        if not self.sender:
            return
            
        addr = (self.target.host, self.target.port)
        packets = list(ddp_iter_packets(
            frame_data, self.target.output_id, self.seq, fmt=self.pixel_format
        ))
        
        if not spacing_s or spacing_s <= 0:
            # Send all packets immediately
            for pkt in packets:
                self.sender.sendto(pkt, addr)
                self.tracker.record_packet()
                self._packets_enqueued += 1
            return
            
        # Send with spreading
        loop = asyncio.get_running_loop()
        start_time = loop.time()
        slot_idx = 0
        group_left = group_n
        
        for pkt in packets:
            self.sender.sendto(pkt, addr)
            self.tracker.record_packet()
            self._packets_enqueued += 1
            
            group_left -= 1
            if group_left <= 0:
                slot_idx += 1
                target_time = start_time + slot_idx * spacing_s
                sleep_time = max(0.0, target_time - loop.time())
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                group_left = group_n
                
    async def _handle_still_resends(self, frame_data: bytes) -> None:
        """Handle resending of still frames for reliability."""
        if not self.still_resend_config or not self.sender:
            return
            
        burst = int(self.still_resend_config.get("burst", 0))
        spacing_ms = float(self.still_resend_config.get("spacing_ms", 100.0))
        tail_s = float(self.still_resend_config.get("tail_s", 0.0))
        tail_hz = int(self.still_resend_config.get("tail_hz", 0))
        
        addr = (self.target.host, self.target.port)
        seq = self.last_frame_seq or self.seq
        
        # Burst phase
        if burst > 0:
            print(f"[ddp] still resend burst={burst} output={self.target.output_id}")
            for i in range(burst):
                packets = list(ddp_iter_packets(
                    frame_data, self.target.output_id, seq, fmt=self.pixel_format
                ))
                for pkt in packets:
                    self.sender.sendto(pkt, addr)
                    self.tracker.record_packet()
                    self._packets_enqueued += 1
                    
                if spacing_ms > 0 and i < burst - 1:
                    await asyncio.sleep(spacing_ms / 1000.0)
                    
        # Tail phase
        if tail_s > 0 and tail_hz > 0:
            print(f"[ddp] still resend tail={tail_s}s @ {tail_hz}Hz output={self.target.output_id}")
            loop = asyncio.get_running_loop()
            end_time = loop.time() + tail_s
            tick = 1.0 / tail_hz
            next_time = loop.time()
            
            while loop.time() < end_time:
                packets = list(ddp_iter_packets(
                    frame_data, self.target.output_id, seq, fmt=self.pixel_format
                ))
                for pkt in packets:
                    self.sender.sendto(pkt, addr)
                    self.tracker.record_packet()
                    self._packets_enqueued += 1
                    
                await asyncio.sleep(max(0.0, next_time - loop.time()))
                next_time += tick
                
    async def _log_metrics(self) -> None:
        """Log performance metrics."""
        metrics = self.tracker.get_metrics_and_reset()

        spread_tag = " (spread)" if self.spread_enabled else ""

        if self.mode == "pace":
            # Paced mode logging
            print(
                f"[send] out={self.target.output_id} pace={self.pace_hz}Hz "
                f"fps={metrics['fps']:.2f} pps={metrics['pps']:.0f} "
                f"pkt_jit={metrics['packet_jitter_ms']:.1f}ms frm_jit={metrics['frame_jitter_ms']:.1f}ms "
                f"q_avg={metrics['queue_avg']:.0f}/{self._max_queue_size} "
                f"q_max={metrics['queue_max']} "
                f"enq={self._packets_enqueued} tx={self.sender.packets_sent if self.sender else 0} "
                f"drops={metrics['queue_drops']}{spread_tag}"
            )
        else:
            # Native mode logging with target FPS
            target_fps = 1000.0 / max(self.last_delay_ms or 33.33, 1.0)
            print(
                f"[send] out={self.target.output_id} native "
                f"fps={metrics['fps']:.2f} (~{target_fps:.1f} tgt) pps={metrics['pps']:.0f} "
                f"pkt_jit={metrics['packet_jitter_ms']:.1f}ms frm_jit={metrics['frame_jitter_ms']:.1f}ms "
                f"q_avg={metrics['queue_avg']:.0f}/{self._max_queue_size} "
                f"q_max={metrics['queue_max']} "
                f"enq={self._packets_enqueued} tx={self.sender.packets_sent if self.sender else 0} "
                f"drops={metrics['queue_drops']}{spread_tag}"
            )


# Register the DDP protocol
OutputProtocolFactory.register("ddp", DDPOutput)