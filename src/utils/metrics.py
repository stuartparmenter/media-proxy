# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import time
import math
from typing import List


class RateMeter:
    """Rolling rate/jitter meter using a short timestamp window."""
    
    def __init__(self, window_s: float = 2.5):
        self.window_s = float(window_s)
        self.ts: List[float] = []

    def tick(self, t: float) -> None:
        """Record a timestamp."""
        self.ts.append(t)
        cut = t - self.window_s
        i = 0
        for i, v in enumerate(self.ts):
            if v >= cut:
                break
        if i > 0:
            del self.ts[:i]

    def rate_hz(self) -> float:
        """Calculate current rate in Hz."""
        n = len(self.ts)
        if n < 2:
            return 0.0
        duration = self.ts[-1] - self.ts[0]
        return (n - 1) / duration if duration > 0 else 0.0

    def jitter_ms(self) -> float:
        """Calculate timing jitter in milliseconds."""
        n = len(self.ts)
        if n < 3:
            return 0.0
        diffs = [(self.ts[i] - self.ts[i - 1]) for i in range(1, n)]
        mean = sum(diffs) / len(diffs)
        var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
        return math.sqrt(var) * 1000.0

    def clear(self) -> None:
        """Clear all recorded timestamps."""
        self.ts.clear()


class PerformanceTracker:
    """Tracks multiple performance metrics."""
    
    def __init__(self, log_interval_s: float = 1.0):
        self.log_interval_s = log_interval_s
        self.last_log = time.perf_counter()
        
        self.frame_meter = RateMeter()
        self.packet_meter = RateMeter()
        
        # Counters
        self.frames_processed = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.queue_drops = 0
        
        # Queue occupancy samples
        self.queue_samples: List[int] = []
        
    def record_frame(self) -> None:
        """Record a frame being processed."""
        now = time.perf_counter()
        self.frame_meter.tick(now)
        self.frames_processed += 1
        
    def record_packet(self) -> None:
        """Record a packet being sent."""
        now = time.perf_counter()
        self.packet_meter.tick(now)
        self.packets_sent += 1
        
    def record_bytes(self, byte_count: int) -> None:
        """Record bytes sent."""
        self.bytes_sent += byte_count
        
    def record_queue_drop(self) -> None:
        """Record a queue drop event."""
        self.queue_drops += 1
        
    def record_queue_size(self, size: int) -> None:
        """Record current queue occupancy."""
        self.queue_samples.append(size)
        
    def should_log(self) -> bool:
        """Check if it's time to log metrics."""
        now = time.perf_counter()
        return (now - self.last_log) >= self.log_interval_s
        
    def get_metrics_and_reset(self) -> dict:
        """Get current metrics and reset counters."""
        fps = self.frame_meter.rate_hz()
        pps = self.packet_meter.rate_hz()
        frame_jitter = self.frame_meter.jitter_ms()
        packet_jitter = self.packet_meter.jitter_ms()
        
        queue_avg = (sum(self.queue_samples) / len(self.queue_samples)) if self.queue_samples else 0
        queue_max = max(self.queue_samples) if self.queue_samples else 0
        
        metrics = {
            "fps": fps,
            "pps": pps,
            "frame_jitter_ms": frame_jitter,
            "packet_jitter_ms": packet_jitter,
            "frames_processed": self.frames_processed,
            "packets_sent": self.packets_sent,
            "bytes_sent": self.bytes_sent,
            "queue_drops": self.queue_drops,
            "queue_avg": queue_avg,
            "queue_max": queue_max,
        }
        
        # Reset counters
        self.frames_processed = 0
        self.packets_sent = 0
        self.bytes_sent = 0
        self.queue_drops = 0
        self.queue_samples.clear()
        self.last_log = time.perf_counter()
        
        return metrics