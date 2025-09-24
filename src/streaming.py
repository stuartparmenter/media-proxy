# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import os
from typing import Dict, Any, AsyncIterator, Tuple, Optional
from urllib.parse import unquote

import urllib.error
import av.error
import yt_dlp
from .config import Config

from .control.fields import ControlFields
from .media.sources import resolve_media_source, MediaUnavailableError
from .media.protocol import FrameIteratorFactory, FrameIteratorConfig
from .media.images import cleanup_active_image_sources
from .output.protocol import OutputProtocolFactory, OutputTarget, FrameMetadata
from .utils.helpers import (
    resolve_local_path, is_http_url, probe_http_content_type,
    parse_expand_mode, parse_hw_preference, parse_pace_hz, truthy, is_youtube_url,
    normalize_pixel_format
)
from .utils.hardware import choose_decode_preference


# Removed _get_gif_fps_from_pyav - PIL handles GIF timing directly


async def stream_frames(
    src: str,
    size: Tuple[int, int],
    loop_video: bool = True,
    *,
    expand_mode: int,
    hw_prefer: str = None
) -> AsyncIterator[Tuple[bytes, float]]:
    """
    Stream frames from a media source with automatic retry and error recovery.
    """
    src_url = unquote(src)
    original_src = src_url
    max_retries = 3
    retry_count = 0

    while True:
        try:
            # For YouTube URLs, resolve to get actual stream URL and options
            if is_youtube_url(src_url):
                try:
                    source = await resolve_media_source(src_url)
                    resolved_url = source.resolved_url
                    # Check if resolution actually succeeded (resolved URL should be different)
                    if resolved_url == src_url:
                        print(f"[retry] YouTube URL resolution failed on attempt {retry_count + 1}, will retry on PyAV error")
                except Exception as e:
                    if isinstance(e, yt_dlp.DownloadError):
                        # YouTube format unavailable - trigger retry immediately
                        print(f"[retry] YouTube DownloadError detected: {e}")
                        retry_count += 1
                        if retry_count > max_retries:
                            raise MediaUnavailableError(f"YouTube retry failed after {max_retries} attempts: {e}", original_src, e) from e
                        print(f"[retry] YouTube DownloadError (attempt {retry_count}/{max_retries}), re-resolving...")
                        await asyncio.sleep(0.5)
                        continue
                    else:
                        # Other exceptions during URL resolution - re-raise
                        raise
            else:
                resolved_url = src_url

            # Create frame iterator configuration
            config = FrameIteratorConfig(
                size=size,
                loop_video=loop_video,
                expand_mode=expand_mode,
                hw_prefer=hw_prefer
            )

            # Use the factory to create the appropriate iterator
            iterator = FrameIteratorFactory.create(src_url, config)

            # If this is a PyAV iterator and we have a resolved URL, update it
            if hasattr(iterator, 'real_src_url') and is_youtube_url(src_url):
                iterator.real_src_url = resolved_url

            # Iterate frames using the protocol
            frames_yielded = False
            for rgb888, delay_ms in iterator:
                # Reset retry count on first successful frame
                if not frames_yielded:
                    retry_count = 0
                    frames_yielded = True
                yield rgb888, delay_ms

            # If we get here without exception and we're looping, continue the loop
            if not loop_video:
                break


        except urllib.error.HTTPError as e:
            # HTTP errors from images.py - convert to MediaUnavailableError for consistent handling
            raise MediaUnavailableError(f"HTTP {e.code}: {e.reason}", original_src, e) from e
        except urllib.error.URLError as e:
            # Network errors from images.py - convert to MediaUnavailableError for consistent handling
            raise MediaUnavailableError(f"Network error: {e.reason}", original_src, e) from e
        except (av.error.HTTPError, av.error.HTTPClientError, av.error.HTTPServerError, av.error.InvalidDataError, av.error.ValueError) as e:
            # PyAV errors from video.py
            if is_youtube_url(original_src):
                # For YouTube URLs, these errors likely mean URL expiration or failed resolution - trigger retry
                print(f"[retry] YouTube error detected: {e}")
                retry_count += 1
                if retry_count > max_retries:
                    raise MediaUnavailableError(f"YouTube retry failed after {max_retries} attempts: {e}", original_src, e) from e
                print(f"[retry] YouTube URL issue (attempt {retry_count}/{max_retries}), re-resolving...")
                await asyncio.sleep(0.5)
                continue
            else:
                # Non-YouTube errors - convert to MediaUnavailableError
                raise MediaUnavailableError(f"Media error: {e}", original_src, e) from e
        except FileNotFoundError as e:
            # File not found - re-raise with original source context
            raise FileNotFoundError(f"cannot open source: {original_src}") from e
        except RuntimeError as e:
            # RuntimeErrors are not retryable
            raise
        except Exception as e:
            # Other errors are not retryable
            raise RuntimeError(f"Frame iteration error: {e}") from e

        if not loop_video:
            break


async def create_streaming_task(session, params: Dict[str, Any]) -> asyncio.Task:
    """Create a streaming task that connects media input to output."""
    config = Config()

    # Extract parameters using ControlFields for consistency
    output_id = int(params["out"])
    width = int(params["w"])
    height = int(params["h"])
    src = str(params["src"])
    ddp_port = int(params.get("ddp_port", 4048))

    # Parse options using field-aware helpers
    opts = {}

    # Process each field that can be applied to the stream
    for field_name, value in params.items():
        field_def = ControlFields.get_field_info(field_name)
        if field_def is None:
            continue

        if field_name == "loop":
            opts["loop"] = truthy(str(value)) if value is not None else config.get("playback.loop")
        elif field_name == "expand":
            opts["expand_mode"] = parse_expand_mode(value, config.get("video.expand_mode"))
        elif field_name == "hw":
            opts["hw"] = parse_hw_preference(value, config.get("hw.prefer"))
        elif field_name == "pace":
            opts["pace_hz"] = parse_pace_hz(value)
        elif field_name == "ema":
            opts["ema_alpha"] = max(0.0, min(float(value), 1.0)) if value is not None else 0.0
        elif field_name == "fmt":
            opts["fmt"] = normalize_pixel_format(str(value)) if value is not None else "rgb888"

    # Set defaults for any missing options
    if "loop" not in opts:
        opts["loop"] = config.get("playback.loop")
    if "expand_mode" not in opts:
        opts["expand_mode"] = config.get("video.expand_mode")
    if "hw" not in opts:
        opts["hw"] = config.get("hw.prefer")
    if "pace_hz" not in opts:
        opts["pace_hz"] = 0
    if "ema_alpha" not in opts:
        opts["ema_alpha"] = 0.0
    if "fmt" not in opts:
        opts["fmt"] = normalize_pixel_format("rgb888")

    # Note: Local file validation is handled during frame iteration

    print(f"* start_stream {session.client_ip} dev={session.device_id} out={output_id} "
          f"size={width}x{height} ddp_port={ddp_port} src={src} "
          f"pace={opts['pace_hz']} ema={opts['ema_alpha']} expand={opts['expand_mode']} "
          f"loop={opts['loop']} hw={opts['hw']} fmt={opts['fmt']}")

    # Create the streaming task
    task = asyncio.create_task(
        streaming_task(
            target_ip=session.client_ip,
            target_port=ddp_port,
            output_id=output_id,
            size=(width, height),
            src=src,
            opts=opts
        )
    )

    def on_done(t: asyncio.Task):
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        except Exception as e:
            print(f"[task] out={output_id} exception(): {e!r}")
            return
        if exc:
            print(f"[task] out={output_id} crashed: {exc!r}")
    task.add_done_callback(on_done)

    return task


async def streaming_task(target_ip: str, target_port: int, output_id: int, *, 
                        size: Tuple[int, int], src: str, opts: Dict[str, Any]):
    """Main streaming task that orchestrates media input -> processing -> output."""
    config = Config()
    
    # Choose hardware decode preference intelligently
    hw_prefer = choose_decode_preference(
        src, opts["hw"], size, opts["expand_mode"]
    )
    
    # Create output target and protocol
    target = OutputTarget(
        host=target_ip,
        port=target_port,
        output_id=output_id,
        protocol="ddp"
    )
    
    output_config = {
        "fmt": opts["fmt"],
        "log_interval_s": config.get("log.rate_ms") / 1000.0,
        "log_metrics": config.get("log.metrics"),
        "log_detail": config.get("log.detail"),
        "max_queue_size": 4096,
        "mode": "pace" if opts["pace_hz"] > 0 else "native",
        "pace_hz": opts["pace_hz"],
    }
    
    output = OutputProtocolFactory.create(target, output_config)
    
    try:
        await output.start()

        if opts["pace_hz"] > 0:
            # Paced mode: producer + sampler
            await _run_paced_streaming(output, size, src, opts, hw_prefer)
        else:
            # Native cadence mode
            await _run_native_streaming(output, size, src, opts, hw_prefer)

    except MediaUnavailableError as e:
        # Media source unavailable (HTTP/network errors) - stop this stream task
        print(f"[stream] {target.output_id} media unavailable: {e}")
        return
    except FileNotFoundError as e:
        # Actual file not found - stop this stream task
        print(f"[stream] {target.output_id} file not found: {e}")
        return
    except Exception as e:
        # Technical errors (codec issues, etc.) - stop this stream task
        print(f"[stream] {target.output_id} technical error: {e}")
        return
    finally:
        # For non-looping content, ensure all packets are fully sent
        if not opts["loop"] and hasattr(output, 'flush_and_stop'):
            await output.flush_and_stop()
        else:
            await output.stop()
        # Clean up any temp files from image caching
        cleanup_active_image_sources()


async def _run_native_streaming(output, size: Tuple[int, int], src: str,
                               opts: Dict[str, Any], hw_prefer: str):
    """Run streaming at native source cadence."""
    frames_emitted = 0
    seq = 0
    next_frame_time = asyncio.get_event_loop().time()

    async for rgb888, delay_ms in stream_frames(
        src, size,
        loop_video=opts["loop"],
        expand_mode=opts["expand_mode"],
        hw_prefer=hw_prefer
    ):
        # Create frame metadata
        metadata = FrameMetadata(
            sequence=seq,
            timestamp_ms=asyncio.get_event_loop().time() * 1000,
            delay_ms=delay_ms,
            size=size,
            format=opts["fmt"],
            is_still=(not opts["loop"] and frames_emitted == 0),
            is_last_frame=(not opts["loop"] and frames_emitted == 0)
        )

        # Send frame
        await output.send_frame(rgb888, metadata)

        frames_emitted += 1
        seq = (seq + 1) & 0xFF

        # Precise timing: wait until next scheduled frame time
        next_frame_time += max(0.01, delay_ms / 1000.0)
        current_time = asyncio.get_event_loop().time()
        sleep_duration = next_frame_time - current_time

        # If we're more than 100ms behind, reset timing to prevent runaway catch-up
        if sleep_duration < -0.1:
            next_frame_time = current_time
        elif sleep_duration > 0:
            await asyncio.sleep(sleep_duration)



async def _run_paced_streaming(output, size: Tuple[int, int], src: str,
                              opts: Dict[str, Any], hw_prefer: str):
    """Run streaming with fixed pacing (producer + sampler pattern)."""
    import numpy as np
    
    pace_hz = opts["pace_hz"]
    ema_alpha = opts["ema_alpha"]
    
    # Shared state
    latest_frame: Dict[str, bytes] = {"data": None}
    latest_lock = asyncio.Lock()
    
    # Producer task
    async def producer():
        async for rgb888, delay_ms in stream_frames(
            src, size,
            loop_video=opts["loop"],
            expand_mode=opts["expand_mode"],
            hw_prefer=hw_prefer
        ):
            async with latest_lock:
                latest_frame["data"] = rgb888

            # Respect source timing for producer
            await asyncio.sleep(max(0.01, delay_ms / 1000.0))
    
    # Sampler task  
    async def sampler():
        tick = 1.0 / pace_hz
        next_time = asyncio.get_event_loop().time()
        ema_buf_f32: np.ndarray = None
        seq = 0
        
        while True:
            frame_data = latest_frame["data"]
            if frame_data is not None:
                output_data = frame_data
                
                # Apply EMA if configured
                if ema_alpha > 0.0:
                    cur = np.frombuffer(frame_data, dtype=np.uint8)
                    if ema_buf_f32 is None or ema_buf_f32.shape != cur.shape:
                        ema_buf_f32 = cur.astype(np.float32, copy=True)
                    else:
                        ema_buf_f32 *= (1.0 - ema_alpha)
                        ema_buf_f32 += cur.astype(np.float32) * ema_alpha
                    output_data = ema_buf_f32.astype(np.uint8, copy=False).tobytes()
                
                # Create metadata
                metadata = FrameMetadata(
                    sequence=seq,
                    timestamp_ms=asyncio.get_event_loop().time() * 1000,
                    delay_ms=tick * 1000,
                    size=size,
                    format=opts["fmt"]
                )
                
                # Send frame
                await output.send_frame(output_data, metadata)
                seq = (seq + 1) & 0xFF
            
            # Wait for next tick
            await asyncio.sleep(max(0.0, next_time - asyncio.get_event_loop().time()))
            next_time += tick
    
    # Run both tasks concurrently
    producer_task = asyncio.create_task(producer())
    sampler_task = asyncio.create_task(sampler())
    
    try:
        await producer_task
    finally:
        sampler_task.cancel()
        try:
            await sampler_task
        except asyncio.CancelledError:
            pass