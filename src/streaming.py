# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import os
from typing import Dict, Any, AsyncIterator, Tuple, Optional
from urllib.parse import unquote

from .config import Config
from .media.sources import resolve_media_source, StreamUrlExpiredError
from .media.images import iter_frames_imageio
from .media.video import iter_frames_pyav
from .output.protocol import OutputProtocolFactory, OutputTarget, FrameMetadata
from .utils.helpers import (
    resolve_local_path, is_http_url, probe_http_content_type,
    parse_expand_mode, parse_hw_preference, parse_pace_hz, truthy, is_youtube_url
)
from .utils.hardware import choose_decode_preference


async def _get_gif_fps_from_pyav(src_url: str) -> Optional[float]:
    """Try to get accurate FPS from a GIF using PyAV (quick probe)."""
    try:
        source = await resolve_media_source(src_url)

        # Quick probe with PyAV to get fps
        import av
        container = av.open(source.resolved_url, mode="r", options=source.options)
        try:
            vstream = next((s for s in container.streams if s.type == "video"), None)
            if vstream and vstream.average_rate:
                fps = float(vstream.average_rate)
                print(f"[gif] PyAV detected fps={fps}")
                return fps
        finally:
            container.close()
    except Exception as e:
        print(f"[gif] PyAV fps probe failed: {e!r}")
    return None


async def iter_frames_async(
    src: str,
    size: Tuple[int, int],
    loop_video: bool = True,
    *,
    expand_mode: int,
    hw_prefer: str = None
) -> AsyncIterator[Tuple[bytes, float]]:
    """
    Async version of iter_frames that handles YouTube URL resolution properly.
    """
    src_url = unquote(src)
    low = src_url.lower()
    
    # Prefer extension check first for local paths
    is_image_ext = any(low.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"))
    if is_image_ext:
        # For GIFs, try to get accurate FPS from PyAV first
        actual_fps = None
        if low.endswith(".gif"):
            actual_fps = await _get_gif_fps_from_pyav(src_url)

        for rgb888, delay_ms in iter_frames_imageio(src_url, size, loop_video, fps_override=actual_fps):
            yield rgb888, delay_ms
        return

    # For HTTP(S) sources where the URL doesn't reveal type, probe Content-Type
    if is_http_url(src_url):
        ct = probe_http_content_type(src_url)
        if ct:
            if ct.startswith("image/"):
                print(f"[detect] http Content-Type={ct} -> using imageio still path")
                for rgb888, delay_ms in iter_frames_imageio(src_url, size, loop_video):
                    yield rgb888, delay_ms
                return

    # Handle video streams (including YouTube) with retry logic
    original_src = src_url
    max_retries = 3
    retry_count = 0
    
    while True:
        try:
            # Resolve URL asynchronously if it's a YouTube URL
            source = await resolve_media_source(src_url)

            # Use the existing PyAV frame iterator
            frames_yielded = False
            for rgb888, delay_ms in iter_frames_pyav(
                src_url, source.resolved_url, source.options, size, loop_video,
                expand_mode=expand_mode, hw_prefer=hw_prefer
            ):
                # Reset retry count on first successful frame (indicates successful resolution)
                if not frames_yielded:
                    retry_count = 0
                    frames_yielded = True
                yield rgb888, delay_ms
                
            # If we get here without exception and we're looping, continue the loop
            if not loop_video:
                break
                
        except StreamUrlExpiredError as e:
            # URL expired, try to re-resolve
            if not is_youtube_url(original_src):
                # Non-YouTube URLs can't be re-resolved
                raise RuntimeError(f"Non-YouTube URL failed and cannot be re-resolved: {e}") from e
                
            retry_count += 1
            if retry_count > max_retries:
                raise RuntimeError(f"Max retries ({max_retries}) exceeded for YouTube URL resolution") from e
                
            print(f"[retry] YouTube URL expired (attempt {retry_count}/{max_retries}), re-resolving...")
            await asyncio.sleep(0.5)  # Brief delay before retry
            continue
            
        except Exception as e:
            # Other errors are not retryable
            msg = str(e).lower()
            if isinstance(e, FileNotFoundError) or "no such file" in msg or "not found" in msg:
                raise FileNotFoundError(f"cannot open source: {original_src}") from e
            raise RuntimeError(f"Frame iteration error: {e}") from e
            
        if not loop_video:
            break


async def create_streaming_task(session, params: Dict[str, Any]) -> asyncio.Task:
    """Create a streaming task that connects media input to output."""
    config = Config()
    
    # Extract parameters
    output_id = int(params["out"])
    width = int(params["w"])
    height = int(params["h"])
    src = str(params["src"])
    ddp_port = int(params.get("ddp_port", 4048))
    
    # Parse options
    opts = {
        "loop": truthy(str(params.get("loop", config.get("playback.loop")))),
        "expand_mode": parse_expand_mode(params.get("expand"), config.get("video.expand_mode")),
        "hw": parse_hw_preference(params.get("hw"), config.get("hw.prefer")),
        "pace_hz": parse_pace_hz(params.get("pace", 0)),
        "ema_alpha": max(0.0, min(float(params.get("ema", 0.0)), 1.0)),
        "fmt": params.get("fmt", "rgb888"),
    }
    
    # Validate local file exists
    src_url = unquote(src)
    local_path = resolve_local_path(src_url)
    if local_path is not None and not os.path.exists(local_path):
        raise FileNotFoundError(f"no such file: {local_path}")
    
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
        "log_interval_s": config.get("log.rate_ms", 1000) / 1000.0,
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
            
    finally:
        await output.stop()


async def _run_native_streaming(output, size: Tuple[int, int], src: str, 
                               opts: Dict[str, Any], hw_prefer: str):
    """Run streaming at native source cadence."""
    frames_emitted = 0
    seq = 0
    
    async for rgb888, delay_ms in iter_frames_async(
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
        
        # Wait for next frame
        await asyncio.sleep(max(0.01, delay_ms / 1000.0))


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
        async for rgb888, delay_ms in iter_frames_async(
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