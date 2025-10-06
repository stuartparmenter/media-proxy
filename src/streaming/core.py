# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import logging
import os
from typing import Dict, Any, AsyncIterator, Tuple, Optional
from urllib.parse import unquote

import urllib.error
import av.error
import yt_dlp
from ..config import Config

from ..utils.fields import ControlFields
from .options import StreamOptions
from ..media.sources import resolve_media_source, MediaUnavailableError, is_internal_url, rewrite_internal_url
from ..media.protocol import FrameIteratorFactory
from ..media.images import cleanup_active_image_sources
from ..output.protocol import OutputProtocolFactory, OutputTarget, FrameMetadata, BufferedOutputProtocol
from ..utils.helpers import is_youtube_url


# Removed _get_gif_fps_from_pyav - PIL handles GIF timing directly


async def stream_frames(stream_options: StreamOptions) -> AsyncIterator[Tuple[bytes, float]]:
    """
    Stream frames from a media source with automatic retry and error recovery.
    """
    src_url = unquote(stream_options.source)
    original_src = src_url
    max_retries = 3
    retry_count = 0

    # Resolve YouTube URL once before retry loop
    resolved_url = src_url
    source_options = {}

    if is_youtube_url(src_url):
        try:
            source = await resolve_media_source(src_url, stream_options)
            resolved_url = source.resolved_url
            source_options = source.options

            # Check if we should enable FFmpeg caching
            if source.should_enable_cache(stream_options.loop):
                config = Config()
                filesize = source.info.get('filesize') or source.info.get('filesize_approx')
                max_size = config.get('youtube.cache.max_size')
                max_size_mb = max_size / 1024 / 1024
                size_str = f"{filesize/1024/1024:.1f}MB" if filesize else "unknown"
                logging.getLogger('streaming').info(
                    f"Enabling FFmpeg cache: size={size_str}, max={max_size_mb:.0f}MB"
                )
                stream_options.enable_cache = True

        except Exception as e:
            if isinstance(e, yt_dlp.DownloadError):  # type: ignore[attr-defined]
                # YouTube format unavailable - re-raise for upstream handling
                logging.getLogger('streaming').error(f"YouTube DownloadError: {e}")
                raise MediaUnavailableError(f"YouTube resolution failed: {e}", original_src, e) from e
            else:
                # Other exceptions during URL resolution - re-raise
                raise

    # Retry loop for YouTube URL expiration
    while True:
        try:
            # Prepare HTTP options with resilience settings for YouTube
            http_options = None
            if is_youtube_url(src_url) and source_options:
                http_options = source_options.copy()
                http_options.update({
                    'reconnect': '1',
                    'reconnect_streamed': '1',
                    'reconnect_delay_max': '5',
                    'timeout': '10000000',
                })

            # Create and initialize iterator (factory handles property setting)
            iterator = await FrameIteratorFactory.create(
                src_url,
                stream_options,
                resolved_url=resolved_url if is_youtube_url(src_url) else None,
                http_opts=http_options
            )

            # Iterate frames - the iterator handles looping internally
            frames_yielded = False
            for rgb888, delay_ms in iterator:
                # Reset retry count on first successful frame
                if not frames_yielded:
                    retry_count = 0
                    frames_yielded = True

                yield rgb888, delay_ms

            # Iterator exhausted
            break


        except urllib.error.HTTPError as e:
            # HTTP errors from images.py - convert to MediaUnavailableError for consistent handling
            raise MediaUnavailableError(f"HTTP {e.code}: {e.reason}", original_src, e) from e
        except urllib.error.URLError as e:
            # Network errors from images.py - convert to MediaUnavailableError for consistent handling
            raise MediaUnavailableError(f"Network error: {e.reason}", original_src, e) from e
        except (av.error.HTTPError, av.error.HTTPClientError, av.error.HTTPServerError, av.error.InvalidDataError, av.error.ValueError) as e:
            # PyAV errors from video.py
            errno_val = getattr(e, 'errno', None)
            if is_youtube_url(original_src):
                # Log connection-specific diagnostics for YouTube streams
                if errno_val == 5:
                    logging.getLogger('streaming').warning(
                        f"YouTube I/O error (errno 5 - connection lost): {e} "
                        f"- attempt {retry_count + 1}/{max_retries}"
                    )
                else:
                    logging.getLogger('streaming').info(
                        f"YouTube error detected (errno={errno_val}): {e}"
                    )

                # For YouTube URLs, these errors likely mean URL expiration or connection loss - trigger retry
                retry_count += 1
                if retry_count > max_retries:
                    raise MediaUnavailableError(f"YouTube retry failed after {max_retries} attempts: {e}", original_src, e) from e
                logging.getLogger('streaming').info(f"YouTube URL issue (attempt {retry_count}/{max_retries}), re-resolving...")
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

        if not stream_options.loop:
            break


def create_streaming_task(session, params: Dict[str, Any]) -> asyncio.Task:
    """Create a streaming task that connects media input to output."""
    config = Config()

    # Resolve internal: URLs using session's server info
    src = params.get("src")
    if src and is_internal_url(src):
        params = params.copy()  # Don't mutate original
        params["src"] = rewrite_internal_url(src, session.server_host)

    # Create strongly typed streaming options from control parameters
    stream_options = StreamOptions.from_control_params(params)

    # Set network target from session
    stream_options.target_ip = session.client_ip

    # Log the streaming options
    stream_options.log_info(f"{session.client_ip} dev={session.device_id}")

    # Create the streaming task
    task = asyncio.create_task(streaming_task(stream_options))

    def on_done(t: asyncio.Task):
        try:
            exc = t.exception()
        except asyncio.CancelledError:
            return
        except Exception as e:
            logging.getLogger('streaming').error(f"out={stream_options.output_id} exception(): {e!r}")
            return
        if exc:
            logging.getLogger('streaming').error(f"out={stream_options.output_id} crashed: {exc!r}")
    task.add_done_callback(on_done)

    return task


async def streaming_task(stream_options: StreamOptions):
    """Main streaming task that orchestrates media input -> processing -> output."""
    config = Config()

    # Create output target and protocol
    target = OutputTarget(
        host=stream_options.target_ip,
        port=stream_options.ddp_port,
        output_id=stream_options.output_id,
        protocol="ddp"
    )

    output = OutputProtocolFactory.create(target, stream_options)

    try:
        await output.start()

        if stream_options.pace > 0:
            # Paced mode: producer + sampler
            await _run_paced_streaming(output, stream_options)
        else:
            # Native cadence mode
            await _run_native_streaming(output, stream_options)

    except MediaUnavailableError as e:
        # Media source unavailable (HTTP/network errors) - stop this stream task
        logging.getLogger('streaming').warning(f"{target.output_id} media unavailable: {e}")
        return
    except FileNotFoundError as e:
        # Actual file not found - stop this stream task
        logging.getLogger('streaming').warning(f"{target.output_id} file not found: {e}")
        return
    except Exception as e:
        # Technical errors (codec issues, etc.) - stop this stream task
        logging.getLogger('streaming').error(f"{target.output_id} technical error: {e}")
        return
    finally:
        # For non-looping content, ensure all packets are fully sent
        if not stream_options.loop and isinstance(output, BufferedOutputProtocol):
            await output.flush_and_stop()
        else:
            await output.stop()
        # Clean up any temp files from image caching
        cleanup_active_image_sources()


async def _run_native_streaming(output, stream_options: StreamOptions):
    """Run streaming at native source cadence."""
    frames_emitted = 0
    seq = 0
    next_frame_time = asyncio.get_event_loop().time()

    async for rgb888, delay_ms in stream_frames(stream_options):
        # Create frame metadata
        metadata = FrameMetadata(
            sequence=seq,
            timestamp_ms=asyncio.get_event_loop().time() * 1000,
            delay_ms=delay_ms,
            size=stream_options.size,
            format=stream_options.fmt,
            is_still=(not stream_options.loop and frames_emitted == 0),
            is_last_frame=(not stream_options.loop and frames_emitted == 0)
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



async def _run_paced_streaming(output, stream_options: StreamOptions) -> None:
    """Run streaming with fixed pacing (producer + sampler pattern)."""
    import numpy as np

    pace_hz = stream_options.pace
    ema_alpha = stream_options.ema

    # Shared state
    latest_frame: Dict[str, Optional[bytes]] = {"data": None}
    latest_lock = asyncio.Lock()

    # Producer task
    async def producer():
        async for rgb888, delay_ms in stream_frames(stream_options):
            async with latest_lock:
                latest_frame["data"] = rgb888

            # Respect source timing for producer
            await asyncio.sleep(max(0.01, delay_ms / 1000.0))

    # Sampler task
    async def sampler() -> None:
        tick = 1.0 / pace_hz
        next_time = asyncio.get_event_loop().time()
        ema_buf_f32: Optional[np.ndarray] = None
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
                    size=stream_options.size,
                    format=stream_options.fmt
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