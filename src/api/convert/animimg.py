# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import io
import logging
import zipfile
from typing import Dict, Any, List, Tuple
from PIL import Image
from aiohttp import web
import yaml

from ...config import Config
from ...streaming.options import StreamOptions
from ...media.protocol import FrameIteratorFactory


async def handle_animimg_request(request) -> web.Response:
    """Handle POST /api/animimg request to extract frames for ESPHome LVGL animimg."""
    try:
        # Parse JSON request
        data = await request.json()

        # Validate required parameters
        required_params = ["source", "width", "height"]
        for param in required_params:
            if param not in data:
                return web.json_response(
                    {"error": f"Missing required parameter: {param}"},
                    status=400
                )

        # Extract parameters with defaults
        source = data["source"]
        width = int(data["width"])
        height = int(data["height"])
        frame_limit = int(data.get("frame_limit", 100))
        fps_limit = data.get("fps_limit", None)
        fit = data.get("fit", "cover")

        # Create minimal StreamOptions for processing
        # We don't need output_id, ddp_port, etc. for frame extraction
        stream_options = create_minimal_stream_options(
            source=source,
            width=width,
            height=height,
            fit=fit
        )

        logging.getLogger('animimg').info(
            f"Processing animimg request: {source} -> {width}x{height}, "
            f"frame_limit={frame_limit}, fit={fit}"
        )

        # Extract frames using existing pipeline
        frames = await extract_frames(stream_options, frame_limit, fps_limit)

        if not frames:
            return web.json_response(
                {"error": "No frames could be extracted from source"},
                status=400
            )

        # Create ZIP file with frames and YAML config
        zip_data = create_animimg_zip(frames, width, height)

        # Return ZIP file download
        return web.Response(
            body=zip_data,
            content_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=animimg_frames.zip"
            }
        )

    except ValueError as e:
        return web.json_response({"error": f"Invalid parameter: {e}"}, status=400)
    except Exception as e:
        logging.getLogger('animimg').error(f"Error processing animimg request: {e}", exc_info=True)
        return web.json_response(
            {"error": f"Processing failed: {e}"},
            status=500
        )


def create_minimal_stream_options(source: str, width: int, height: int, fit: str) -> StreamOptions:
    """Create minimal StreamOptions for frame extraction (no streaming needed)."""
    config = Config()

    # Create fake control params to use existing from_control_params method
    fake_params = {
        "out": 0,  # Not used for frame extraction
        "w": width,
        "h": height,
        "src": source,
        "ddp_port": 4048,  # Not used for frame extraction
        "fit": fit,
        "loop": False,  # Extract once, don't loop
    }

    return StreamOptions.from_control_params(fake_params, config)


async def extract_frames(stream_options: StreamOptions, frame_limit: int, fps_limit: float = None) -> List[Tuple[bytes, float]]:
    """Extract frames using existing media processing pipeline."""
    frames = []

    try:
        # Create frame iterator using existing factory
        iterator = FrameIteratorFactory.create(stream_options.source, stream_options)

        frame_count = 0
        last_frame_time = 0.0
        min_frame_interval = 1000.0 / fps_limit if fps_limit else 0.0

        # Extract frames with limits
        for rgb888_bytes, delay_ms in iterator:
            # Apply FPS limit if specified
            if fps_limit and (last_frame_time + min_frame_interval > last_frame_time + delay_ms):
                continue

            frames.append((rgb888_bytes, delay_ms))
            frame_count += 1
            last_frame_time += delay_ms

            # Check frame limit
            if frame_count >= frame_limit:
                break

        # Clean up iterator
        iterator.cleanup()

        logging.getLogger('animimg').info(f"Extracted {len(frames)} frames")
        return frames

    except Exception as e:
        logging.getLogger('animimg').error(f"Frame extraction failed: {e}")
        raise


def create_animimg_zip(frames: List[Tuple[bytes, float]], width: int, height: int) -> bytes:
    """Create ZIP file containing PNG frames and ESPHome YAML config."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        frame_files = []

        # Convert frames to PNG and add to ZIP
        for i, (rgb888_bytes, delay_ms) in enumerate(frames):
            # Convert RGB888 bytes to PIL Image
            img = Image.frombuffer('RGB', (width, height), rgb888_bytes, 'raw', 'RGB', 0, 1)

            # Save as PNG in memory
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_data = img_buffer.getvalue()

            # Add to ZIP with sequential filename
            filename = f"frame_{i+1:03d}.png"
            zip_file.writestr(filename, img_data)

            # Track for YAML config
            frame_files.append({
                'file': filename,
                'delay_ms': int(delay_ms)
            })

        # Create ESPHome YAML configuration
        yaml_config = create_esphome_yaml_config(frame_files)
        zip_file.writestr('animimg_config.yaml', yaml_config)

        # Add README with usage instructions
        readme = create_readme(len(frames), width, height)
        zip_file.writestr('README.txt', readme)

    return zip_buffer.getvalue()


def create_esphome_yaml_config(frame_files: List[Dict[str, Any]]) -> str:
    """Create ESPHome YAML configuration for animimg widget."""

    # Create proper ESPHome image configuration
    image_configs = []
    frame_ids = []
    for i, frame_info in enumerate(frame_files):
        # Generate valid ESPHome ID (no dots, underscores only)
        frame_id = f"frame_{i+1:03d}"
        frame_ids.append(frame_id)

        image_configs.append({
            'file': f"images/{frame_info['file']}",
            'id': frame_id,
            'type': 'RGB565'
        })

    # Calculate average frame duration in ms
    total_duration_ms = sum(frame_info['delay_ms'] for frame_info in frame_files)
    avg_duration_ms = int(total_duration_ms / len(frame_files)) if frame_files else 100

    # Create ESPHome configuration
    config = {
        'image': image_configs,
        'lvgl': {
            'pages': [
                {
                    'id': 'animation_page',
                    'widgets': [
                        {
                            'animimg': {
                                'id': 'my_animation',
                                'src': frame_ids,
                                'duration': f"{avg_duration_ms}ms",
                                'repeat_count': 'forever'
                            }
                        }
                    ]
                }
            ]
        }
    }

    # Add header comment
    yaml_content = f"""# ESPHome animimg configuration
# Generated by media-proxy animimg API
#
# Usage:
# 1. Create an 'images' directory in your ESPHome config directory
# 2. Copy all PNG files to the images/ directory
# 3. Add the image and lvgl sections to your ESPHome YAML
# 4. Customize the animimg widget properties as needed
#
# Frames: {len(frame_files)}
# Average frame duration: {avg_duration_ms}ms
# Total animation duration: {total_duration_ms}ms

"""

    yaml_content += yaml.dump(config, default_flow_style=False, sort_keys=False)

    return yaml_content


def create_readme(frame_count: int, width: int, height: int) -> str:
    """Create README with usage instructions."""
    return f"""ESPHome LVGL AnimImg Files
==========================

This ZIP contains {frame_count} frame images extracted and processed for ESPHome LVGL animimg widget.

Files included:
- frame_001.png through frame_{frame_count:03d}.png ({width}x{height} pixels each)
- animimg_config.yaml (ESPHome configuration template)
- README.txt (this file)

Usage Instructions:
1. Extract all PNG files to your ESPHome project directory
2. Copy the relevant sections from animimg_config.yaml to your ESPHome configuration
3. Customize the animimg widget properties (position, size, etc.) as needed
4. Flash your ESPHome device

ESPHome Documentation:
- Images: https://esphome.io/components/image/
- LVGL AnimImg: https://esphome.io/components/lvgl/widgets/#animimg

Generated by media-proxy animimg API
"""