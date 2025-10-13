# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import io
import logging
from urllib.parse import unquote_plus

import aiohttp
from aiohttp import web

from ...render.bbcode_renderer import render_bbcode_text
from ...utils.ha_client import HomeAssistantClient
from .placeholder import auto_contrast_color, parse_color


async def handle_homeassistant_request(request):
    """Handle GET /api/internal/homeassistant/{spec}

    Renders Home Assistant entity state or template to PNG image with BBCode support.

    URL patterns (extension required):
        # Entity mode (recommended - uses Template Helper)
        /homeassistant/64x64.png?entity=sensor.led_display_text
        /homeassistant/128x32/blue/white.png?entity=sensor.clock_display

        # Template mode (fallback - for quick prototyping)
        /homeassistant/64x64.png?template={{ states('sensor.temp') }}
        /homeassistant/800.png?template={{ now().strftime('%H:%M') }}

    BBCode support (in entity state or template result):
        [color=red]text[/color] or [red]text[/red] - Text color
        [font=8x16]text[/font] - Font size (5x8, 6x12, 8x16)
        [left], [center], [right] - Text alignment
        [b]text[/b] - Bold text (simulated)
        Plain text works without any tags (automatic word wrapping)
    """
    logger = logging.getLogger("homeassistant")

    # HEAD request optimization - return headers immediately without any processing
    if request.method == "HEAD":
        return web.Response(
            content_type="image/png",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",  # Don't cache - data changes
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    try:
        # Parse path: {width}x{height}[/{bg_color}[/{text_color}]].ext
        spec = request.match_info.get("spec", "")

        # Extension is required - extract and validate
        if not spec.endswith(".png"):
            return web.Response(status=400, text="Extension required. Use .png (e.g., /homeassistant/64x64.png)")

        # Remove .png extension before parsing
        spec = spec[:-4]
        parts = spec.split("/")

        # Parse dimensions
        if not parts:
            return web.Response(status=400, text="Invalid size format. Examples: 800x600, 64x64, 800 (square)")

        # Support both "800x600" and "800" (square)
        try:
            if "x" in parts[0]:
                dims = parts[0].split("x")
                width = int(dims[0])
                height = int(dims[1]) if len(dims) > 1 else width
            else:
                # Single number means square
                width = height = int(parts[0])
        except (ValueError, OverflowError) as e:
            logger.debug(f"Invalid dimensions: {e}")
            return web.Response(status=400, text="Invalid dimensions format")

        # Validate dimensions (10-4096px)
        if not (10 <= width <= 4096 and 10 <= height <= 4096):
            return web.Response(status=400, text="Dimensions must be 10-4096px")

        # Parse colors
        bg_color = parse_color(parts[1]) if len(parts) > 1 else (0, 0, 0)  # Default black
        text_color = parse_color(parts[2]) if len(parts) > 2 else auto_contrast_color(bg_color)

        # Check for required query parameters (entity OR template, not both)
        entity_id = request.query.get("entity")
        template = request.query.get("template")

        if not entity_id and not template:
            return web.Response(status=400, text="Missing required parameter: 'entity' or 'template'")

        if entity_id and template:
            return web.Response(status=400, text="Specify only one: 'entity' or 'template' (not both)")

        # Initialize HA client (creates its own session per request)
        ha_client = HomeAssistantClient()

        try:
            if entity_id:
                # Entity mode: Get entity state (e.g., Template Helper)
                entity_id = unquote_plus(entity_id)
                logger.debug(f"Fetching entity state: {entity_id}")
                state_obj = await ha_client.get_entity_state(entity_id)
                rendered_text = str(state_obj.get("state", ""))
            else:
                # Template mode: Render template directly
                template = unquote_plus(template)
                logger.debug(f"Rendering template: {template[:50]}...")
                rendered_text = await ha_client.render_template(template)

        except RuntimeError as e:
            # SUPERVISOR_TOKEN not available - render error message
            logger.warning(f"HA API not available: {e}")
            rendered_text = f"HA API Error:\n{e!s}"
        except aiohttp.ClientError as e:
            # API call failed - render error message
            logger.error(f"HA API call failed: {e}")
            rendered_text = f"Error:\n{e!s}"

        # Replace literal \n with actual newlines for multiline support
        rendered_text = rendered_text.replace("\\n", "\n")

        # Render text with BBCode support
        img = render_bbcode_text(
            rendered_text,
            width,
            height,
            default_fg=text_color,
            bg_color=bg_color,
        )

        # Convert PIL Image to PNG bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        png_data = buffer.getvalue()

        return web.Response(
            body=png_data,
            content_type="image/png",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",  # Don't cache - data changes
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    except ValueError as e:
        logger.debug(f"Invalid parameter: {e}")
        return web.Response(status=400, text="Invalid parameter value")
    except Exception as e:
        logger.error(f"Error generating HA image: {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")
