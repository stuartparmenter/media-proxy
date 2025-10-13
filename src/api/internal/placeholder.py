# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import io
import logging
from urllib.parse import unquote_plus

from aiohttp import web
from PIL import Image, ImageColor, ImageDraw, ImageFont


def parse_color(color_str: str) -> tuple[int, int, int]:
    """Parse CSS color name or hex to RGB using Pillow's ImageColor"""
    try:
        # Prepend # if needed for hex colors (3 or 6 digits without #)
        if not color_str.startswith("#") and len(color_str) in (3, 6):
            color_str = f"#{color_str}"
        color = ImageColor.getrgb(color_str)
        # ImageColor.getrgb can return RGBA - extract only RGB
        return (color[0], color[1], color[2])
    except ValueError:
        # Fallback to gray if color invalid
        return (204, 204, 204)


def auto_contrast_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white based on background luminance (WCAG formula)"""
    r, g, b = [c / 255.0 for c in bg_rgb]
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (255, 255, 255) if luminance < 0.5 else (0, 0, 0)


def generate_placeholder_png(width: int, height: int, bg_color: tuple, text_color: tuple, text: str) -> bytes:
    """Generate placeholder PNG image"""
    # Create image with background color
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Use default PIL font
    font = ImageFont.load_default()

    # Handle multiline text (newline separator)
    lines = text.split("\n")

    # Calculate text size and position for centering
    line_heights = [
        draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines
    ]
    total_height = sum(line_heights)
    line_height = total_height // len(lines) if lines else 0

    y_offset = (height - total_height) // 2

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y_offset), line, fill=text_color, font=font)
        y_offset += line_height

    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


async def handle_placeholder_request(request):
    """Handle GET /api/internal/placeholder/{spec}

    URL patterns (extension required):
        /placeholder/64x64.png
        /placeholder/600x400/orange/white.png
        /placeholder/600x400/ff0000.png
        /placeholder/800.png?text=Hello+World
    """
    # HEAD request optimization - return headers immediately without any processing
    if request.method == "HEAD":
        return web.Response(
            content_type="image/png",
            headers={
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
            },
        )

    try:
        # Parse path: {width}x{height}[/{bg_color}[/{text_color}]].ext
        spec = request.match_info.get("spec", "")

        # Extension is required - extract and validate
        if not spec.endswith(".png"):
            return web.Response(status=400, text="Extension required. Use .png (e.g., /placeholder/64x64.png)")

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
            logging.getLogger("placeholder").debug(f"Invalid dimensions: {e}")
            return web.Response(status=400, text="Invalid dimensions format")

        # Validate dimensions (10-4096px)
        if not (10 <= width <= 4096 and 10 <= height <= 4096):
            return web.Response(status=400, text="Dimensions must be 10-4096px")

        # Parse colors
        bg_color = parse_color(parts[1]) if len(parts) > 1 else (204, 204, 204)  # Default gray
        text_color = parse_color(parts[2]) if len(parts) > 2 else auto_contrast_color(bg_color)

        # Parse text from query params (or default to dimensions)
        text = unquote_plus(request.query.get("text", f"{width}x{height}"))
        # Replace literal \n with actual newlines for multiline support
        text = text.replace("\\n", "\n")

        # Generate PNG
        png_data = generate_placeholder_png(width, height, bg_color, text_color, text)

        return web.Response(
            body=png_data,
            content_type="image/png",
            headers={
                "Cache-Control": "public, max-age=31536000",  # Cache for 1 year
            },
        )

    except ValueError as e:
        logging.getLogger("placeholder").debug(f"Invalid parameter: {e}")
        return web.Response(status=400, text="Invalid parameter value")
    except Exception as e:
        logging.getLogger("placeholder").error(f"Error generating placeholder: {e}", exc_info=True)
        return web.Response(status=500, text="Internal server error")
