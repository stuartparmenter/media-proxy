# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""BBCode text rendering with support for colors, fonts, alignment, and icons."""

from typing import Any

import bbcode
from PIL import Image, ImageColor

from .text_renderer import (
    Alignment,
    FontSize,
    load_bdf_font,
    render_text,
    wrap_text,
)


def parse_color(color_str: str) -> tuple[int, int, int]:
    """Parse color string to RGB tuple.

    Supports:
    - Named colors: 'red', 'blue', 'white', etc.
    - Hex colors: '#ff0000', '#f00', 'ff0000', 'f00'

    Args:
        color_str: Color string

    Returns:
        RGB tuple (r, g, b)
    """
    try:
        # Prepend # if needed for hex colors
        if (
            not color_str.startswith("#")
            and len(color_str) in (3, 6)
            and all(c in "0123456789abcdefABCDEF" for c in color_str)
        ):
            color_str = f"#{color_str}"

        color = ImageColor.getrgb(color_str)
        # ImageColor.getrgb can return RGBA - extract only RGB
        return (color[0], color[1], color[2])
    except ValueError:
        # Fallback to white if color invalid
        return (255, 255, 255)


def render_bbcode_text(
    text: str,
    width: int,
    height: int,
    default_font: FontSize = "5x8",
    default_align: Alignment = "left",
    default_fg: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render BBCode-formatted text to a PIL Image.

    Supported BBCode tags:
    - [color=red]...[/color] or [red]...[/red] - Set text color
    - [font=5x8]...[/font] - Set Spleen font size (5x8, 6x12, 8x16)
    - [left], [center], [right] - Set text alignment for following lines
    - [b]...[/b] - Bold text (simulated with double-draw)
    - Plain text (no tags) works as-is with word wrapping

    Args:
        text: BBCode-formatted text
        width: Image width in pixels
        height: Image height in pixels
        default_font: Default Spleen font size
        default_align: Default text alignment
        default_fg: Default foreground color RGB tuple
        bg_color: Background color RGB tuple

    Returns:
        PIL Image with rendered text
    """
    # If no BBCode tags, render as plain text
    if "[" not in text:
        return render_text(text, width, height, default_font, default_align, default_fg, bg_color)

    # Parse BBCode
    parser = bbcode.Parser()

    # Track state
    segments: list[dict[str, Any]] = []
    current_font = default_font
    current_color = default_fg
    current_align = default_align
    current_bold = False

    # Custom BBCode tag handlers
    def handle_color(tag_name: str, value: str, options: dict, parent: Any, context: Any) -> str:
        nonlocal current_color
        # If tag_name is a color name (like "red"), use it; otherwise use the color option
        color_value = tag_name if tag_name != "color" else options.get("color", value)
        current_color = parse_color(color_value)
        return value

    def handle_font(tag_name: str, value: str, options: dict, parent: Any, context: Any) -> str:
        nonlocal current_font
        font_value = options.get("font", "8x16")
        if font_value in ("5x8", "6x12", "8x16"):
            current_font = font_value
        return value

    def handle_align(tag_name: str, value: str, options: dict, parent: Any, context: Any) -> str:
        nonlocal current_align
        if tag_name in ("left", "center", "right"):
            current_align = tag_name  # type: ignore
        return value

    def handle_bold(tag_name: str, value: str, options: dict, parent: Any, context: Any) -> str:
        nonlocal current_bold
        current_bold = True
        result = value
        current_bold = False
        return result

    # Register tag handlers
    parser.add_formatter("color", handle_color)
    parser.add_formatter("font", handle_font)
    parser.add_formatter("left", handle_align)
    parser.add_formatter("center", handle_align)
    parser.add_formatter("right", handle_align)
    parser.add_formatter("b", handle_bold)

    # Register short color tag names (red, blue, green, etc.)
    # These are treated as standalone tags, not options
    for color_name in [
        "red",
        "green",
        "blue",
        "yellow",
        "white",
        "black",
        "cyan",
        "magenta",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
    ]:
        parser.add_formatter(color_name, handle_color)

    # Simplified approach: use tokenizer directly for more control
    tokens = parser.tokenize(text)

    # Color tag names for recognition
    color_tags = {
        "red",
        "green",
        "blue",
        "yellow",
        "white",
        "black",
        "cyan",
        "magenta",
        "orange",
        "purple",
        "pink",
        "brown",
        "gray",
        "grey",
        "color",
    }

    # Build segments with styling
    for token_type, tag_name, tag_opts, token_text in tokens:
        if token_type == 1:  # Opening tag
            if tag_name in color_tags:
                # For [color=X], use tag_opts; for [red], use tag_name itself
                color_value = tag_opts.get("color", tag_name if tag_name != "color" else "white")
                current_color = parse_color(color_value)
            elif tag_name == "font":
                font_val = tag_opts.get("font", default_font)
                if font_val in ("5x8", "6x12", "8x16"):
                    current_font = font_val  # type: ignore
            elif tag_name in ("left", "center", "right"):
                current_align = tag_name  # type: ignore
            elif tag_name == "b":
                current_bold = True

        elif token_type == 2:  # Closing tag
            if tag_name in color_tags:
                current_color = default_fg
            elif tag_name == "font":
                current_font = default_font
            elif tag_name == "b":
                current_bold = False

        elif token_type == 3:  # Newline
            segments.append(
                {
                    "text": "\n",
                    "font": current_font,
                    "color": current_color,
                    "align": current_align,
                    "bold": current_bold,
                }
            )

        elif token_type == 4:  # Text
            if token_text:
                segments.append(
                    {
                        "text": token_text,
                        "font": current_font,
                        "color": current_color,
                        "align": current_align,
                        "bold": current_bold,
                    }
                )

    # Render segments to image
    img = Image.new("RGB", (width, height), bg_color)

    # Group segments by line/alignment
    y = 0
    current_line_segments: list[dict] = []
    current_line_align = default_align

    def render_line() -> None:
        nonlocal y, current_line_segments, current_line_align

        if not current_line_segments:
            return

        # Combine text from segments with same font
        line_text = "".join(seg["text"] for seg in current_line_segments)
        if not line_text.strip():
            # Empty line, just advance y
            font = load_bdf_font(current_line_segments[0]["font"])
            font_height = font.headers.get("fbby", 16)
            y += font_height
            current_line_segments = []
            return

        # For now, use first segment's font/color for entire line
        # TODO: In future, support multi-styled segments per line
        first_seg = current_line_segments[0]
        font = load_bdf_font(first_seg["font"])
        color = first_seg["color"]
        bold = first_seg["bold"]

        font_height = font.headers.get("fbby", 16)

        if y + font_height > height:
            return  # Out of space

        # Word wrap the line
        wrapped = wrap_text(line_text, font, width)

        for wrapped_line in wrapped:
            if y + font_height > height:
                break

            if wrapped_line:
                line_bitmap_obj = font.draw(wrapped_line)
                line_bitmap = line_bitmap_obj.todata(2)
                # Use actual bitmap dimensions
                actual_width = line_bitmap_obj.width()
                actual_height = line_bitmap_obj.height()

                # Calculate x based on alignment
                if current_line_align == "center":
                    x = (width - actual_width) // 2
                elif current_line_align == "right":
                    x = width - actual_width
                else:
                    x = 0

                # Render bitmap
                line_img = Image.new("RGB", (actual_width, actual_height), bg_color)
                pixels = line_img.load()

                if pixels is not None:  # Type guard for mypy
                    for row_idx, row in enumerate(line_bitmap):
                        for col_idx, pixel in enumerate(row):
                            if pixel:
                                pixels[col_idx, row_idx] = color

                # Paste onto main image
                img.paste(line_img, (x, y))

                # If bold, draw again with 1px offset (no mask needed for opaque image)
                if bold:
                    img.paste(line_img, (x + 1, y))

            y += font_height

        current_line_segments = []

    # Process segments
    for seg in segments:
        if seg["text"] == "\n":
            render_line()
            current_line_align = seg["align"]
        else:
            # Check if alignment changed
            if seg["align"] != current_line_align and current_line_segments:
                render_line()
                current_line_align = seg["align"]
            elif seg["align"] != current_line_align:
                current_line_align = seg["align"]

            current_line_segments.append(seg)

    # Render final line
    render_line()

    return img
