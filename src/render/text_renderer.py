# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Text rendering utilities with BDF fonts and word wrapping."""

from pathlib import Path
from typing import Literal

import bdfparser
from PIL import Image, ImageFont


# Font sizes available
FontSize = Literal["5x8", "6x12", "8x16"]
Alignment = Literal["left", "center", "right"]

# Module-level font cache
_BDF_FONTS: dict[str, bdfparser.Font] = {}
_ICON_FONT: ImageFont.FreeTypeFont | None = None


def get_font_dir() -> Path:
    """Get the absolute path to the fonts directory."""
    return Path(__file__).parent.parent / "fonts"


def load_bdf_font(size: FontSize) -> bdfparser.Font:
    """Load and cache a Spleen BDF font.

    Args:
        size: Font size identifier (5x8, 6x12, or 8x16)

    Returns:
        Parsed BDF font object
    """
    if size not in _BDF_FONTS:
        font_path = get_font_dir() / f"spleen-{size}.bdf"
        _BDF_FONTS[size] = bdfparser.Font(str(font_path))
    return _BDF_FONTS[size]


def load_icon_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Load and cache Material Design Icons TTF font.

    Args:
        size: Font size in pixels (default: 16)

    Returns:
        PIL FreeTypeFont object
    """
    global _ICON_FONT
    if _ICON_FONT is None:
        font_path = get_font_dir() / "materialdesignicons-webfont.ttf"
        _ICON_FONT = ImageFont.truetype(str(font_path), size)
    return _ICON_FONT


def measure_text_width(text: str, font: bdfparser.Font) -> int:
    """Measure the pixel width of text with a BDF font.

    Args:
        text: Text to measure
        font: BDF font object

    Returns:
        Width in pixels
    """
    # bdfparser uses `draw()` to render, then we can get dimensions from the bitmap
    if not text:
        return 0

    # Get glyph metrics - use dwx0 (device width x) from meta
    width = 0
    for char in text:
        try:
            glyph = font.glyph(char)
            # dwx0 is the character advance width
            if glyph is not None:
                width += glyph.meta.get("dwx0", 8)
            else:
                width += font.headers.get("FONTBOUNDINGBOX", [8])[0]
        except KeyError:
            # Character not in font, use default glyph width
            width += font.headers.get("FONTBOUNDINGBOX", [8])[0]

    return width


def wrap_text(text: str, font: bdfparser.Font, max_width: int) -> list[str]:
    """Wrap text to fit within max_width pixels.

    Uses word-based wrapping with pixel-accurate measurements.

    Args:
        text: Text to wrap
        font: BDF font to use for measurements
        max_width: Maximum line width in pixels

    Returns:
        List of wrapped lines
    """
    if not text:
        return []

    lines = []

    # Handle explicit newlines first
    paragraphs = text.split("\n")

    for paragraph in paragraphs:
        if not paragraph:
            lines.append("")
            continue

        words = paragraph.split()
        current_line: list[str] = []

        for word in words:
            # Build test line
            test_line = " ".join([*current_line, word])
            test_width = measure_text_width(test_line, font)

            if test_width <= max_width:
                current_line.append(word)
            else:
                # Line would be too long
                if current_line:
                    # Save current line and start new one
                    lines.append(" ".join(current_line))
                    current_line = [word]
                else:
                    # Single word too long - add it anyway (no choice)
                    lines.append(word)
                    current_line = []

        # Add remaining words
        if current_line:
            lines.append(" ".join(current_line))

    return lines


def render_text(
    text: str,
    width: int,
    height: int,
    font_size: FontSize = "8x16",
    align: Alignment = "left",
    fg_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Render plain text to a PIL Image with word wrapping.

    Args:
        text: Text to render
        width: Image width in pixels
        height: Image height in pixels
        font_size: Spleen font size to use
        align: Text alignment (left, center, right)
        fg_color: Foreground (text) color RGB tuple
        bg_color: Background color RGB tuple

    Returns:
        PIL Image with rendered text
    """
    # Load font
    font = load_bdf_font(font_size)

    # Get font metrics - use 'fbby' (font bounding box y) for height
    font_height = font.headers.get("fbby", 16)

    # Wrap text
    lines = wrap_text(text, font, width)

    # Create image
    img = Image.new("RGB", (width, height), bg_color)

    # Render each line
    y = 0
    for line in lines:
        if y + font_height > height:
            break  # Out of vertical space

        # Render line to bitmap using bdfparser
        if line:
            line_bitmap_obj = font.draw(line)
            line_bitmap = line_bitmap_obj.todata(2)  # 2 = bitmap format
            # Use actual bitmap dimensions from bdfparser
            actual_width = line_bitmap_obj.width()
            actual_height = line_bitmap_obj.height()

            # Calculate x position based on alignment
            if align == "center":
                x = (width - actual_width) // 2
            elif align == "right":
                x = width - actual_width
            else:  # left
                x = 0

            # Convert bdfparser bitmap to PIL Image and composite
            # bdfparser returns bitmap as 2D list of integers (0 or non-zero)
            line_img = Image.new("RGB", (actual_width, actual_height), bg_color)
            pixels = line_img.load()

            if pixels is not None:  # Type guard for mypy
                for row_idx, row in enumerate(line_bitmap):
                    for col_idx, pixel in enumerate(row):
                        if pixel:
                            pixels[col_idx, row_idx] = fg_color

            # Paste onto main image
            img.paste(line_img, (x, y))

        y += font_height

    return img
