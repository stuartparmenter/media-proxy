# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import numpy as np
from PIL import Image, ImageFilter
from typing import Tuple
from ..config import Config


def resize_pad_to_rgb_bytes(img: Image.Image, size: Tuple[int, int], config: Config = None) -> bytes:
    """Resize image to fit in target size with padding, return RGB888 bytes.

    For paletted images, preserves the palette through the resize operation
    to avoid quantization losses.
    """
    if config is None:
        config = Config()

    w, h = size

    # Choose resample method - optimized for LED displays
    cfg = config.get("image", {}) or {}
    method_s = str(cfg.get("method")).lower()
    M = Image.Resampling
    METHOD_MAP = {
        "lanczos": M.LANCZOS,
        "bicubic": M.BICUBIC,
        "bilinear": M.BILINEAR,
        "box":     M.BOX,       # area-average; good for big downscales, softer micro-contrast
        "nearest": M.NEAREST,   # use for pixel art/UI icons
    }

    # Compute scale factor to determine optimal resampling
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return b""
    scale = min(w / src_w, h / src_h) if src_w and src_h else 1.0

    # LED display optimized resampling strategy
    if method_s not in METHOD_MAP:
        # Auto-select based on scale for LED displays
        if scale >= 1.0:
            # Upscaling: use nearest to preserve pixel boundaries
            resample = M.NEAREST
        elif scale >= 0.5:
            # Modest downscaling: check if it's close to integer ratio
            x_ratio = src_w / w
            y_ratio = src_h / h
            avg_ratio = (x_ratio + y_ratio) / 2
            if abs(avg_ratio - round(avg_ratio)) < 0.1:
                # Close to integer ratio, use nearest for clean pixels
                resample = M.NEAREST
            else:
                # Non-integer ratio, use box for smooth downscaling
                resample = M.BOX
        else:
            # Heavy downscaling: use box to avoid aliasing
            resample = M.BOX
    else:
        resample = METHOD_MAP.get(method_s, M.NEAREST)

    # Check if this is a paletted image that we can preserve
    is_paletted = img.mode == 'P'
    original_palette = None
    if is_paletted and hasattr(img, 'palette') and img.palette:
        # Extract the palette as RGB values
        palette_data = img.palette.getdata()[1]  # (mode, palette_bytes)
        if len(palette_data) >= 3:  # Ensure we have RGB data
            # Convert palette bytes to RGB array
            palette_rgb = np.frombuffer(palette_data, dtype=np.uint8)
            if len(palette_rgb) % 3 == 0:
                original_palette = palette_rgb.reshape(-1, 3)

    # Gamma-aware resize (linear light)
    gamma_correct = bool(cfg.get("gamma_correct", True))

    def _to_linear_u8(rgb_u8: np.ndarray) -> np.ndarray:
        # sRGB -> linear via 1D LUT (fast & accurate for u8)
        if not hasattr(_to_linear_u8, "_lut"):
            # build once
            x = np.arange(256, dtype=np.float32) / 255.0
            lut = np.where(
                x <= 0.04045,
                x / 12.92,
                ((x + 0.055) / 1.055) ** 2.4
            )
            _to_linear_u8._lut = (lut * 65535.0 + 0.5).astype(np.uint16)  # promote for precision
        return _to_linear_u8._lut[rgb_u8]

    def _to_srgb_u8(lin_u16: np.ndarray) -> np.ndarray:
        if not hasattr(_to_srgb_u8, "_lut"):
            y = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
            lut = np.where(
                y <= 0.0031308,
                y * 12.92,
                1.055 * (y ** (1/2.4)) - 0.055
            )
            _to_srgb_u8._lut = (lut * 255.0 + 0.5).astype(np.uint8)
        return _to_srgb_u8._lut[lin_u16]

    # Handle paletted images specially to preserve colors
    transparency_index = None
    if is_paletted and original_palette is not None:
        # Track transparency index if present
        if "transparency" in img.info:
            transparency_index = img.info["transparency"]
        # Resize the palette indices, then convert to RGB using original palette
        # This preserves the exact original colors
        palette_indices = np.asarray(img, dtype=np.uint8)

        # Resize the index array using nearest neighbor to preserve palette integrity
        index_img = Image.fromarray(palette_indices, mode='L')
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized_indices = index_img.resize((new_w, new_h), resample=M.NEAREST)

        # Convert back to RGB using original palette
        indices_array = np.asarray(resized_indices, dtype=np.uint8)

        # Map indices to RGB using original palette
        # Clamp indices to valid palette range
        max_idx = len(original_palette) - 1
        safe_indices = np.clip(indices_array, 0, max_idx)
        rgb_array = original_palette[safe_indices]

        # Handle transparency after palette mapping
        if transparency_index is not None:
            # Create alpha mask: opaque where index != transparency_index
            alpha_mask = (indices_array != transparency_index).astype(np.uint8) * 255

            # Create RGBA image with alpha channel
            rgba_array = np.concatenate([rgb_array, alpha_mask[..., np.newaxis]], axis=-1)
            rgba_img = Image.fromarray(rgba_array.astype(np.uint8), mode='RGBA')

            # Blend with black background
            background = Image.new("RGB", rgba_img.size, (0, 0, 0))
            background.paste(rgba_img, mask=Image.fromarray(alpha_mask, mode='L'))
            im = background
        else:
            im = Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')
    else:
        # Convert to RGB early (handle "L", "LA", "RGBA", etc.)
        if img.mode != "RGB":
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                # Handle transparency with black background
                background = Image.new("RGB", img.size, (0, 0, 0))
                if img.mode == "P":
                    img = img.convert("RGBA")
                # After conversion, img is now RGBA so use the alpha channel as mask
                if img.mode in ("RGBA", "LA"):
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img)
                img = background
            else:
                img = img.convert("RGB")

        # Compute contain size once (no resampling yet)
        src_w, src_h = img.size
        if src_w == 0 or src_h == 0:
            return b""
        scale = min(w / src_w, h / src_h) if src_w and src_h else 1.0
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        new_size = (new_w, new_h)

        if gamma_correct and resample not in (M.NEAREST,):
            # Work in linear: resize each channel individually as 16-bit, then convert back to sRGB u8
            arr = np.asarray(img, dtype=np.uint8)
            lin = _to_linear_u8(arr)  # (H,W,3) uint16
            # Resize channels one by one (avoid merge("I;16"))
            r = Image.fromarray(lin[..., 0]).convert("I;16").resize(new_size, resample=resample)
            g = Image.fromarray(lin[..., 1]).convert("I;16").resize(new_size, resample=resample)
            b = Image.fromarray(lin[..., 2]).convert("I;16").resize(new_size, resample=resample)
            lin_res = np.stack(
                [np.array(r, dtype=np.uint16), np.array(g, dtype=np.uint16), np.array(b, dtype=np.uint16)],
                axis=-1
            )  # (new_h,new_w,3) uint16
            srgb_u8 = _to_srgb_u8(lin_res)  # (new_h,new_w,3) u8
            im = Image.fromarray(srgb_u8).convert("RGB")
        else:
            # Standard sRGB-space resize
            im = img.resize(new_size, resample=resample)

    # Optional mild sharpen to recover micro-contrast on very small outputs
    us = cfg.get("unsharp", {}) or {}
    amt = float(us.get("amount", 0.0))
    if amt > 0.0:
        radius = max(0.1, float(us.get("radius", 0.6)))
        thresh = max(0, int(us.get("threshold", 2)))
        im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amt * 100), threshold=thresh))

    # Pad to final size if needed
    if im.size != size:
        canvas = Image.new("RGB", size, (0, 0, 0))
        canvas.paste(im, ((w - im.size[0]) // 2, (h - im.size[1]) // 2))
        im = canvas
        
    return np.asarray(im, dtype=np.uint8).tobytes()


def rgb888_to_565_bytes(rgb_bytes: bytes, endian: str) -> bytes:
    """Convert RGB888 to RGB565 with proper quantization."""
    arr = np.frombuffer(rgb_bytes, dtype=np.uint8)
    if arr.size % 3 != 0:
        # Truncate any ragged tail (shouldn't happen for correctly sized frames)
        arr = arr[: (arr.size // 3) * 3]

    pix = arr.reshape((-1, 3)).astype(np.float32)

    # Simple quantization with proper rounding
    r = (pix[:, 0] / 255 * 31 + 0.5).astype(np.uint16)
    g = (pix[:, 1] / 255 * 63 + 0.5).astype(np.uint16)
    b = (pix[:, 2] / 255 * 31 + 0.5).astype(np.uint16)

    v = (r << 11) | (g << 5) | b
    if endian == "be":
        return v.byteswap().tobytes()
    else:
        return v.tobytes()