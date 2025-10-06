# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import io
import logging

import numpy as np
from PIL import Image, ImageCms, ImageFilter
from PIL.ImageCms import Intent

from ..config import Config


# Module-level LUT caches for gamma correction
_LINEAR_LUT: np.ndarray | None = None
_SRGB_LUT: np.ndarray | None = None


def convert_to_srgb(img: Image.Image) -> Image.Image:
    """Convert image from embedded ICC profile to sRGB if needed.

    Args:
        img: PIL Image with potential ICC profile

    Returns:
        Image converted to sRGB color space
    """
    # Check if color correction is enabled
    cfg = Config().get("image")
    if not cfg.get("color_correction"):
        return img

    try:
        # Check if image has an ICC profile
        if "icc_profile" not in img.info:
            return img

        # Get the embedded ICC profile
        icc_profile = img.info["icc_profile"]
        if not icc_profile:
            return img

        # Create profile objects
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_profile))
        srgb_profile = ImageCms.createProfile("sRGB")

        # Check if source is already sRGB (avoid unnecessary conversion)
        try:
            source_colorspace = ImageCms.getProfileName(source_profile).lower()
            if "srgb" in source_colorspace:
                return img
        except Exception:  # noqa: S110  # Skip conversion if profile check fails
            pass

        # Get profile name for logging
        try:
            profile_name = ImageCms.getProfileName(source_profile)
        except Exception:
            profile_name = "unknown"

        logging.getLogger("processing").info(f"Converting {profile_name.strip()} -> sRGB")

        # Create transformation from source to sRGB
        transform = ImageCms.buildTransformFromOpenProfiles(
            source_profile, srgb_profile, img.mode, img.mode, renderingIntent=Intent.RELATIVE_COLORIMETRIC
        )

        # Apply the transformation
        converted_img = ImageCms.applyTransform(img, transform)
        assert converted_img is not None, "ImageCms.applyTransform must return an image"

        # Remove the old ICC profile
        converted_img.info = img.info.copy()
        if "icc_profile" in converted_img.info:
            del converted_img.info["icc_profile"]

        return converted_img

    except Exception as e:
        logging.getLogger("processing").warning(f"Color conversion failed: {e}")
        return img


def resize_pad_to_rgb_bytes(img: Image.Image, size: tuple[int, int], fit: str = "pad") -> bytes:
    """Resize image to target size using specified fit mode, return RGB888 bytes.

    Args:
        img: PIL Image to resize
        size: Target (width, height) tuple
        fit: Fit mode - "pad", "cover", or "auto"

    For paletted images, preserves the palette through the resize operation
    to avoid quantization losses.
    """
    config = Config()

    # Convert ICC profile to sRGB if needed
    img = convert_to_srgb(img)

    w, h = size

    # Choose resample method - optimized for LED displays
    cfg = config.get("image")
    method_s = str(cfg.get("method")).lower()
    method_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
        "bilinear": Image.Resampling.BILINEAR,
        "box": Image.Resampling.BOX,  # area-average; good for big downscales, softer micro-contrast
        "nearest": Image.Resampling.NEAREST,  # use for pixel art/UI icons
    }

    # Compute scale factor to determine optimal resampling
    src_w, src_h = img.size
    if src_w == 0 or src_h == 0:
        return b""
    scale = min(w / src_w, h / src_h) if src_w and src_h else 1.0

    # LED display optimized resampling strategy
    if method_s == "auto" or method_s not in method_map:
        # Auto-select based on scale for LED displays
        if scale >= 1.0:
            # Upscaling: use nearest to preserve pixel boundaries
            resample = Image.Resampling.NEAREST
        elif scale >= 0.5:
            # Modest downscaling: check if it's close to integer ratio
            x_ratio = src_w / w
            y_ratio = src_h / h
            avg_ratio = (x_ratio + y_ratio) / 2
            # Close to integer ratio: use nearest for clean pixels; otherwise use box for smooth downscaling
            resample = Image.Resampling.NEAREST if abs(avg_ratio - round(avg_ratio)) < 0.1 else Image.Resampling.BOX
        else:
            # Heavy downscaling: use box to avoid aliasing
            resample = Image.Resampling.BOX
    else:
        resample = method_map[method_s]

    # Check if this is a paletted image that we can preserve
    is_paletted = img.mode == "P"
    original_palette = None
    if is_paletted and hasattr(img, "palette") and img.palette:
        # Extract the palette as RGB values
        palette_data = img.palette.getdata()[1]  # (mode, palette_bytes)
        if len(palette_data) >= 3:  # Ensure we have RGB data
            # Convert palette bytes to RGB array
            palette_rgb = np.frombuffer(palette_data, dtype=np.uint8)  # type: ignore[call-overload]  # numpy stubs limitation with bytes/buffer input
            if len(palette_rgb) % 3 == 0:
                original_palette = palette_rgb.reshape(-1, 3)

    # Gamma-aware resize (linear light)
    gamma_correct = bool(cfg.get("gamma_correct"))

    def _to_linear_u8(rgb_u8: np.ndarray) -> np.ndarray:
        # sRGB -> linear via 1D LUT (fast & accurate for u8)
        global _LINEAR_LUT
        if _LINEAR_LUT is None:
            # build once
            x = np.arange(256, dtype=np.float32) / 255.0
            lut = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
            _LINEAR_LUT = (lut * 65535.0 + 0.5).astype(np.uint16)  # promote for precision
        return _LINEAR_LUT[rgb_u8]

    def _to_srgb_u8(lin_u16: np.ndarray) -> np.ndarray:
        global _SRGB_LUT
        if _SRGB_LUT is None:
            y = np.linspace(0.0, 1.0, 65536, dtype=np.float32)
            lut = np.where(y <= 0.0031308, y * 12.92, 1.055 * (y ** (1 / 2.4)) - 0.055)
            _SRGB_LUT = (lut * 255.0 + 0.5).astype(np.uint8)
        return _SRGB_LUT[lin_u16]

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
        index_img = Image.fromarray(palette_indices, mode="L")
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized_indices = index_img.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

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
            rgba_img = Image.fromarray(rgba_array.astype(np.uint8), mode="RGBA")

            # Blend with black background
            background = Image.new("RGB", rgba_img.size, (0, 0, 0))
            background.paste(rgba_img, mask=Image.fromarray(alpha_mask, mode="L"))
            im = background
        else:
            im = Image.fromarray(rgb_array.astype(np.uint8), mode="RGB")
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

        # Compute resize parameters based on fit mode
        src_w, src_h = img.size
        if src_w == 0 or src_h == 0:
            return b""

        # Calculate aspect ratios for auto mode
        src_ratio = src_w / src_h
        target_ratio = w / h

        if fit == "cover":
            # Scale to fill, may crop content
            scale = max(w / src_w, h / src_h) if src_w and src_h else 1.0
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))
        elif fit == "auto":
            # Smart fit: direct scale if aspect ratios match, else pad
            if abs(src_ratio - target_ratio) < 0.01:
                # Aspect ratios match - scale directly to target size
                new_w, new_h = w, h
            else:
                # Aspect ratios differ - use contain scaling (like pad mode)
                scale = min(w / src_w, h / src_h) if src_w and src_h else 1.0
                new_w = max(1, int(round(src_w * scale)))
                new_h = max(1, int(round(src_h * scale)))
        else:  # "pad" mode
            # Scale to fit, preserve aspect ratio
            scale = min(w / src_w, h / src_h) if src_w and src_h else 1.0
            new_w = max(1, int(round(src_w * scale)))
            new_h = max(1, int(round(src_h * scale)))

        new_size = (new_w, new_h)

        if gamma_correct and resample not in (Image.Resampling.NEAREST,):
            # Work in linear: resize each channel individually as 16-bit, then convert back to sRGB u8
            arr = np.asarray(img, dtype=np.uint8)
            lin = _to_linear_u8(arr)  # (H,W,3) uint16
            # Resize channels one by one (avoid merge("I;16"))
            r = Image.fromarray(lin[..., 0]).convert("I;16").resize(new_size, resample=resample)
            g = Image.fromarray(lin[..., 1]).convert("I;16").resize(new_size, resample=resample)
            b = Image.fromarray(lin[..., 2]).convert("I;16").resize(new_size, resample=resample)
            lin_res = np.stack(
                [np.array(r, dtype=np.uint16), np.array(g, dtype=np.uint16), np.array(b, dtype=np.uint16)], axis=-1
            )  # (new_h,new_w,3) uint16
            srgb_u8 = _to_srgb_u8(lin_res)  # (new_h,new_w,3) u8
            im = Image.fromarray(srgb_u8).convert("RGB")
        else:
            # Standard sRGB-space resize
            im = img.resize(new_size, resample=resample)

    # Optional mild sharpen to recover micro-contrast on very small outputs
    us = cfg.get("unsharp")
    amt = float(us.get("amount"))
    if amt > 0.0:
        radius = max(0.1, float(us.get("radius")))
        thresh = max(0, int(us.get("threshold")))
        im = im.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amt * 100), threshold=thresh))

    # Final cropping/padding based on fit mode
    if im.size != size:
        if fit == "cover":
            # Crop to exact size (center crop)
            left = max(0, (im.size[0] - w) // 2)
            top = max(0, (im.size[1] - h) // 2)
            right = left + w
            bottom = top + h
            im = im.crop((left, top, right, bottom))
        else:
            # For "pad" and "auto" modes - add padding to reach target size
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
