# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

import tomllib
import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "hw": {"prefer": "auto"},
    "video": {
        "expand_mode": 2,  # 0=never, 1=auto(limited->full), 2=force
        "fit": "auto",  # "pad" | "cover" | "auto"
        "autocrop": {
            "enabled": False,  # Disabled by default for safety
            "probe_frames": 24,  # More frames for stability
            "luma_thresh": 16,  # Lower threshold for conservative detection
            "max_bar_ratio": 0.15,  # More conservative cropping limit
            "min_bar_px": 2,
        },
    },
    "playback": {"loop": True},
    "youtube": {
        "60fps": True,
        "cache": {
            "enabled": True,
            "max_size": 5242880,  # 5MB - use FFmpeg cache: protocol for videos under this size
        },
    },
    # Still-image quality controls for tiny/low-DPI targets
    "image": {
        # method: lanczos | bicubic | bilinear | box | nearest
        "method": "lanczos",
        # Do resize in linear light (gamma-aware) to preserve tones
        "gamma_correct": False,
        # Convert embedded ICC profiles to sRGB for consistent color
        "color_correction": True,
        # Optional mild sharpen after resize (0 disables)
        "unsharp": {"amount": 0.0, "radius": 0.6, "threshold": 2},
        # Frame caching for animated GIFs with loop=true
        "frame_cache_mb": 32,  # Max memory for cached frames (0 = disabled)
        "frame_cache_min_frames": 5,  # Only cache if animation has >= N frames
    },
    "log": {
        "send_ms": False,
        "rate_ms": 5000,
        "level": "info",
        "metrics": True,
    },
    "net": {
        "win_timer_res": True,
        "spread_packets": True,
        "spread_max_fps": 60,
        "spread_min_ms": 3.0,
        "spread_max_sleeps": 0,
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    },
    # Optional resend policy for single-frame (non-looping) stills over UDP.
    "playback_still": {
        "burst": 3,
        "spacing_ms": 100,
        "tail_s": 2.0,
        "tail_hz": 2,
    },
}


def load_config_file(path: str) -> dict[str, Any]:
    """Load configuration from a file (YAML, TOML, or JSON)."""
    logger = logging.getLogger("config")
    path_obj = Path(path)
    ext = path_obj.suffix.lower()

    if not path_obj.exists():
        return {}

    try:
        if ext in (".yaml", ".yml"):
            with Path(path).open(encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        elif ext == ".toml":
            with Path(path).open("rb") as f:
                return tomllib.load(f) or {}

        elif ext == ".json":
            with Path(path).open(encoding="utf-8") as f:
                return json.load(f) or {}

        else:
            logger.warning(f"Unknown config extension: {ext}")
            return {}

    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    """Deep merge configuration dictionaries."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | None = None) -> dict[str, Any]:
    """Load configuration with defaults and optional file override."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # Deep copy

    if path:
        deep_update(cfg, load_config_file(path))

    return cfg


class Config:
    """Configuration singleton."""

    _instance: ClassVar["Config | None"] = None
    _config: ClassVar[dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def load(cls, path: str | None = None) -> None:
        """Load configuration from file."""
        cls._config = load_config(path)

    @classmethod
    def get(cls, key: str | None = None) -> Any:
        """Get configuration value by key path (e.g., 'video.fit')."""
        if key is None:
            return cls._config

        keys = key.split(".")
        value = cls._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(f"Configuration key not found: {key}")

        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key path."""
        keys = key.split(".")
        target = self._config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def update(self, updates: dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        deep_update(self._config, updates)

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment."""
        self.set(key, value)
