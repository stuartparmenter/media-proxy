# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import json
import os
from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "hw": {"prefer": "auto"},
    "video": {
        "expand_mode": 1,  # 0=never, 1=auto(limited->full), 2=force
        "fit": "auto-pad",  # "pad" | "cover" | "auto-pad" | "auto-cover"
        "autocrop": {
            "enabled": True,
            "probe_frames": 8,
            "luma_thresh": 22,
            "max_bar_ratio": 0.20,
            "min_bar_px": 2
        }
    },
    "playback": {"loop": True},
    # Still-image quality controls for tiny/low-DPI targets
    "image": {
        # method: lanczos | bicubic | bilinear | box | nearest
        "method": "lanczos",
        # Do resize in linear light (gamma-aware) to preserve tones
        "gamma_correct": True,
        # Optional mild sharpen after resize (0 disables)
        "unsharp": {"amount": 0.0, "radius": 0.6, "threshold": 2}
    },
    "log": {"send_ms": False, "rate_ms": 1000, "detail": False, "metrics": True},
    "net": {
        "win_timer_res": True,
        "spread_packets": True,
        "spread_max_fps": 60,
        "spread_min_ms": 3.0,
        "spread_max_sleeps": 0
    },
    # Optional resend policy for single-frame (non-looping) stills over UDP.
    "playback_still": {
        "burst": 3,
        "spacing_ms": 100,
        "tail_s": 2.0,
        "tail_hz": 2
    }
}


def load_config_file(path: str) -> Dict[str, Any]:
    """Load configuration from a file (YAML, TOML, or JSON)."""
    ext = os.path.splitext(path)[1].lower()
    if not os.path.exists(path):
        return {}
    
    try:
        if ext in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except Exception:
                print("[config] PyYAML not installed; skipping YAML.")
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        
        elif ext == ".toml":
            try:
                import tomllib  # 3.11+
            except Exception:
                try:
                    import tomli as tomllib  # type: ignore
                except Exception:
                    print("[config] tomllib/tomli not installed; skipping TOML.")
                    return {}
            with open(path, "rb") as f:
                return tomllib.load(f) or {}
        
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        
        else:
            print(f"[config] Unknown config extension: {ext}")
            return {}
            
    except Exception as e:
        print(f"[config] Failed to load {path}: {e}")
        return {}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge configuration dictionaries."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str = None) -> Dict[str, Any]:
    """Load configuration with defaults and optional file override."""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # Deep copy
    
    if path:
        deep_update(cfg, load_config_file(path))
    else:
        # Try auto-loading from current directory
        auto = "ws_ddp_proxy.yaml"
        if os.path.exists(auto):
            deep_update(cfg, load_config_file(auto))
    
    return cfg


class Config:
    """Configuration singleton."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load(self, path: str = None) -> None:
        """Load configuration from file."""
        self._config = load_config(path)
        
    def get(self, key: str = None, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'video.fit')."""
        if key is None:
            return self._config
            
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key path."""
        keys = key.split('.')
        target = self._config
        
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
            
        target[keys[-1]] = value
        
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary."""
        deep_update(self._config, updates)
        
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access."""
        return self.get(key)
        
    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dict-like assignment."""
        self.set(key, value)