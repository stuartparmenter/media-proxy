# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import logging
from dataclasses import dataclass, field
from typing import Any

from ..config import Config
from ..utils.fields import MediaFields, NetworkFields, ProcessingFields
from ..utils.hardware import pick_hw_backend


@dataclass
class StreamOptions:
    """Strongly typed options for streaming operations."""

    # Core parameters (no defaults)
    output_id: int = field(metadata={"field_def": NetworkFields.OUT})
    width: int = field(metadata={"field_def": MediaFields.WIDTH})
    height: int = field(metadata={"field_def": MediaFields.HEIGHT})
    source: str = field(metadata={"field_def": MediaFields.SOURCE})
    ddp_port: int = field(metadata={"field_def": NetworkFields.DDP_PORT})

    # Media processing options (no defaults)
    loop: bool = field(metadata={"field_def": MediaFields.LOOP})
    expand: int = field(metadata={"field_def": MediaFields.EXPAND})
    hw: str = field(metadata={"field_def": MediaFields.HARDWARE})
    fit: str = field(metadata={"field_def": MediaFields.FIT})
    fmt: str = field(metadata={"field_def": MediaFields.FORMAT})
    led_gamma: str = field(metadata={"field_def": MediaFields.LED_GAMMA})

    # Processing options (no defaults)
    pace: int = field(metadata={"field_def": ProcessingFields.PACE})
    ema: float = field(metadata={"field_def": ProcessingFields.EMA})

    # Caching options (set internally based on config + video size)
    enable_cache: bool = field(init=False, default=False, repr=False)

    # Network configuration (set by create_streaming_task from session)
    _target_ip: str = field(init=False, default="", repr=False)

    @property
    def target_ip(self) -> str:
        return self._target_ip

    @target_ip.setter
    def target_ip(self, value: str):
        import ipaddress

        try:
            ipaddress.ip_address(value)
        except ValueError as e:
            raise ValueError(f"Invalid IP address: {value}") from e
        self._target_ip = value

    @property
    def size(self) -> tuple[int, int]:
        """Get size as (width, height) tuple."""
        return (self.width, self.height)

    @classmethod
    def from_control_params(cls, params: dict[str, Any], session=None) -> "StreamOptions":
        """Create StreamOptions from control protocol parameters with validation and defaults."""
        config = Config()

        # Extract and validate required parameters
        output_id = int(params["out"])
        width = int(params["w"])
        height = int(params["h"])
        source = MediaFields.SOURCE.transform(str(params["src"]))
        ddp_port = int(params.get("ddp_port", 4048))

        # Extract ddp_host (defaults to session client IP if not provided)
        ddp_host = params.get("ddp_host") if "ddp_host" in params else (session.client_ip if session else None)

        # Extract optional parameters directly - ControlFields validation is sufficient
        stream_config_dict = {}

        # Control field names match StreamOptions field names - no mapping needed
        optional_fields = ["loop", "expand", "hw", "pace", "ema", "fmt", "fit", "led_gamma"]

        for field_name in optional_fields:
            if field_name in params and params[field_name] is not None:
                # Direct assignment - validation already done by ControlFields
                stream_config_dict[field_name] = params[field_name]

        # Set defaults from config or field definitions for any missing values
        if "loop" not in stream_config_dict:
            stream_config_dict["loop"] = MediaFields.LOOP.get_default(config)
        if "expand" not in stream_config_dict:
            stream_config_dict["expand"] = MediaFields.EXPAND.get_default(config)
        if "hw" not in stream_config_dict:
            stream_config_dict["hw"] = MediaFields.HARDWARE.get_default(config)
        if "pace" not in stream_config_dict:
            stream_config_dict["pace"] = ProcessingFields.PACE.get_default()
        if "ema" not in stream_config_dict:
            stream_config_dict["ema"] = ProcessingFields.EMA.get_default()
        if "fmt" not in stream_config_dict:
            stream_config_dict["fmt"] = MediaFields.FORMAT.get_default()
        if "fit" not in stream_config_dict:
            stream_config_dict["fit"] = MediaFields.FIT.get_default(config)
        if "led_gamma" not in stream_config_dict:
            stream_config_dict["led_gamma"] = MediaFields.LED_GAMMA.get_default(config)

        # Resolve hardware backend for format optimization
        if stream_config_dict["hw"] == "auto":
            resolved_hw_backend = pick_hw_backend("auto")
            stream_config_dict["hw"] = resolved_hw_backend
            logging.getLogger("streaming").debug(f"Resolved hw=auto to hw={resolved_hw_backend}")

        # Create StreamOptions instance
        stream_options = cls(
            output_id=output_id, width=width, height=height, source=source, ddp_port=ddp_port, **stream_config_dict
        )

        # Set target_ip from extracted ddp_host
        if ddp_host:
            stream_options.target_ip = ddp_host

        return stream_options

    def get_applied_params(self) -> dict[str, Any]:
        """Get parameters that were applied to the stream (for control protocol response)."""
        return {
            "src": self.source,
            "pace": self.pace,
            "ema": self.ema,
            "expand": self.expand,
            "loop": self.loop,
            "hw": self.hw,
            "fmt": self.fmt,
            "fit": self.fit,
            "led_gamma": self.led_gamma,
        }

    def log_info(self, session_info: str) -> None:
        """Log streaming configuration info."""
        cache_str = f" cache={self.enable_cache}" if self.enable_cache else ""
        gamma_str = f" led_gamma={self.led_gamma}" if self.led_gamma != "none" else ""
        logging.getLogger("streaming").info(
            f"start_stream {session_info} out={self.output_id} "
            f"size={self.width}x{self.height} ddp_port={self.ddp_port} src={self.source} "
            f"pace={self.pace} ema={self.ema} expand={self.expand} "
            f"loop={self.loop} hw={self.hw} fmt={self.fmt} fit={self.fit}{gamma_str}{cache_str}"
        )
