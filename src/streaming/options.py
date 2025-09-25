# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import logging

from ..config import Config
from ..utils.fields import MediaFields, ProcessingFields, NetworkFields, AllFields
from ..utils.hardware import pick_hw_backend


@dataclass
class StreamOptions:
    """Strongly typed options for streaming operations."""

    # Core parameters (no defaults)
    output_id: int = field(metadata={'field_def': NetworkFields.OUT})
    width: int = field(metadata={'field_def': MediaFields.WIDTH})
    height: int = field(metadata={'field_def': MediaFields.HEIGHT})
    source: str = field(metadata={'field_def': MediaFields.SOURCE})
    ddp_port: int = field(metadata={'field_def': NetworkFields.DDP_PORT})

    # Media processing options (no defaults)
    loop: bool = field(metadata={'field_def': MediaFields.LOOP})
    expand: int = field(metadata={'field_def': MediaFields.EXPAND})
    hw: str = field(metadata={'field_def': MediaFields.HARDWARE})
    fit: str = field(metadata={'field_def': MediaFields.FIT})
    fmt: str = field(metadata={'field_def': MediaFields.FORMAT})

    # Processing options (no defaults)
    pace: int = field(metadata={'field_def': ProcessingFields.PACE})
    ema: float = field(metadata={'field_def': ProcessingFields.EMA})

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
        except ValueError:
            raise ValueError(f"Invalid IP address: {value}")
        self._target_ip = value

    @property
    def size(self) -> Tuple[int, int]:
        """Get size as (width, height) tuple."""
        return (self.width, self.height)

    @classmethod
    def from_control_params(cls, params: Dict[str, Any], config: Config) -> 'StreamOptions':
        """Create StreamOptions from control protocol parameters with validation and defaults."""

        # Extract and validate required parameters
        output_id = int(params["out"])
        width = int(params["w"])
        height = int(params["h"])
        source = str(params["src"])
        ddp_port = int(params.get("ddp_port", 4048))

        # Extract optional parameters directly - ControlFields validation is sufficient
        stream_config_dict = {}

        # Control field names match StreamOptions field names - no mapping needed
        optional_fields = ["loop", "expand", "hw", "pace", "ema", "fmt", "fit"]

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

        # Resolve hardware backend for format optimization
        if stream_config_dict["hw"] == "auto":
            resolved_hw_backend = pick_hw_backend("auto")
            stream_config_dict["hw"] = resolved_hw_backend
            logging.getLogger('streaming').debug(f"Resolved hw=auto to hw={resolved_hw_backend}")

        return cls(
            output_id=output_id,
            width=width,
            height=height,
            source=source,
            ddp_port=ddp_port,
            **stream_config_dict
        )



    def get_applied_params(self) -> Dict[str, Any]:
        """Get parameters that were applied to the stream (for control protocol response)."""
        return {
            "src": self.source,
            "pace": self.pace,
            "ema": self.ema,
            "expand": self.expand,
            "loop": self.loop,
            "hw": self.hw,
            "fmt": self.fmt,
            "fit": self.fit
        }

    def log_info(self, session_info: str) -> None:
        """Log streaming configuration info."""
        logging.getLogger('streaming').info(
            f"start_stream {session_info} out={self.output_id} "
            f"size={self.width}x{self.height} ddp_port={self.ddp_port} src={self.source} "
            f"pace={self.pace} ema={self.ema} expand={self.expand} "
            f"loop={self.loop} hw={self.hw} fmt={self.fmt} fit={self.fit}"
        )