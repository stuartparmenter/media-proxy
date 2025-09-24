# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Dict, Any, Set, Optional, Type, Union
from enum import Enum


class FieldType(Enum):
    """Field data types for validation."""
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"


@dataclass
class FieldDef:
    """Definition of a control protocol field."""
    name: str
    field_type: FieldType
    required_for: Set[str]  # Operations that require this field
    description: str = ""
    validator: Optional[callable] = None


class ControlFields:
    """Centralized field definitions and validation for control protocol."""

    # Core stream fields
    OUT = FieldDef("out", FieldType.INTEGER, {"start_stream", "stop_stream", "update"},
                   "Output ID", lambda x: int(x) >= 0)
    WIDTH = FieldDef("w", FieldType.INTEGER, {"start_stream"},
                     "Display width", lambda x: int(x) > 0)
    HEIGHT = FieldDef("h", FieldType.INTEGER, {"start_stream"},
                      "Display height", lambda x: int(x) > 0)
    SOURCE = FieldDef("src", FieldType.STRING, {"start_stream"},
                      "Media source (file, URL, etc.)")

    # Network fields
    DDP_PORT = FieldDef("ddp_port", FieldType.INTEGER, set(),
                        "DDP output port", lambda x: 1 <= int(x) <= 65535)

    # Processing fields
    PACE = FieldDef("pace", FieldType.INTEGER, set(),
                    "Frame pacing frequency in Hz", lambda x: int(x) >= 0)
    EMA = FieldDef("ema", FieldType.FLOAT, set(),
                   "EMA filter alpha", lambda x: 0.0 <= float(x) <= 1.0)
    EXPAND = FieldDef("expand", FieldType.INTEGER, set(),
                      "FFmpeg color range expansion (0=never, 1=auto, 2=force)",
                      lambda x: int(x) in (0, 1, 2))
    LOOP = FieldDef("loop", FieldType.BOOLEAN, set(),
                    "Loop media playback")
    HARDWARE = FieldDef("hw", FieldType.STRING, set(),
                        "Hardware acceleration preference",
                        lambda x: str(x) in ("auto", "none", "cuda", "qsv", "vaapi", "videotoolbox", "d3d11va"))
    FORMAT = FieldDef("fmt", FieldType.STRING, set(),
                      "Pixel format preference",
                      lambda x: str(x) in ("rgb888", "rgb565", "rgb565le", "rgb565be"))
    FIT = FieldDef("fit", FieldType.STRING, set(),
                   "Video/image fit mode",
                   lambda x: str(x) in ("pad", "cover"))

    # Control fields
    TIMESTAMP = FieldDef("t", FieldType.INTEGER, set(),
                         "Timestamp for ping/pong")
    TYPE = FieldDef("type", FieldType.STRING, {"start_stream", "stop_stream", "update", "ping"},
                    "Message type")
    DEVICE_ID = FieldDef("device_id", FieldType.STRING, set(),
                         "Client device identifier")

    # All fields registry
    ALL_FIELDS = {
        "out": OUT, "w": WIDTH, "h": HEIGHT, "src": SOURCE,
        "ddp_port": DDP_PORT, "pace": PACE, "ema": EMA,
        "expand": EXPAND, "loop": LOOP, "hw": HARDWARE, "fmt": FORMAT, "fit": FIT,
        "t": TIMESTAMP, "type": TYPE, "device_id": DEVICE_ID
    }

    # Field groups for different operations
    REQUIRED_FOR_START = {"out", "w", "h", "src"}
    REQUIRED_FOR_STOP = {"out"}
    REQUIRED_FOR_UPDATE = {"out"}

    # Fields that can be updated/applied
    UPDATABLE_FIELDS = {"w", "h", "ddp_port", "src", "pace", "ema", "expand", "loop", "hw", "fmt", "fit"}
    APPLIED_FIELDS = {"src", "pace", "ema", "expand", "loop", "hw", "fmt", "fit"}

    @classmethod
    def validate_fields(cls, params: Dict[str, Any], operation: str) -> None:
        """Validate that required fields are present and valid for an operation."""
        # Get required fields for this operation
        required = getattr(cls, f"REQUIRED_FOR_{operation.upper()}", set())

        # Check missing fields
        missing = [f for f in required if f not in params]
        if missing:
            field_names = [cls.ALL_FIELDS[f].name for f in missing if f in cls.ALL_FIELDS]
            raise ValueError(f"{operation} requires {', '.join(field_names)} (missing: {', '.join(missing)})")

        # Validate field values
        for field_name, value in params.items():
            if field_name in cls.ALL_FIELDS:
                field_def = cls.ALL_FIELDS[field_name]
                if field_def.validator:
                    try:
                        if not field_def.validator(value):
                            raise ValueError(f"Invalid {field_name}: {value}")
                    except (ValueError, TypeError) as e:
                        raise ValueError(f"Invalid {field_name}: {value} ({e})")

    @classmethod
    def extract_applied_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters that were applied to the stream."""
        return {k: v for k, v in params.items() if k in cls.APPLIED_FIELDS}

    @classmethod
    def extract_updatable_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters that can be updated in a stream."""
        return {k: v for k, v in params.items() if k in cls.UPDATABLE_FIELDS}

    @classmethod
    def get_field_info(cls, field_name: str) -> Optional[FieldDef]:
        """Get field definition by name."""
        return cls.ALL_FIELDS.get(field_name)


if __name__ == "__main__":
    # Test basic validation
    test_params = {"out": 1, "w": 1920, "h": 1080, "src": "test.mp4"}
    try:
        ControlFields.validate_fields(test_params, "start")
        print("Basic field validation working correctly")
    except Exception as e:
        print(f"Basic field validation failed: {e}")

    # Test field-specific validation
    test_cases = [
        ({"out": 1, "expand": 1}, "Valid expand value"),
        ({"out": 1, "expand": 5}, "Invalid expand value (should fail)"),
        ({"out": 1, "fmt": "rgb888"}, "Valid format value"),
        ({"out": 1, "fmt": "invalid"}, "Invalid format value (should fail)"),
        ({"out": 1, "hw": "cuda"}, "Valid hardware value"),
        ({"out": 1, "hw": "invalid"}, "Invalid hardware value (should fail)"),
        ({"out": 1, "fit": "cover"}, "Valid fit value"),
        ({"out": 1, "fit": "pad"}, "Valid pad fit value"),
        ({"out": 1, "fit": "invalid"}, "Invalid fit value (should fail)"),
    ]

    for params, desc in test_cases:
        try:
            ControlFields.validate_fields(params, "update")
            print(f"  {desc}: PASS")
        except Exception as e:
            print(f"  {desc}: FAIL ({e})")

    applied = ControlFields.extract_applied_params({"src": "test.mp4", "pace": True, "fit": "cover", "out": 1})
    print(f"Applied params extraction: {applied}")

    print("Fields module validation complete")