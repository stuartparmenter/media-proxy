# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse


class FieldType(Enum):
    """Field data types for validation."""

    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    FLOAT = "float"


@dataclass
class FieldDef:
    """Definition of a field with type, validation, and defaults."""

    name: str
    field_type: type
    validator: Callable[[Any], bool] | None = None
    default_factory: Callable[..., Any] | None = None
    transformer: Callable[[Any], Any] | None = None
    description: str = ""

    def validate(self, value: Any) -> bool:
        """Validate a field value."""
        if self.validator:
            try:
                return self.validator(value)
            except (ValueError, TypeError):
                return False
        return True

    def get_default(self, config=None) -> Any:
        """Get default value for this field."""
        if self.default_factory:
            if config is not None:
                return self.default_factory(config)
            else:
                return self.default_factory()
        return None

    def transform(self, value: Any) -> Any:
        """Transform a field value."""
        if self.transformer:
            return self.transformer(value)
        return value


def _normalize_source_url(source: str) -> str:
    """Convert local file paths to file:// URLs, leave other URLs unchanged."""
    try:
        parsed = urlparse(source)
        if parsed.scheme:
            # Already has a scheme (http, https, file, etc.) - return as-is
            return source
        else:
            # No scheme - treat as local path and convert to file:// URL
            # Convert to absolute path and then to file:// URL
            abs_path = Path(source).resolve()
            return abs_path.as_uri()
    except Exception:
        # If anything fails, return the original source
        return source


class MediaFields:
    """Fields related to media content and source parameters."""

    WIDTH = FieldDef("w", int, lambda x: int(x) > 0, description="Display width")
    HEIGHT = FieldDef("h", int, lambda x: int(x) > 0, description="Display height")
    SOURCE = FieldDef(
        "src",
        str,
        validator=lambda x: len(str(x).strip()) > 0,
        transformer=_normalize_source_url,
        description="Media source (file, URL, etc.)",
    )

    LOOP = FieldDef(
        "loop", bool, default_factory=lambda config: config.get("playback.loop"), description="Loop media playback"
    )

    HARDWARE = FieldDef(
        "hw",
        str,
        lambda x: str(x) in ("auto", "none", "cuda", "qsv", "vaapi", "videotoolbox", "d3d11va"),
        default_factory=lambda config: config.get("hw.prefer"),
        description="Hardware acceleration preference",
    )

    FORMAT = FieldDef(
        "fmt",
        str,
        lambda x: str(x) in ("rgb888", "rgb565le", "rgb565be"),
        default_factory=lambda: "rgb888",
        description="Pixel format preference",
    )

    FIT = FieldDef(
        "fit",
        str,
        lambda x: str(x) in ("pad", "cover", "auto"),
        default_factory=lambda config: config.get("video.fit"),
        description="Video/image fit mode",
    )

    EXPAND = FieldDef(
        "expand",
        int,
        lambda x: int(x) in (0, 1, 2),
        default_factory=lambda config: config.get("video.expand_mode"),
        description="FFmpeg color range expansion (0=never, 1=auto, 2=force)",
    )

    LED_GAMMA = FieldDef(
        "led_gamma",
        str,
        lambda x: str(x) in ("none", "cie1931"),
        default_factory=lambda config: config.get("output.led_gamma"),
        description="LED display gamma compensation mode",
    )


class ProcessingFields:
    """Fields related to output processing and frame manipulation."""

    PACE = FieldDef(
        "pace", int, lambda x: int(x) >= 0, default_factory=lambda: 0, description="Frame pacing frequency in Hz"
    )

    EMA = FieldDef(
        "ema", float, lambda x: 0.0 <= float(x) <= 1.0, default_factory=lambda: 0.0, description="EMA filter alpha"
    )


class NetworkFields:
    """Fields related to transport and routing."""

    DDP_PORT = FieldDef(
        "ddp_port", int, lambda x: 1 <= int(x) <= 65535, default_factory=lambda: 4048, description="DDP output port"
    )

    DDP_HOST = FieldDef(
        "ddp_host", str, validator=lambda x: len(str(x).strip()) > 0, description="Target host for DDP packets"
    )

    OUT = FieldDef("out", int, lambda x: int(x) >= 0, description="Output ID")


class ProtocolFields:
    """Fields specific to control protocol infrastructure."""

    TYPE = FieldDef("type", str, description="Message type")
    TIMESTAMP = FieldDef("t", int, description="Timestamp for ping/pong")
    DEVICE_ID = FieldDef("device_id", str, description="Client device identifier")


class AllFields:
    """Centralized registry of all field definitions organized by domain."""

    # Create a flat registry for backward compatibility
    ALL_FIELDS: ClassVar[dict[str, Any]] = {}

    def __init_subclass__(cls):
        super().__init_subclass__()
        # Auto-populate the registry when the class is created
        cls._populate_registry()

    @classmethod
    def _populate_registry(cls):
        """Populate the flat field registry from domain classes."""
        for domain_class in [MediaFields, ProcessingFields, NetworkFields, ProtocolFields]:
            for attr_name in dir(domain_class):
                attr = getattr(domain_class, attr_name)
                if isinstance(attr, FieldDef):
                    cls.ALL_FIELDS[attr.name] = attr

    # Field groups for different operations (backward compatibility)
    REQUIRED_FOR_START: ClassVar[set[str]] = {"out", "w", "h", "src"}
    REQUIRED_FOR_STOP: ClassVar[set[str]] = {"out"}
    REQUIRED_FOR_UPDATE: ClassVar[set[str]] = {"out"}

    # Fields that can be updated/applied
    UPDATABLE_FIELDS: ClassVar[set[str]] = {
        "w",
        "h",
        "ddp_port",
        "ddp_host",
        "src",
        "pace",
        "ema",
        "expand",
        "loop",
        "hw",
        "fmt",
        "fit",
        "led_gamma",
    }
    APPLIED_FIELDS: ClassVar[set[str]] = {
        "src",
        "pace",
        "ema",
        "expand",
        "loop",
        "hw",
        "fmt",
        "fit",
        "ddp_host",
        "led_gamma",
    }

    @classmethod
    def validate_fields(cls, params: dict[str, Any], operation: str) -> None:
        """Validate that required fields are present and valid for an operation."""
        # Get required fields for this operation
        required: set[str] = getattr(cls, f"REQUIRED_FOR_{operation.upper()}", set())

        # Check missing fields
        missing = [f for f in required if f not in params]
        if missing:
            field_names = [cls.ALL_FIELDS[f].description for f in missing if f in cls.ALL_FIELDS]
            raise ValueError(f"{operation} requires {', '.join(field_names)} (missing: {', '.join(missing)})")

        # Validate field values
        for field_name, value in params.items():
            if field_name in cls.ALL_FIELDS:
                field_def = cls.ALL_FIELDS[field_name]
                if not field_def.validate(value):
                    raise ValueError(f"Invalid {field_name}: {value}")

    @classmethod
    def extract_applied_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Extract parameters that were applied to the stream."""
        return {k: v for k, v in params.items() if k in cls.APPLIED_FIELDS}

    @classmethod
    def extract_updatable_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        """Extract parameters that can be updated in a stream."""
        return {k: v for k, v in params.items() if k in cls.UPDATABLE_FIELDS}

    @classmethod
    def get_field_info(cls, field_name: str) -> FieldDef | None:
        """Get field definition by name."""
        return cls.ALL_FIELDS.get(field_name)


# Populate the registry
AllFields._populate_registry()

# Backward compatibility alias
ControlFields = AllFields


if __name__ == "__main__":
    # Test basic validation
    test_params = {"out": 1, "w": 1920, "h": 1080, "src": "test.mp4"}
    try:
        AllFields.validate_fields(test_params, "start")
        print("Basic field validation working correctly")
    except Exception as e:
        print(f"Basic field validation failed: {e}")

    # Test field-specific validation
    test_cases: list[tuple[dict[str, Any], str]] = [
        ({"out": 1, "expand": 1}, "Valid expand value"),
        ({"out": 1, "expand": 5}, "Invalid expand value (should fail)"),
        ({"out": 1, "fmt": "rgb888"}, "Valid format value"),
        ({"out": 1, "fmt": "invalid"}, "Invalid format value (should fail)"),
        ({"out": 1, "hw": "cuda"}, "Valid hardware value"),
        ({"out": 1, "hw": "invalid"}, "Invalid hardware value (should fail)"),
        ({"out": 1, "fit": "cover"}, "Valid fit value"),
        ({"out": 1, "fit": "pad"}, "Valid pad fit value"),
        ({"out": 1, "fit": "auto"}, "Valid auto fit value"),
        ({"out": 1, "fit": "invalid"}, "Invalid fit value (should fail)"),
    ]

    for params, desc in test_cases:
        try:
            AllFields.validate_fields(params, "update")
            print(f"  {desc}: PASS")
        except Exception as e:
            print(f"  {desc}: FAIL ({e})")

    applied = AllFields.extract_applied_params({"src": "test.mp4", "pace": 30, "fit": "cover", "out": 1})
    print(f"Applied params extraction: {applied}")

    print("Fields module validation complete")
