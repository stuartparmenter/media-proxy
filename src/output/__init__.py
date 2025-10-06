# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Output protocol implementations for different streaming formats."""

# Import specific implementations to register them
from .ddp import DDPOutput
from .protocol import (
    BufferedOutputProtocol,
    FrameMetadata,
    OutputMetrics,
    OutputProtocol,
    OutputProtocolFactory,
    OutputTarget,
)


__all__ = [
    "BufferedOutputProtocol",
    # Implementations
    "DDPOutput",
    "FrameMetadata",
    "OutputMetrics",
    # Base protocols
    "OutputProtocol",
    # Factory
    "OutputProtocolFactory",
    # Data structures
    "OutputTarget",
]
