# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Output protocol implementations for different streaming formats."""

from .protocol import (
    OutputProtocol, BufferedOutputProtocol, StreamingOutputProtocol,
    OutputTarget, FrameMetadata, OutputMetrics, OutputProtocolFactory
)

# Import specific implementations to register them
from .ddp import DDPOutput

__all__ = [
    # Base protocols
    "OutputProtocol", "BufferedOutputProtocol", "StreamingOutputProtocol",
    # Data structures
    "OutputTarget", "FrameMetadata", "OutputMetrics",
    # Factory
    "OutputProtocolFactory",
    # Implementations
    "DDPOutput"
]