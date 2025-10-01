# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

"""Output protocol implementations for different streaming formats."""

from .protocol import (
    OutputProtocol, BufferedOutputProtocol,
    OutputTarget, FrameMetadata, OutputMetrics, OutputProtocolFactory
)

# Import specific implementations to register them
from .ddp import DDPOutput

__all__ = [
    # Base protocols
    "OutputProtocol", "BufferedOutputProtocol",
    # Data structures
    "OutputTarget", "FrameMetadata", "OutputMetrics",
    # Factory
    "OutputProtocolFactory",
    # Implementations
    "DDPOutput"
]