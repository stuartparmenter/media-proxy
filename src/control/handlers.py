# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Type
import logging


class ControlHandler(ABC):
    """Abstract base class for protocol-specific control handlers."""

    @staticmethod
    @abstractmethod
    def detect(params: Dict[str, Any]) -> bool:
        """Detect if this handler should be used for the given parameters."""
        pass

    @staticmethod
    @abstractmethod
    def get_stream_key(session, params: Dict[str, Any]) -> Any:
        """Generate a stream key for the given session and parameters."""
        pass


class DDPControlHandler(ControlHandler):
    """Handler for DDP protocol control operations."""

    @staticmethod
    def detect(params: Dict[str, Any]) -> bool:
        """Detect DDP protocol based on presence of out parameter (with optional ddp_port)."""
        # DDP protocol is identified by the presence of 'out' parameter
        # start_stream/update_stream include ddp_port, stop_stream only includes out
        return "out" in params

    @staticmethod
    def get_stream_key(session, params: Dict[str, Any]) -> Tuple[str, int]:
        """Generate DDP stream key as (target_ip, output_id) tuple."""
        # TODO: Add support for explicit target_ip parameter in future
        # For now, use the client IP as the target (existing behavior)
        target_ip = session.client_ip
        output_id = int(params["out"])
        return (target_ip, output_id)


class ControlHandlerRegistry:
    """Registry for control handlers with automatic protocol detection."""

    _handlers: list[Type[ControlHandler]] = [DDPControlHandler]

    @classmethod
    def detect_handler(cls, params: Dict[str, Any]) -> Optional[Type[ControlHandler]]:
        """Find the appropriate handler for the given parameters."""
        for handler_class in cls._handlers:
            if handler_class.detect(params):
                return handler_class
        return None

    @classmethod
    def get_stream_key(cls, session, params: Dict[str, Any]) -> Any:
        """Get stream key using the appropriate handler."""
        handler = cls.detect_handler(params)
        if handler:
            return handler.get_stream_key(session, params)

        # Fallback: no stream key (no conflicts)
        logging.getLogger('control').debug(f"No handler found for params: {list(params.keys())}")
        return None

    @classmethod
    def register_handler(cls, handler_class: type) -> None:
        """Register a new control handler."""
        if handler_class not in cls._handlers:
            cls._handlers.append(handler_class)