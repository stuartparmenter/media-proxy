# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import asyncio
import logging
from ..utils.fields import ControlFields


class ControlSession:
    """Represents a client control session."""

    def __init__(self, client_id: str, client_ip: str):
        self.client_id = client_id
        self.client_ip = client_ip
        self.device_id: Optional[str] = None
        self.active_streams: Dict[Any, asyncio.Task] = {}  # stream_key -> task
        self.created_at = asyncio.get_event_loop().time()

    def __repr__(self):
        return f"ControlSession(id={self.client_id}, ip={self.client_ip}, device={self.device_id})"


class ControlProtocol(ABC):
    """Abstract base class for control protocols (WebSocket, HTTP, etc.)."""

    def __init__(self):
        self.sessions: Dict[str, ControlSession] = {}

    @abstractmethod
    async def send_response(self, session: ControlSession, response: Dict[str, Any]) -> bool:
        """Send a response back to the client. Returns True if successful."""
        pass

    @abstractmethod
    async def send_error(self, session: ControlSession, code: str, message: str) -> bool:
        """Send an error response to the client."""
        pass

    async def handle_start_stream(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle start_stream request. Base implementation with validation."""
        ControlFields.validate_fields(params, "start")
        out_id = int(params["out"])

        logging.getLogger('protocol').info(f"start_stream request: out={out_id} from session {session.client_id}")

        # Create output protocol to get stream key and handle conflicts
        output = self._create_output_protocol(params)
        stream_key = output.get_stream_key(session, params)

        # Stop existing stream for this key (session-level)
        await self._stop_stream_internal(session, stream_key, output)

        # Let output protocol handle conflict resolution if needed
        if stream_key is not None:
            await output.ensure_exclusive_access(stream_key)

        # Create and track stream
        stream_task = await self._create_stream_task(session, params)
        session_key = stream_key if stream_key is not None else f"stream_{id(stream_task)}"
        session.active_streams[session_key] = stream_task

        # Register stream with output protocol if needed
        if stream_key is not None:
            await output.register_stream(stream_key, stream_task)

        # Get applied params for response
        applied_params = ControlFields.extract_applied_params(params)
        await self.send_response(session, {
            "type": "ack",
            "out": out_id,
            "applied": applied_params
        })

    async def handle_stop_stream(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle stop_stream request."""
        ControlFields.validate_fields(params, "stop")
        out_id = int(params["out"])

        # Create output protocol to get stream key
        output = self._create_output_protocol(params)
        stream_key = output.get_stream_key(session, params)

        await self._stop_stream_internal(session, stream_key, output)
        await self.send_response(session, {"type": "ack", "out": out_id})

    async def handle_update_stream(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle update_stream request by stopping and restarting."""
        ControlFields.validate_fields(params, "update")
        out_id = int(params["out"])

        # Create output protocol to get stream key
        output = self._create_output_protocol(params)
        stream_key = output.get_stream_key(session, params)
        session_key = stream_key if stream_key is not None else f"out_{out_id}"

        if session_key not in session.active_streams:
            raise ValueError(f"update for unknown out={out_id} (no active stream)")

        # Build new start_stream params with updatable fields
        base_params = {"type": "start_stream", "out": out_id}
        updatable_params = ControlFields.extract_updatable_params(params)
        base_params.update(updatable_params)

        # Stop existing stream
        await self._stop_stream_internal(session, stream_key, output)

        # Let output protocol handle conflict resolution if needed
        if stream_key is not None:
            await output.ensure_exclusive_access(stream_key)

        # Create new stream
        stream_task = await self._create_stream_task(session, base_params)
        session.active_streams[session_key] = stream_task

        # Register with output protocol if needed
        if stream_key is not None:
            await output.register_stream(stream_key, stream_task)

        # Get applied params for response
        applied_params = ControlFields.extract_applied_params(updatable_params)
        await self.send_response(session, {
            "type": "ack",
            "out": out_id,
            "applied": applied_params
        })

    async def handle_ping(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle ping request."""
        await self.send_response(session, {"type": "pong", "t": params.get("t")})

    async def cleanup_session(self, session: ControlSession) -> None:
        """Clean up all streams for a session."""
        for session_key in list(session.active_streams.keys()):
            # We don't have the original params here, so we can't create the output protocol
            # Just stop the stream task directly
            task = session.active_streams.pop(session_key, None)
            if task and not task.done():
                logging.getLogger('protocol').info(f"stopping session stream {session_key}")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logging.getLogger('protocol').error(f"stream cleanup error {session_key}: {e!r}")

    @abstractmethod
    async def _create_stream_task(self, session: ControlSession, params: Dict[str, Any]) -> asyncio.Task:
        """Create and start a new streaming task. Must be implemented by subclasses."""
        pass

    def _create_output_protocol(self, params: Dict[str, Any]):
        """Create output protocol for stream management. Override if needed."""
        # Default to DDP - subclasses can override for different protocols
        from ..streaming.options import StreamOptions
        from ..config import Config
        from ..output.protocol import OutputProtocolFactory, OutputTarget

        config = Config()
        stream_options = StreamOptions.from_control_params(params, config)

        target = OutputTarget(
            host="127.0.0.1",  # Temporary, will be set properly in streaming task
            port=stream_options.ddp_port,
            output_id=stream_options.output_id,
            protocol="ddp"
        )

        return OutputProtocolFactory.create(target, stream_options)

    async def _stop_stream_internal(self, session: ControlSession, stream_key, output) -> None:
        """Internal method to stop a stream task."""
        session_key = stream_key if stream_key is not None else None
        if session_key is None:
            # Find by output_id for backwards compatibility
            # This is a fallback when stream_key is None
            return

        task = session.active_streams.pop(session_key, None)
        if task and not task.done():
            logging.getLogger('protocol').info(f"stopping session stream {session_key}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logging.getLogger('protocol').error(f"stream cleanup error {session_key}: {e!r}")

            # Clean up with output protocol if possible
            if stream_key is not None:
                await output.cleanup_stream(stream_key, task)
