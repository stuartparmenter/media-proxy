# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio


class ControlSession:
    """Represents a client control session."""
    
    def __init__(self, client_id: str, client_ip: str):
        self.client_id = client_id
        self.client_ip = client_ip
        self.device_id: Optional[str] = None
        self.active_streams: Dict[int, asyncio.Task] = {}
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
        self._require_fields(params, ("out", "w", "h", "src"), "start_stream")
        
        out_id = int(params["out"])
        width = int(params["w"])
        height = int(params["h"])
        
        if width <= 0 or height <= 0:
            raise ValueError(f"start_stream requires positive w/h (got {width}x{height})")
            
        # Stop existing stream on this output
        await self._stop_stream_internal(session, out_id)
        
        # Delegate to implementation
        stream_task = await self._create_stream_task(session, params)
        session.active_streams[out_id] = stream_task
        
        await self.send_response(session, {
            "type": "ack", 
            "out": out_id,
            "applied": self._extract_applied_params(params)
        })
        
    async def handle_stop_stream(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle stop_stream request."""
        self._require_fields(params, ("out",), "stop_stream")
        out_id = int(params["out"])
        
        await self._stop_stream_internal(session, out_id)
        await self.send_response(session, {"type": "ack", "out": out_id})
        
    async def handle_update_stream(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle update_stream request by stopping and restarting."""
        self._require_fields(params, ("out",), "update")
        out_id = int(params["out"])
        
        if out_id not in session.active_streams:
            raise ValueError(f"update for unknown out={out_id} (no active stream)")
            
        # Build new start_stream params
        base_params = {"type": "start_stream", "out": out_id}
        applied = {}
        for key in ("w", "h", "ddp_port", "src", "pace", "ema", "expand", "loop", "hw", "fmt"):
            if key in params:
                base_params[key] = params[key]
                applied[key] = params[key]
                
        await self._stop_stream_internal(session, out_id)
        stream_task = await self._create_stream_task(session, base_params)
        session.active_streams[out_id] = stream_task
        
        await self.send_response(session, {"type": "ack", "out": out_id, "applied": applied})
        
    async def handle_ping(self, session: ControlSession, params: Dict[str, Any]) -> None:
        """Handle ping request."""
        await self.send_response(session, {"type": "pong", "t": params.get("t")})
        
    async def cleanup_session(self, session: ControlSession) -> None:
        """Clean up all streams for a session."""
        for out_id in list(session.active_streams.keys()):
            await self._stop_stream_internal(session, out_id)
            
    @abstractmethod
    async def _create_stream_task(self, session: ControlSession, params: Dict[str, Any]) -> asyncio.Task:
        """Create and start a new streaming task. Must be implemented by subclasses."""
        pass
        
    async def _stop_stream_internal(self, session: ControlSession, out_id: int) -> None:
        """Internal method to stop a stream task."""
        task = session.active_streams.pop(out_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[control] stream cleanup error out={out_id}: {e!r}")
                
    def _require_fields(self, params: Dict[str, Any], fields: tuple, operation: str) -> None:
        """Validate required fields are present."""
        missing = [f for f in fields if f not in params]
        if missing:
            raise ValueError(f"{operation} requires {', '.join(fields)} (missing: {', '.join(missing)})")
            
    def _extract_applied_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters that were applied to the stream."""
        applied = {}
        for key in ("src", "pace", "ema", "expand", "loop", "hw", "fmt"):
            if key in params:
                applied[key] = params[key]
        return applied