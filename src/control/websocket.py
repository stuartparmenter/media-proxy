# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import contextlib
import json
import logging
from typing import Dict, Any
import websockets
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK, ConnectionClosedError

from .protocol import ControlProtocol, ControlSession
from ..streaming import create_streaming_task
from ..utils.helpers import truthy


def is_benign_disconnect(exc: BaseException) -> bool:
    """Check if an exception represents a benign disconnection."""
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (64, 121):
        return True
    return isinstance(exc, (ConnectionClosedOK, ConnectionClosedError))


class WebSocketControlProtocol(ControlProtocol):
    """WebSocket implementation of the control protocol."""
    
    def __init__(self):
        super().__init__()
        
    async def send_response(self, session: ControlSession, response: Dict[str, Any]) -> bool:
        """Send a response back to the WebSocket client."""
        ws = getattr(session, 'websocket', None)
        if not ws:
            return False
            
        try:
            await ws.send(json.dumps(response, separators=(",", ":")))
            return True
        except (ConnectionClosed, OSError):
            return False
            
    async def send_error(self, session: ControlSession, code: str, message: str) -> bool:
        """Send an error response to the WebSocket client."""
        return await self.send_response(session, {
            "type": "error", 
            "code": code, 
            "message": message
        })
        
    async def _create_stream_task(self, session: ControlSession, params: Dict[str, Any]) -> asyncio.Task:
        """Create and start a new streaming task."""
        return await create_streaming_task(session, params)
        
    async def handle_websocket(self, ws):
        """Handle a WebSocket connection."""
        # Create session
        remote_addr = getattr(ws, "remote_address", ("unknown",))
        session = ControlSession(
            client_id=f"ws-{id(ws)}",
            client_ip=remote_addr[0] if remote_addr else "unknown"
        )
        session.websocket = ws  # Store websocket reference
        
        try:
            # Handshake
            hello_successful = await self._handle_handshake(session)
            if not hello_successful:
                return
                
            logging.getLogger('websocket').info(f"hello from {session.client_ip} dev={session.device_id}")
            
            # Message loop
            await self._handle_message_loop(session)
            
        except Exception as exc:
            if is_benign_disconnect(exc):
                reason = getattr(exc, 'reason', '') or (getattr(exc, 'args', [''])[0])
                logging.getLogger('websocket').info(f"disconnect {session.client_ip} ({type(exc).__name__}: {reason})")
            else:
                logging.getLogger('websocket').warning(f"websocket error from {session.client_ip}: {exc!r}")
        finally:
            # Clean up all streams
            await self.cleanup_session(session)
            # Remove from sessions
            self.sessions.pop(session.client_id, None)
            
    async def _handle_handshake(self, session: ControlSession) -> bool:
        """Handle the initial handshake."""
        ws = session.websocket
        
        try:
            raw = await ws.recv()
            hello = json.loads(raw)
        except ConnectionClosed as e:
            logging.getLogger('websocket').info(f"disconnect during handshake from {session.client_ip} "
                  f"({getattr(e, 'code', '')} {getattr(e, 'reason', '')})")
            return False
        except Exception as e:
            await self.send_error(session, "proto", f"invalid hello: {e}")
            with contextlib.suppress(Exception):
                await ws.close(code=4001, reason="protocol")
            return False
            
        if hello.get("type") != "hello":
            await self.send_error(session, "proto", "expect 'hello' first")
            await ws.close(code=4001, reason="protocol")
            return False
            
        session.device_id = hello.get("device_id", "unknown")
        self.sessions[session.client_id] = session
        
        await self.send_response(session, {
            "type": "hello_ack", 
            "server_version": "media-proxy/1.0"
        })
        return True
        
    async def _handle_message_loop(self, session: ControlSession) -> None:
        """Handle incoming messages from the WebSocket."""
        ws = session.websocket
        
        async for raw in ws:
            try:
                msg = json.loads(raw)
                msg_type = msg.get("type")
                
                if msg_type == "start_stream":
                    await self.handle_start_stream(session, msg)
                elif msg_type == "stop_stream":
                    await self.handle_stop_stream(session, msg)
                elif msg_type == "update":
                    await self.handle_update_stream(session, msg)
                elif msg_type == "ping":
                    await self.handle_ping(session, msg)
                else:
                    await self.send_error(session, "bad_type", f"unknown type {msg_type}")
                    
            except (ValueError, FileNotFoundError) as e:
                await self.send_error(session, "bad_request", str(e))
            except Exception as e:
                await self.send_error(session, "server_error", str(e))


async def dispatch_websocket(websocket):
    """Dispatch WebSocket connections to the control protocol."""
    path = getattr(getattr(websocket, "request", None), "path", "/")
    
    try:
        if str(path).startswith("/control"):
            protocol = WebSocketControlProtocol()
            await protocol.handle_websocket(websocket)
        else:
            await websocket.close(code=4003, reason="unknown path")
    except Exception as e:
        if not is_benign_disconnect(e):
            logging.getLogger('websocket').warning(f"websocket handler error: {e!r}")
        with contextlib.suppress(Exception):
            await websocket.close()


async def start_websocket_server(host: str = "0.0.0.0", port: int = 8788):
    """Start the WebSocket server."""
    logging.getLogger('websocket').info(f"WebSocket server on ws://{host}:{port}/control")
    
    server = await websockets.serve(
        dispatch_websocket, 
        host, 
        port,
        max_size=2**22,
        compression=None,
        ping_interval=20, 
        ping_timeout=20,
        close_timeout=1.0
    )
    
    return server