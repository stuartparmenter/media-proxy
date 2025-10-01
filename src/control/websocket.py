# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import asyncio
import contextlib
import json
import logging
from typing import Dict, Any

from aiohttp import WSMsgType
from aiohttp.web_ws import WebSocketResponse

from .protocol import ControlProtocol, ControlSession
from ..streaming import create_streaming_task


def is_benign_disconnect(exc: BaseException) -> bool:
    """Check if an exception represents a benign disconnection."""
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in (64, 121):
        return True
    if isinstance(exc, ConnectionResetError):
        return True
    return False


class WebSocketControlProtocol(ControlProtocol):
    """aiohttp WebSocket implementation of the control protocol."""

    def __init__(self):
        super().__init__()

    async def send_response(self, session: ControlSession, response: Dict[str, Any]) -> bool:
        """Send a response back to the WebSocket client."""
        ws = getattr(session, 'websocket', None)
        if not ws or ws.closed:
            return False

        try:
            await ws.send_str(json.dumps(response, separators=(",", ":")))
            return True
        except (ConnectionResetError, OSError):
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

    async def handle_websocket(self, ws: WebSocketResponse, request):
        """Handle a WebSocket connection using aiohttp."""
        # Create session
        remote_addr = request.remote
        session = ControlSession(
            client_id=f"ws-{id(ws)}",
            client_ip=remote_addr if remote_addr else "unknown",
            websocket=ws
        )

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
                reason = str(exc) if exc.args else ''
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
            msg = await ws.receive()
            if msg.type != WSMsgType.TEXT:
                await self.send_error(session, "proto", "expected text message")
                await ws.close(code=4001, message=b"protocol")
                return False

            hello = json.loads(msg.data)
        except Exception as e:
            await self.send_error(session, "proto", f"invalid hello: {e}")
            with contextlib.suppress(Exception):
                await ws.close(code=4001, message=b"protocol")
            return False

        if hello.get("type") != "hello":
            await self.send_error(session, "proto", "expect 'hello' first")
            await ws.close(code=4001, message=b"protocol")
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

        logging.getLogger('websocket').debug(f"starting message loop for {session.client_ip}, ws.closed={ws.closed}")

        # Use aiohttp's built-in message iteration with receive_timeout for connection drops
        try:
            async for msg in ws:
                logging.getLogger('websocket').debug(f"received {msg.type} from {session.client_ip}, ws.closed={ws.closed}")

                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        msg_type = data.get("type")

                        if msg_type == "start_stream":
                            await self.handle_start_stream(session, data)
                        elif msg_type == "stop_stream":
                            await self.handle_stop_stream(session, data)
                        elif msg_type == "update":
                            await self.handle_update_stream(session, data)
                        elif msg_type == "ping":
                            await self.handle_ping(session, data)
                        else:
                            await self.send_error(session, "bad_type", f"unknown type {msg_type}")

                    except (ValueError, FileNotFoundError) as e:
                        await self.send_error(session, "bad_request", str(e))
                    except Exception as e:
                        await self.send_error(session, "server_error", str(e))

                elif msg.type == WSMsgType.ERROR:
                    logging.getLogger('websocket').warning(f'WebSocket error: {ws.exception()}')
                    break
                elif msg.type == WSMsgType.CLOSE:
                    logging.getLogger('websocket').info(f'WebSocket CLOSE received from {session.client_ip}')
                    break
                elif msg.type == WSMsgType.CLOSED:
                    logging.getLogger('websocket').info(f'WebSocket CLOSED detected for {session.client_ip}')
                    break
                elif msg.type == WSMsgType.PING:
                    logging.getLogger('websocket').debug(f'PING received from {session.client_ip}')
                elif msg.type == WSMsgType.PONG:
                    logging.getLogger('websocket').debug(f'PONG received from {session.client_ip}')
                else:
                    logging.getLogger('websocket').warning(f'Unknown WebSocket message type: {msg.type} from {session.client_ip}')

        except asyncio.TimeoutError:
            logging.getLogger('websocket').info(f'WebSocket receive timeout (20s) for {session.client_ip} - connection likely dropped')
            # Close the WebSocket to clean up resources
            if not ws.closed:
                await ws.close()
        except Exception as e:
            if is_benign_disconnect(e):
                logging.getLogger('websocket').info(f'WebSocket disconnect for {session.client_ip}: {e}')
            else:
                logging.getLogger('websocket').warning(f'WebSocket error for {session.client_ip}: {e}')

        logging.getLogger('websocket').info(f"message loop exited for {session.client_ip}, ws.closed={ws.closed}")


async def websocket_handler(request):
    """Handle WebSocket upgrade requests."""
    ws = WebSocketResponse(
        heartbeat=20.0,      # Send ping every 20 seconds (match websockets library)
        receive_timeout=20.0, # Timeout if no message in 20 seconds (match websockets library)
        autoping=True,       # Auto-respond to client pings with pongs
    )
    await ws.prepare(request)

    remote_addr = request.remote or "unknown"
    logging.getLogger('websocket').info(f"WebSocket connection established from {remote_addr}, heartbeat=20s, receive_timeout=20s")

    protocol = WebSocketControlProtocol()
    await protocol.handle_websocket(ws, request)

    return ws