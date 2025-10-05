# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import logging
from aiohttp import web

from ..control.websocket import websocket_handler
from .convert.animimg import handle_animimg_request
from .internal.placeholder import handle_placeholder_request


async def health_check_handler(request):
    """Simple health check endpoint."""
    return web.json_response({"status": "ok", "service": "media-proxy"})


async def create_app():
    """Create and configure the unified HTTP/WebSocket application."""
    app = web.Application()

    # WebSocket endpoint (existing functionality)
    app.router.add_get('/control', websocket_handler)

    # HTTP API endpoints (new functionality)
    app.router.add_post('/api/convert/animimg', handle_animimg_request)
    app.router.add_get('/api/system/health', health_check_handler)

    # Internal protocol endpoints
    app.router.add_get('/api/internal/placeholder/{spec:.*}', handle_placeholder_request)

    return app


async def start_unified_server(host: str = "0.0.0.0", port: int = 8788):
    """Start the unified HTTP/WebSocket server."""
    app = await create_app()

    # Create HTTP server
    runner = web.AppRunner(app)
    await runner.setup()

    site = web.TCPSite(runner, host, port)
    await site.start()

    logging.getLogger('server').info(f"Server on http://{host}:{port}/ (WebSocket: /control, Convert API: /api/convert/)")

    return runner