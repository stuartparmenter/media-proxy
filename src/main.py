# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import argparse
import asyncio
import os
import sys

# Handle both direct execution and module execution
if __name__ == "__main__" and __package__ is None:
    # Direct execution: python main.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import Config
    from control.websocket import start_websocket_server
    from utils.hardware import set_windows_timer_resolution
else:
    # Module execution: python -m src.main
    from .config import Config
    from .control.websocket import start_websocket_server
    from .utils.hardware import set_windows_timer_resolution


async def main():
    """Main entry point for the media proxy server."""
    parser = argparse.ArgumentParser(description="Media Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8788, help="Port to bind to")
    parser.add_argument("--config", default=None, 
                       help="Path to YAML/TOML/JSON config (default: ws_ddp_proxy.yaml if present)")
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.load(args.config)
    print(f"* loaded config: {config.get()}")

    try:
        # Set Windows timer resolution for better timing accuracy
        if config.get("net.win_timer_res", True):
            set_windows_timer_resolution(True)

        # Start WebSocket server
        server = await start_websocket_server(args.host, args.port)
        
        # Keep running until interrupted
        await asyncio.Future()  # run forever
        
    finally:
        # Clean up Windows timer resolution
        if config.get("net.win_timer_res", True):
            set_windows_timer_resolution(False)


def run():
    """Entry point for setuptools console scripts."""
    try:
        # Use uvloop on non-Windows systems for better performance
        if os.name != "nt":
            try:
                import uvloop
                asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                print("[perf] uvloop enabled")
            except ImportError:
                print("[perf] uvloop not available, using default asyncio loop")
        else:
            print("[perf] using default asyncio loop on Windows")

        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n* Shutting down...")


if __name__ == "__main__":
    run()