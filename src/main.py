# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import argparse
import asyncio
import logging
import os

from .api.server import start_unified_server

# Always use relative imports (run via run.py or python -m src.main)
from .config import Config
from .utils.hardware import set_windows_timer_resolution


def setup_logging(config, override_level=None):
    """Configure logging based on config settings."""
    # Determine log level from override, config, or default
    log_level_str = override_level.lower() if override_level else config.get("log.level").lower()

    # Map string levels to logging constants
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "warn": logging.WARNING,  # alias
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    log_level = level_map.get(log_level_str, logging.INFO)

    # Set root logger level with proper format including logger name and level
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] [%(name)s] %(message)s",  # Level first, then logger name
        handlers=[logging.StreamHandler()],
        force=True,  # Reset any existing configuration
    )


async def main():
    """Main entry point for the media proxy server."""
    parser = argparse.ArgumentParser(description="Media Proxy Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8788, help="Port to bind to (WebSocket + HTTP API)")
    parser.add_argument("--config", default=None, help="Path to YAML/TOML/JSON config file")
    parser.add_argument(
        "--log-level",
        default=None,
        choices=["debug", "info", "warning", "warn", "error", "critical"],
        type=str.lower,
        help="Set logging level (overrides config file)",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config()
    config.load(args.config)

    # Setup logging
    setup_logging(config, override_level=args.log_level)
    logger = logging.getLogger("main")
    logger.info(f"loaded config: {config.get()}")

    # Setup high-performance event loop (uvloop/winloop)
    if os.name == "nt":
        # Windows: use winloop
        try:
            import winloop  # type: ignore[import-not-found]

            asyncio.set_event_loop_policy(winloop.EventLoopPolicy())
            logger.info("winloop enabled")
        except ImportError:
            logger.info("winloop not available, using default asyncio loop")
    else:
        # Non-Windows: use uvloop
        try:
            import uvloop  # type: ignore[import-not-found]

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
            logger.info("uvloop enabled")
        except ImportError:
            logger.info("uvloop not available, using default asyncio loop")

    try:
        # Set Windows timer resolution for better timing accuracy
        if config.get("net.win_timer_res"):
            set_windows_timer_resolution(True)

        # Start unified server (WebSocket + HTTP API)
        await start_unified_server(args.host, args.port)

        # Keep running until interrupted
        await asyncio.Future()  # run forever

    finally:
        # Clean up Windows timer resolution
        if config.get("net.win_timer_res"):
            set_windows_timer_resolution(False)


def run():
    """Entry point for setuptools console scripts."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.getLogger("main").info("Shutting down...")


if __name__ == "__main__":
    run()
