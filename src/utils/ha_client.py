# © Copyright 2025 Stuart Parmenter
# SPDX-License-Identifier: MIT

import logging
import os
from typing import Any

import aiohttp


class HomeAssistantClient:
    """Client for communicating with Home Assistant from within an addon.

    Creates its own ClientSession for each request to avoid deadlocks
    when called from within aiohttp server handlers.
    """

    def __init__(self):
        """Initialize HA client.

        Supports two modes:
        1. Addon mode: Uses SUPERVISOR_TOKEN and http://supervisor/core/api
        2. Development mode: Uses HA_TOKEN and HA_URL (for external testing)
        """
        self.logger = logging.getLogger("ha_client")

        # Check for addon mode first (SUPERVISOR_TOKEN exists)
        self.token = os.getenv("SUPERVISOR_TOKEN")

        if self.token:
            # Addon mode: Use supervisor proxy
            self.base_url = "http://supervisor/core/api"
            self.logger.info("HA client in addon mode (supervisor proxy)")
        else:
            # Development mode: Use external HA instance
            self.token = os.getenv("HA_TOKEN")
            self.base_url = os.getenv("HA_URL", "http://homeassistant.local:8123/api")

            if self.token:
                self.logger.info(f"HA client in development mode ({self.base_url})")
            else:
                self.logger.warning(
                    "No HA credentials found - set SUPERVISOR_TOKEN (addon) or HA_TOKEN+HA_URL (development)"
                )

    def _get_headers(self) -> dict[str, str]:
        """Get authorization headers for API requests."""
        if not self.token:
            raise RuntimeError("SUPERVISOR_TOKEN not available - not running in Home Assistant addon?")

        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    async def render_template(self, template: str) -> str:
        """Render a Home Assistant template string.

        Args:
            template: Jinja2 template string (e.g., "{{ states('sensor.temp') }}")

        Returns:
            Rendered template as string

        Raises:
            RuntimeError: If SUPERVISOR_TOKEN not available
            aiohttp.ClientError: If API request fails
        """
        url = f"{self.base_url}/template"
        headers = self._get_headers()

        async with (
            aiohttp.ClientSession() as session,
            session.post(url, json={"template": template}, headers=headers) as resp,
        ):
            if resp.status != 200:
                error_text = await resp.text()
                self.logger.error(f"Template render failed ({resp.status}): {error_text}")
                raise aiohttp.ClientError(f"Template render failed ({resp.status}): {error_text}")

            result = await resp.text()
            self.logger.debug(f"Rendered template: {template[:50]}... -> {result[:100]}")
            return result

    async def get_entity_state(self, entity_id: str) -> dict[str, Any]:
        """Get the state of a specific entity.

        Args:
            entity_id: Entity ID (e.g., "sensor.temperature")

        Returns:
            State object containing entity_id, state, attributes, etc.

        Raises:
            RuntimeError: If SUPERVISOR_TOKEN not available
            aiohttp.ClientError: If API request fails or entity not found
        """
        url = f"{self.base_url}/states/{entity_id}"
        headers = self._get_headers()

        async with (
            aiohttp.ClientSession() as session,
            session.get(url, headers=headers) as resp,
        ):
            if resp.status == 404:
                raise aiohttp.ClientError(f"Entity not found: {entity_id}")

            if resp.status != 200:
                error_text = await resp.text()
                self.logger.error(f"Get entity state failed ({resp.status}): {error_text}")
                raise aiohttp.ClientError(f"Get entity state failed: {error_text}")

            result = await resp.json()
            self.logger.debug(f"Got entity state: {entity_id} -> {result.get('state')}")
            return result
