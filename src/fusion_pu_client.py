import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


async def _send_primary_data(
    server_host: str,
    server_port: int,
    pu_id: str,
    events: list[dict[str, Any]],
):
    if not events:
        return

    url = f"http://{server_host}:{int(server_port)}/primary-users/{pu_id}"
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url,
            json={"pu_id": pu_id, "events": events},
        ) as response:
            if response.status >= 400:
                message = await response.text()
                raise RuntimeError(
                    f"Fusion server rejected PU data ({response.status}): {message}"
                )
            logger.info("Sent %d events for PU %s", len(events), pu_id)


def send_primary_data(
    server_host: str,
    server_port: int,
    pu_id: str,
    events: list[dict[str, Any]],
):
    """Send accumulated PU events synchronously when the simulation ends."""
    asyncio.run(_send_primary_data(server_host, server_port, pu_id, events))