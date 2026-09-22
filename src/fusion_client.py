import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class FusionSUClient:
    """Non-blocking client that streams SU frames from a background event loop."""

    def __init__(
        self,
        server_host: str,
        server_port: int,
        su_id: str,
        queue_size: int = 16,
    ):
        self.su_id = su_id
        self.url = f"http://{server_host}:{int(server_port)}/datasets/{su_id}"
        self.queue_size = queue_size
        self._loop = asyncio.new_event_loop()
        self._queue: asyncio.Queue[dict[str, Any]] | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"fusion-client-{su_id}",
            daemon=True,
        )
        self._frame_id = 0
        self._started = False
        self._last_error = None

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread.start()
        self._ready.wait()

    def send_iq_frames(
        self,
        samples,
        occupied_channels: dict[int, bool],
        snr_db: dict[int, float],
        channel_count: int,
        frame_size: int = 1024,
    ) -> int:
        """Queue complete IQ frames without waiting for the fusion server."""
        if not self._started:
            raise RuntimeError("FusionClient must be started before sending")
        if len(samples) < frame_size:
            return 0

        occupancy = [
            bool(occupied_channels.get(channel_index, False))
            for channel_index in range(channel_count)
        ]
        snr_values = [
            _json_number_or_null(snr_db.get(channel_index, float("-inf")))
            for channel_index in range(channel_count)
        ]
        timestamp = datetime.now(timezone.utc).isoformat()
        frames = []
        for offset in range(0, len(samples) - frame_size + 1, frame_size):
            frame = samples[offset : offset + frame_size]
            frames.append(
                {
                    "timestamp": timestamp,
                    "frame_id": self._frame_id,
                    "iq": [[float(sample.real), float(sample.imag)] for sample in frame],
                    "channels": occupancy,
                    "snr_db": snr_values,
                }
            )
            self._frame_id += 1

        payload = {"su_id": self.su_id, "frames": frames}
        self._loop.call_soon_threadsafe(self._enqueue, payload)
        return len(frames)

    def close(self):
        if not self._started:
            return
        stop_event = self._stop_event
        if stop_event is None:
            raise RuntimeError("FusionClient event loop was not initialized")
        self._loop.call_soon_threadsafe(stop_event.set)
        self._thread.join()
        self._started = False

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._queue = asyncio.Queue(maxsize=self.queue_size)
        self._stop_event = asyncio.Event()
        self._ready.set()
        self._loop.run_until_complete(self._send_until_stopped())
        self._loop.close()

    def _enqueue(self, payload: dict[str, Any]):
        queue = self._queue
        if queue is None:
            return
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            # Dropping a stale frame keeps SDR reception independent of a slow server.
            pass

    async def _send_until_stopped(self):
        queue = self._queue
        stop_event = self._stop_event
        if queue is None or stop_event is None:
            raise RuntimeError("FusionClient event loop was not initialized")
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            sender = asyncio.create_task(self._sender(session))
            await stop_event.wait()
            await queue.join()
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    async def _sender(self, session: aiohttp.ClientSession):
        queue = self._queue
        if queue is None:
            raise RuntimeError("FusionClient event loop was not initialized")
        while True:
            payload = await queue.get()
            try:
                async with session.post(self.url, json=payload) as response:
                    if response.status >= 400:
                        error_text = await response.text()
                        logger.error(
                            "Fusion server rejected SU %s payload (%s): %s",
                            self.su_id,
                            response.status,
                            error_text,
                        )
                    else:
                        logger.debug(
                            "Fusion server accepted %d frames for SU %s",
                            len(payload["frames"]),
                            self.su_id,
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                logger.error(
                    "Unable to send SU %s data to %s: %s",
                    self.su_id,
                    self.url,
                    error,
                )
            finally:
                queue.task_done()


def _json_number_or_null(value: float):
    return value if value != float("-inf") else None


class FusionPUClient:
    """Client that sends accumulated PU events when the simulation ends."""

    def __init__(self, server_host: str, server_port: int, pu_id: str):
        self.pu_id = pu_id
        self.url = f"http://{server_host}:{int(server_port)}/primary-users/{pu_id}"

    def send_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        asyncio.run(self._send_events(events))

    async def _send_events(self, events: list[dict[str, Any]]) -> None:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.url,
                json={"pu_id": self.pu_id, "events": events},
            ) as response:
                if response.status >= 400:
                    message = await response.text()
                    raise RuntimeError(
                        "Fusion server rejected PU data "
                        f"({response.status}): {message}"
                    )
                logger.info("Sent %d events for PU %s", len(events), self.pu_id)


def send_primary_data(
    server_host: str,
    server_port: int,
    pu_id: str,
    events: list[dict[str, Any]],
) -> None:
    """Compatibility helper for sending accumulated PU events."""
    FusionPUClient(server_host, server_port, pu_id).send_events(events)


FusionClient = FusionSUClient
