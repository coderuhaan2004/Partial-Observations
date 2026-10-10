import argparse
import asyncio
import json
from pathlib import Path

from aiohttp import web
import yaml


class FusionDatasetStore:
    def __init__(self, storage_path: str | Path):
        self.storage_path = Path(storage_path)
        self.writers: dict[str, asyncio.Lock] = {}

    async def append(self, su_id: str, frames: list[dict]):
        lock = self.writers.setdefault(su_id, asyncio.Lock())
        async with lock:
            output_path = self.storage_path / f"{su_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing_frames = []
            if output_path.exists() and output_path.stat().st_size:
                with output_path.open() as file:
                    existing_frames = json.load(file)
            if not isinstance(existing_frames, list):
                raise ValueError(f"Dataset for {su_id} must contain a JSON list")
            existing_frames.extend(frames)
            with output_path.open("w") as file:
                json.dump(existing_frames, file)

    async def append_primary_events(self, pu_id: str, events: list[dict]):
        lock = self.writers.setdefault(f"pu:{pu_id}", asyncio.Lock())
        async with lock:
            output_path = self.storage_path / f"{pu_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing_events = []
            if output_path.exists() and output_path.stat().st_size:
                with output_path.open() as file:
                    existing_events = json.load(file)
            if not isinstance(existing_events, list):
                raise ValueError(f"Dataset for {pu_id} must contain a JSON list")
            existing_events.extend(events)
            with output_path.open("w") as file:
                json.dump(existing_events, file)


async def receive_dataset(request: web.Request):
    su_id = request.match_info["su_id"]
    payload = await request.json()
    if payload.get("su_id") != su_id:
        raise web.HTTPBadRequest(text="Path su_id must match payload su_id")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise web.HTTPBadRequest(text="frames must be a list")
    try:
        await request.app["store"].append(su_id, frames)
    except (json.JSONDecodeError, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    return web.json_response({"accepted": len(frames), "su_id": su_id})


async def health_check(request: web.Request):
    return web.json_response({"status": "ok"})


async def receive_primary_data(request: web.Request):
    pu_id = request.match_info["pu_id"]
    payload = await request.json()
    if payload.get("pu_id") != pu_id:
        raise web.HTTPBadRequest(text="Path pu_id must match payload pu_id")
    events = payload.get("events")
    if not isinstance(events, list):
        raise web.HTTPBadRequest(text="events must be a list")
    try:
        await request.app["store"].append_primary_events(pu_id, events)
    except (json.JSONDecodeError, ValueError) as error:
        raise web.HTTPBadRequest(text=str(error)) from error
    return web.json_response({"accepted": len(events), "pu_id": pu_id})


def create_app(storage_path: str | Path) -> web.Application:
    app = web.Application()
    app["store"] = FusionDatasetStore(storage_path)
    app.router.add_get("/health", health_check)
    app.router.add_post("/datasets/{su_id}", receive_dataset)
    app.router.add_post("/primary-users/{pu_id}", receive_primary_data)
    return app


def load_server_config(config_path: str | Path) -> dict:
    config_path = Path(config_path).resolve()
    with config_path.open() as file:
        config = yaml.safe_load(file) or {}
    fusion_config = config.get("FusionServer", {})
    for field in ("host", "port", "storage_path"):
        if field not in fusion_config:
            raise ValueError(f"Missing required FusionServer field: {field}")
    fusion_config = dict(fusion_config)
    storage_path = Path(fusion_config["storage_path"])
    if not storage_path.is_absolute():
        fusion_config["storage_path"] = str(config_path.parent / storage_path)
    return fusion_config


def main():
    default_config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    parser = argparse.ArgumentParser(description="Fusion server for SU datasets")
    parser.add_argument("--config", default=str(default_config_path))
    parser.add_argument(
        "--bind-host",
        "--host",
        dest="bind_host",
        default="0.0.0.0",
        help="Interface inside the host/container to listen on",
    )
    parser.add_argument("--port", type=int)
    parser.add_argument("--storage-path")
    args = parser.parse_args()
    fusion_config = load_server_config(args.config)
    port = args.port or int(fusion_config["port"])
    storage_path = args.storage_path or fusion_config["storage_path"]
    web.run_app(create_app(storage_path), host=args.bind_host, port=port)


if __name__ == "__main__":
    main()
