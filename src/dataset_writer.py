import csv
import json
from datetime import datetime, timezone
from pathlib import Path


class CsvDatasetWriter:
    FRAME_SIZE = 1024

    def __init__(
        self,
        output_path: str | Path,
        channel_count: int,
        frame_size: int = FRAME_SIZE,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output_path.open("a", newline="")
        self.writer = csv.writer(self.file)
        self.channel_count = channel_count
        self.frame_size = frame_size
        self.frame_id = 0
        self.header = (
            ["timestamp", "frame_id"]
            + [
                component
                for sample_index in range(self.frame_size)
                for component in (f"I_{sample_index}", f"Q_{sample_index}")
            ]
            + [f"channel {index}" for index in range(1, channel_count + 1)]
            + [f"snr channel {index}" for index in range(1, channel_count + 1)]
        )

        if self.file.tell() == 0:
            self.writer.writerow(self.header)
            self.file.flush()
        else:
            self.file.seek(0)
            existing_header = next(csv.reader(self.file), [])
            self.file.seek(0, 2)
            if existing_header != self.header:
                raise ValueError(
                    "Dataset already exists with a different schema; "
                    "use a new output path for frame-based data"
                )

    def write_iq_frames(
        self,
        samples,
        occupied_channels: dict[int, bool],
        snr_db: dict[int, float] | None = None,
    ):
        if len(samples) < self.frame_size:
            return 0

        occupancy = [
            bool(occupied_channels.get(channel_index, False))
            for channel_index in range(self.channel_count)
        ]
        snr_values = [
            float((snr_db or {}).get(channel_index, float("-inf")))
            for channel_index in range(self.channel_count)
        ]
        rows = []
        frame_count = len(samples) // self.frame_size
        for frame_index in range(frame_count):
            frame = samples[
                frame_index * self.frame_size : (frame_index + 1) * self.frame_size
            ]
            iq_values = [
                value
                for sample in frame
                for value in (float(sample.real), float(sample.imag))
            ]
            rows.append(
                [
                    datetime.now(timezone.utc).isoformat(),
                    self.frame_id,
                    *iq_values,
                    *occupancy,
                    *snr_values,
                ]
            )
            self.frame_id += 1
        self.writer.writerows(rows)
        self.file.flush()
        return frame_count

    def write_iq_buffer(
        self,
        samples,
        occupied_channels: dict[int, bool],
        sample_rate_hz: float,
        snr_db: dict[int, float] | None = None,
    ):
        """Write complete 1024-sample frames; ignore an incomplete tail."""
        return self.write_iq_frames(samples, occupied_channels, snr_db)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class JsonDatasetWriter:
    FRAME_SIZE = 1024

    def __init__(
        self,
        output_path: str | Path,
        channel_count: int,
        frame_size: int = FRAME_SIZE,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.channel_count = channel_count
        self.frame_size = frame_size
        self.frame_id = 0
        self.frames = []

        if self.output_path.exists() and self.output_path.stat().st_size > 0:
            with self.output_path.open() as file:
                self.frames = json.load(file)
            if not isinstance(self.frames, list):
                raise ValueError("JSON dataset must contain a list of frames")
            self.frame_id = len(self.frames)

    def write_iq_frames(
        self,
        samples,
        occupied_channels: dict[int, bool],
        snr_db: dict[int, float] | None = None,
    ):
        if len(samples) < self.frame_size:
            return 0

        occupancy = [
            bool(occupied_channels.get(channel_index, False))
            for channel_index in range(self.channel_count)
        ]
        frame_count = len(samples) // self.frame_size
        timestamp = datetime.now(timezone.utc).isoformat()

        for frame_index in range(frame_count):
            frame = samples[
                frame_index * self.frame_size : (frame_index + 1) * self.frame_size
            ]
            self.frames.append(
                {
                    "timestamp": timestamp,
                    "frame_id": self.frame_id,
                    "iq": [
                        [float(sample.real), float(sample.imag)]
                        for sample in frame
                    ],
                    "channels": occupancy,
                    "snr_db": [
                        _json_number_or_null(
                            (snr_db or {}).get(channel_index, float("-inf"))
                        )
                        for channel_index in range(self.channel_count)
                    ],
                }
            )
            self.frame_id += 1

        with self.output_path.open("w") as file:
            json.dump(self.frames, file)
        return frame_count

    def write_iq_buffer(
        self,
        samples,
        occupied_channels: dict[int, bool],
        sample_rate_hz: float,
        snr_db: dict[int, float] | None = None,
    ):
        return self.write_iq_frames(samples, occupied_channels, snr_db)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def create_dataset_writer(
    output_path: str | Path,
    channel_count: int,
    dataset_format: str = "csv",
):
    writers = {
        "csv": CsvDatasetWriter,
        "json": JsonDatasetWriter,
    }
    writer_class = writers.get(dataset_format.lower())
    if writer_class is None:
        raise ValueError(
            f"Unsupported dataset format {dataset_format!r}; choose csv or json"
        )
    return writer_class(output_path, channel_count)


def _json_number_or_null(value: float):
    return value if value != float("-inf") else None