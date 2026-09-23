import adi
import numpy as np
import os
from pathlib import Path
import random
import sys
import time
import yaml
from utils import configureSDR
from channel_creator import ChannelPlan, get_channel_group
from config import SpectrumConfig
from fusion_client import FusionPUClient

YAML_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
MIN_OCCUPATION_DURATION_S = 10.0
MAX_OCCUPATION_DURATION_S = 60.0

def generate_channel_signal(
        sample_rate_hz: float,
        frequency_offsets_hz: list[float],
        num_samples: int = 65536,
        amplitude: float = 0.5,
    ) -> np.ndarray:

        t = np.arange(num_samples) / sample_rate_hz
        signal = np.zeros(num_samples, dtype=np.complex64)

        for frequencyOffsetHz in frequency_offsets_hz:
            signal += np.exp(
                1j * 2 * np.pi * frequencyOffsetHz * t
            )

        return (
            amplitude
            * (2 ** 14)
            * signal
            / len(frequency_offsets_hz)
        ).astype(
            np.complex64
        )

def transmit(sdr, samples: np.ndarray):
    sdr.tx(samples)

def createTransmitterSDR(puConfig: dict):
    sdr = adi.Pluto(
         puConfig["uri"]
    )

    # Configure SDR for TX mode
    configureSDR(
        sdr,
        center_frequency_hz=int(
            puConfig["center_frequency_hz"]
        ),
        sample_rate_hz=int(
            puConfig["sample_rate_hz"]
        ),
        bandwidth_hz=int(
            puConfig["bandwidth_hz"]
        ),
        is_tx=True,
    )
    sdr.tx_hardwaregain_chan0 = int(puConfig["tx_hardwaregain_chan0"])

    return sdr



def occupy_channel(
        sdr,
        channel_plan: ChannelPlan,
        channel_indices: list[int],
    duration_s: float = 10.0,
    tx_hardwaregain_chan0: int = 0,
    ):

        cfg = channel_plan.config
        channels, groupCenterHz, occupiedBandwidthHz = get_channel_group(
            channel_plan, channel_indices, require_contiguous=False
        )

        frequencyOffsetsHz = [
            channel.center_frequency_hz - groupCenterHz
            for channel in channels
        ]
        txSampleRateHz = max(
            float(cfg.sample_rate_hz),
            float(occupiedBandwidthHz),
        )

        configureSDR(
            sdr,
            center_frequency_hz=(
                groupCenterHz
            ),

            sample_rate_hz=(
                txSampleRateHz
            ),

            bandwidth_hz=(
                occupiedBandwidthHz
            ),
            is_tx=True,
        )
        sdr.tx_hardwaregain_chan0 = int(tx_hardwaregain_chan0)

        samples = (
            generate_channel_signal(
                sample_rate_hz=(
                    txSampleRateHz
                ),

                frequency_offsets_hz=(
                    frequencyOffsetsHz
                ),
            )
        )

        sdr.tx_cyclic_buffer = True

        try:
            print(
                f"[LOG] PU transmitting on Channels {channel_indices} "
                f"({groupCenterHz / 1e6:.3f} MHz group center) "
                f"for {duration_s:.1f} seconds..."
            )
            transmit(sdr, samples)
            time.sleep(duration_s)
        finally:
            sdr.tx_destroy_buffer()
            print("[LOG] PU transmission stopped.")


def parsePUConfig(yamlPath: str | Path, uri: str) -> dict:
    """
    Load Primary User configuration from a YAML file.
    """
    with open(yamlPath, "r") as file:
        config = yaml.safe_load(file)

    requiredFields = [
        "pu_id",
        "sample_rate_hz",
        "bandwidth_hz",
        "center_frequency_hz",
        "buffer_size",
        "tx_hardwaregain_chan0",
    ]

    pu_config = config.get("Devices", {}).get("Primary_User", {}).get(uri, {})
    
    for field in requiredFields:
        if field not in pu_config:
            raise ValueError(
                f"Missing required configuration field: {field}"
            )

    configured_gains = {
        device_config.get("tx_hardwaregain_chan0")
        for device_config in config.get("Devices", {}).get("Primary_User", {}).values()
    }
    if len(configured_gains) > 1:
        raise ValueError("All Primary_User entries must use the same TX hardware gain")

    return pu_config


def parseFusionServerConfig(yamlPath: str | Path) -> dict:
    with open(yamlPath, "r") as file:
        config = yaml.safe_load(file)
    fusion_config = config.get("FusionServer", {})
    for field in ("host", "port"):
        if field not in fusion_config:
            raise ValueError(f"Missing required FusionServer field: {field}")
    return fusion_config


def choose_random_occupation(channel_count: int) -> tuple[list[int], float]:
    if channel_count < 1:
        raise ValueError("At least one channel must be available to a PU")

    selected_count = random.randint(0, min(4, channel_count))
    selected_indices = sorted(random.sample(range(channel_count), selected_count))
    duration_s = random.uniform(
        MIN_OCCUPATION_DURATION_S,
        MAX_OCCUPATION_DURATION_S,
    )
    return selected_indices, duration_s


def get_device_uri() -> str:
    return sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PLUTO_URI", "")


def parseChannelPlan(yamlPath: str | Path) -> dict:
    """
    Load ChannelPlan configuration from a YAML file.
    """
    with open(yamlPath, "r") as file:
        config = yaml.safe_load(file)

    requiredFields = [
        "center_frequency_hz",
        "total_bandwidth_hz",
        "channel_bandwidth_hz",
        "channel_spacing_hz",
        "sample_rate_hz",
    ]

    channel_plan_config = config.get("ChannelPlan", {})
    
    for field in requiredFields:
        if field not in channel_plan_config:
            raise ValueError(
                f"Missing required ChannelPlan field: {field}"
            )

    return channel_plan_config


def createChannelPlan(channelPlanConfig: dict) -> ChannelPlan:
    """
    Create a channel plan from ChannelPlan configuration.
    """
    spectrum_config = SpectrumConfig(
        center_frequency_hz=float(channelPlanConfig["center_frequency_hz"]),
        total_bandwidth_hz=float(channelPlanConfig["total_bandwidth_hz"]),
        channel_bandwidth_hz=float(channelPlanConfig["channel_bandwidth_hz"]),
        channel_spacing_hz=float(channelPlanConfig["channel_spacing_hz"]),
        sample_rate_hz=float(channelPlanConfig["sample_rate_hz"]),
    )
    
    return ChannelPlan(spectrum_config)


def main():
    """
    Primary User transmitter main execution function.
    """
    print("[LOG] Primary User Transmitter Starting...")

    uri = get_device_uri()
    if not uri:
        raise ValueError("Provide the Pluto URI as the first argument or PLUTO_URI")
    
    print("[LOG] Parsing Primary User configuration...")
    puConfig = parsePUConfig(
        YAML_PATH, uri
    )
    print("[LOG] Configuration parsed successfully.")
    fusionConfig = parseFusionServerConfig(YAML_PATH)

    print(f"[LOG] Device URI retrieved: {uri}")
    puConfig["uri"] = uri

    print("[LOG] Creating and configuring transmitter SDR...")
    sdr = createTransmitterSDR(
        puConfig
    )
    print("[LOG] Transmitter SDR created successfully.")

    sampleRateHz = float(
        puConfig["sample_rate_hz"]
    )

    centerFrequencyHz = float(
        puConfig["center_frequency_hz"]
    )

    print(
        "\n" + "-"*60
    )
    print(
        "Primary User Transmitter Configuration"
    )
    print("-"*60)

    print(
        f"URI: {uri}"
    )

    print(
        "Center Frequency: "
        f"{centerFrequencyHz / 1e6:.3f} MHz"
    )

    print(
        "Sample Rate: "
        f"{sampleRateHz / 1e6:.3f} MHz"
    )

    print(
        f"Buffer Size: "
        f"{puConfig['buffer_size']}"
    )
    print("-"*60 + "\n")

    try:
        print("[LOG] Parsing ChannelPlan configuration...")
        channelPlanConfig = parseChannelPlan(YAML_PATH)
        print("[LOG] ChannelPlan configuration parsed successfully.")
        
        print("[LOG] Creating channel plan...")
        channel_plan = createChannelPlan(channelPlanConfig)
        
        print(
            f"[LOG] Channel Plan Created with "
            f"{len(channel_plan.channels)} channels"
        )
        print(
            f"[LOG] Channel bandwidth: {channel_plan.config.channel_bandwidth_hz/1e3:.1f} kHz"
        )
        print(
            f"[LOG] Total spectrum bandwidth: {channel_plan.config.total_bandwidth_hz/1e6:.1f} MHz"
        )

        print("[LOG] Starting randomized channel occupation loop...")
        occupation_events = []
        while True:
            channelIndices, transmissionDurationS = choose_random_occupation(
                len(channel_plan.channels)
            )
            channels = [
                channel_plan.get_channel(channelIndex)
                for channelIndex in channelIndices
            ]
            if channelIndices:
                print(f"\n[LOG] Occupying channels {channelIndices}...")
            else:
                print(
                    f"\n[LOG] PU idle for {transmissionDurationS:.1f} seconds..."
                )
            for channel in channels:
                print(
                    f"[LOG] Channel {channel.index} center: "
                    f"{channel.center_frequency_hz / 1e6:.3f} MHz"
                )
            if channelIndices:
                print(
                    f"[LOG] Generating tone for a "
                    f"{transmissionDurationS:.1f} second transmission..."
                )
            channel_occupation = [
                channel.index in channelIndices
                for channel in channel_plan.channels
            ]
            occupation_events.append(
                {
                    "timestamp": time.strftime(
                        "%Y-%m-%dT%H:%M:%S%z", time.gmtime()
                    ),
                    "channel_occupation": channel_occupation,
                    "channel_indices": channelIndices,
                    "duration_s": transmissionDurationS,
                }
            )
            if channelIndices:
                occupy_channel(
                    sdr,
                    channel_plan,
                    channel_indices=channelIndices,
                    duration_s=transmissionDurationS,
                    tx_hardwaregain_chan0=int(puConfig["tx_hardwaregain_chan0"]),
                )
            else:
                time.sleep(transmissionDurationS)

    except Exception as e:
        print(
            f"\n[ERROR] Error during transmission: {e}"
        )
        import traceback
        traceback.print_exc()

    finally:
        if "occupation_events" in locals():
            try:
                FusionPUClient(
                    fusionConfig["host"],
                    fusionConfig["port"],
                    puConfig["pu_id"],
                ).send_events(occupation_events)
            except Exception as upload_error:
                print(f"[ERROR] Failed to send PU data: {upload_error}")
        print(
            "[LOG] Primary User Transmitter Shutdown Complete"
        )


if __name__ == "__main__":
    main()
