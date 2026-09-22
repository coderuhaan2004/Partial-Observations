import adi
import numpy as np
import matplotlib
import os
from pathlib import Path
import sys
import yaml

matplotlib.use(
    "TkAgg"
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    else "Agg"
)
import matplotlib.pyplot as plt
from utils import computePowerSpectrum, configureSDR
from channel_creator import ChannelPlan
from config import SpectrumConfig
from fusion_client import FusionClient

YAML_PATH = Path(__file__).resolve().parent.parent / "config.yaml"

def receiveSamples(sdr) -> np.ndarray:
    """
    Receive one buffer of complex IQ samples from PlutoSDR.
    """
    return sdr.rx()


def parseSUConfig(yamlPath: str | Path, uri: str) -> dict:
    """
    Load Secondary User configuration from a YAML file.
    """

    with open(yamlPath, "r") as file:
        config = yaml.safe_load(file)

    requiredFields = [
        "sample_rate_hz",
        "bandwidth_hz",
        "center_frequency_hz",
        "buffer_size",
        "rx_hardwaregain_chan0",
        "detection_threshold_db",
        "su_id",
    ]

    su_config = config.get("Devices", {}).get("Secondary_User", {}).get(uri, {})
    
    for field in requiredFields:
        if field not in su_config:
            raise ValueError(
                f"Missing required configuration field: {field}"
            )

    return su_config


def parseChannelPlan(yamlPath: str | Path) -> ChannelPlan:
    with open(yamlPath, "r") as file:
        config = yaml.safe_load(file)

    return ChannelPlan(SpectrumConfig(**config["ChannelPlan"]))


def parseFusionServerConfig(yamlPath: str | Path) -> dict:
    with open(yamlPath, "r") as file:
        config = yaml.safe_load(file)

    fusion_config = config.get("FusionServer", {})
    for field in ("host", "port"):
        if field not in fusion_config:
            raise ValueError(f"Missing required FusionServer field: {field}")
    return fusion_config


def get_device_uri() -> str:
    return sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PLUTO_URI", "")


def createRecieverSDR(suConfig: dict):
    """
    Create the PlutoSDR receiver.
    """

    sdr = adi.Pluto(
        suConfig["uri"]
    )

    sdr.rx_buffer_size = int(
        suConfig["buffer_size"]
    )
    sdr.rx_hardwaregain_chan0 = int(suConfig["rx_hardwaregain_chan0"])

    # Configure SDR for RX mode
    configureSDR(
        sdr,
        center_frequency_hz=int(
            suConfig["center_frequency_hz"]
        ),
        sample_rate_hz=int(
            suConfig["sample_rate_hz"]
        ),
        bandwidth_hz=int(
            suConfig["bandwidth_hz"]
        ),
        is_tx=False,
    )

    return sdr


def classify_channels(
    frequencies_hz: np.ndarray,
    power_db: np.ndarray,
    channels,
    threshold_db: float,
) -> list[tuple[int, float, bool, float]]:
    channel_powers = []
    linear_power = 10 ** (power_db / 10)
    for channel in channels:
        channel_mask = (
            (frequencies_hz >= channel.lower_frequency_hz)
            & (frequencies_hz < channel.upper_frequency_hz)
        )
        if not np.any(channel_mask):
            raise ValueError(f"No FFT bins found for channel {channel.index}")
        channel_power = float(np.mean(linear_power[channel_mask]))
        channel_powers.append((channel.index, channel_power))

    channel_power_db = [
        (channel_index, float(10 * np.log10(channel_power)), channel_power)
        for channel_index, channel_power in channel_powers
    ]
    available_powers = [
        channel_power
        for _, channel_db, channel_power in channel_power_db
        if channel_db <= threshold_db
    ]
    noise_power = float(
        np.mean(available_powers)
        if available_powers
        else np.percentile([power for _, _, power in channel_power_db], 25)
    )
    results = []
    for channel_index, channel_db, channel_power in channel_power_db:
        occupied = channel_db > threshold_db
        signal_power = max(channel_power - noise_power, 0.0) if occupied else 0.0
        snr_db = (
            float(10 * np.log10(signal_power / noise_power))
            if signal_power > 0 and noise_power > 0
            else float("-inf")
        )
        results.append(
            (
                channel_index,
                channel_db,
                occupied,
                snr_db,
            )
        )
    return results


def print_channel_detections(
    detections: list[tuple[int, float, bool, float]],
    threshold_db: float,
):
    for channel_index, power_db, occupied, snr_db in detections:
        state = "occupied" if occupied else "available"
        print(
            f"[DETECTION] Channel {channel_index}: "
            f"{power_db:.2f} dB, SNR {snr_db:.2f} dB "
            f"({state}, threshold {threshold_db:.2f} dB)"
        )


def main():

    print("[LOG] Secondary User Receiver Starting...")

    uri = get_device_uri()
    if not uri:
        raise ValueError("Provide the Pluto URI as the first argument or PLUTO_URI")
    
    print("[LOG] Parsing Secondary User configuration...")
    suConfig = parseSUConfig(
        YAML_PATH, uri
    )
    fusionConfig = parseFusionServerConfig(YAML_PATH)
    print("[LOG] Configuration parsed successfully.")

    channel_plan = parseChannelPlan(YAML_PATH)
    channel_indices = list(range(len(channel_plan.channels)))
    sensingCenterHz = channel_plan.config.center_frequency_hz
    sensingBandwidthHz = channel_plan.config.total_bandwidth_hz
    configuredBandwidthHz = float(suConfig["bandwidth_hz"])
    if not np.isclose(sensingBandwidthHz, configuredBandwidthHz):
        raise ValueError(
            "Secondary bandwidth_hz must equal the bandwidth of channel_indices "
            f"({sensingBandwidthHz:g} Hz)"
        )
    suConfig["center_frequency_hz"] = sensingCenterHz
    suConfig["bandwidth_hz"] = sensingBandwidthHz
    suConfig["uri"] = uri

    if suConfig["uri"]:
        print(f"[LOG] Device URI retrieved: {suConfig['uri']}")
    else:
        print("[ERROR] Failed to retrieve device URI")
        return

    print("[LOG] Creating and configuring receiver SDR...")
    sdr = createRecieverSDR(
        suConfig
    )
    print("[LOG] Receiver SDR created successfully.")

    sampleRateHz = float(
        suConfig["sample_rate_hz"]
    )

    sensingBandwidthHz = float(suConfig["bandwidth_hz"])
    detectionThresholdDb = float(suConfig["detection_threshold_db"])

    centerFrequencyHz = float(suConfig["center_frequency_hz"])

    print(
        "\n" + "-"*60
    )
    print(
        "Secondary User Receiver Configuration"
    )
    print("-"*60)

    print(
        f"URI: {suConfig['uri']}"
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
        f"{suConfig['buffer_size']}"
    )
    print("-"*60 + "\n")

    # Enable interactive Matplotlib mode.
    print("[LOG] Setting up Matplotlib visualization...")
    plt.ion()

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    line, = ax.plot(
        [],
        [],
    )

    ax.set_xlabel(
        "Frequency (MHz)"
    )

    ax.set_ylabel(
        "Relative Power (dB)"
    )

    ax.set_title(
        "Live PlutoSDR Power Spectrum"
    )

    ax.grid(
        True
    )
    print("[LOG] Matplotlib figure created.")

    fusion_client = FusionClient(
        fusionConfig["host"],
        fusionConfig["port"],
        suConfig["su_id"],
    )
    fusion_client.start()

    # Receive the first buffer.
    print("[LOG] Receiving first sample buffer...")
    samples = receiveSamples(
        sdr
    )
    print(f"[LOG] Received {len(samples)} samples.")

    print("[LOG] Computing power spectrum...")
    frequenciesHz, powerDb = (
        computePowerSpectrum(
            samples=samples,
            sampleRateHz=sampleRateHz,
            centerFrequencyHz=centerFrequencyHz,
        )
    )
    print("[LOG] Power spectrum computed successfully.")

    frequenciesMHz = (
        frequenciesHz / 1e6
    )

    sensingMask = np.abs(
        frequenciesHz - centerFrequencyHz
    ) <= sensingBandwidthHz / 2

    sensingFrequenciesMHz = frequenciesMHz[sensingMask]
    sensingPowerDb = powerDb[sensingMask]
    channelDetections = classify_channels(
        frequenciesHz,
        powerDb,
        channel_plan.channels,
        detectionThresholdDb,
    )
    print_channel_detections(channelDetections, detectionThresholdDb)
    fusion_client.send_iq_frames(
        samples,
        {
            channel_index: occupied
            for channel_index, _, occupied, _ in channelDetections
        },
        {
            channel_index: snr_db
            for channel_index, _, _, snr_db in channelDetections
        },
        len(channel_plan.channels),
    )

    # Configure plot axes once.
    print("[LOG] Configuring plot axes...")
    ax.set_xlim(
        (centerFrequencyHz - sensingBandwidthHz / 2) / 1e6,
        (centerFrequencyHz + sensingBandwidthHz / 2) / 1e6,
    )

    # Initial Y-axis range.
    yMin = np.min(
        sensingPowerDb
    ) - 10

    yMax = np.max(
        sensingPowerDb
    ) + 10

    ax.set_ylim(
        yMin,
        yMax,
    )

    # Set initial data.
    line.set_data(
        sensingFrequenciesMHz,
        sensingPowerDb,
    )

    plt.show(block=False)
    print("[LOG] Matplotlib figure displayed.")

    print("\n[LOG] Starting receiver loop...")
    print("[INFO] Press Ctrl+C to stop the receiver.\n")
    sample_count = 0

    try:

        while True:
            sample_count += 1
            
            if sample_count % 100 == 0:
                print(f"[LOG] Receiving buffer #{sample_count}...")

            samples = receiveSamples(
                sdr
            )

            frequenciesHz, powerDb = (
                computePowerSpectrum(
                    samples=samples,
                    sampleRateHz=sampleRateHz,
                    centerFrequencyHz=centerFrequencyHz,
                )
            )

            frequenciesMHz = (
                frequenciesHz / 1e6
            )

            sensingMask = np.abs(
                frequenciesHz - centerFrequencyHz
            ) <= sensingBandwidthHz / 2

            line.set_data(
                frequenciesMHz[sensingMask],
                powerDb[sensingMask],
            )

            currentPowerDb = powerDb[sensingMask]
            channelDetections = classify_channels(
                frequenciesHz,
                powerDb,
                channel_plan.channels,
                detectionThresholdDb,
            )

            print("========== Sending IQ frames to Fusion Server =====================")
            fusion_client.send_iq_frames(
                samples,
                {
                    channel_index: occupied
                    for channel_index, _, occupied, _ in channelDetections
                },
                {
                    channel_index: snr_db
                    for channel_index, _, _, snr_db in channelDetections
                },
                len(channel_plan.channels),
            )
            if sample_count % 100 == 0:
                print_channel_detections(channelDetections, detectionThresholdDb)
            ax.set_ylim(
                np.min(currentPowerDb) - 10,
                np.max(currentPowerDb) + 10,
            )

            fig.canvas.draw_idle()

            fig.canvas.flush_events()

            plt.pause(
                0.001
            )

    except KeyboardInterrupt:

        print(
            "\n[INFO] Keyboard interrupt received. Stopping receiver..."
        )

    except Exception as e:
        print(
            f"\n[ERROR] Exception occurred: {e}"
        )

    finally:

        print("[LOG] Cleaning up resources...")
        plt.ioff()
        plt.close(fig)
        fusion_client.close()
        print("[LOG] Matplotlib resources released.")

        print(
            f"[LOG] Receiver stopped. Total buffers received: {sample_count}."
        )
        print("[LOG] Secondary User Receiver Shutdown Complete")


if __name__ == "__main__":
    main()
