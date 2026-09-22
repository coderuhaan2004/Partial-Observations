'''
Helper functions that will be used repeatedly
'''

import numpy as np
import subprocess

def getURI() -> str:
    """
    get URI of current secondary user by running iio_info -s shell command
    """
    try:
        result = subprocess.run(
            ["iio_info", "-s"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing iio_info -s: {e}")
        return ""
    except FileNotFoundError:
        print("iio_info command not found. Ensure libiio is installed.")
        return ""
    
def configureSDR(
    sdr,
    center_frequency_hz: float,
    sample_rate_hz: float,
    bandwidth_hz: float,
    is_tx: bool = False,
):
    """
    Configure SDR for either TX or RX mode.
    
    Args:
        sdr: The PlutoSDR device object
        center_frequency_hz: Center frequency to set
        sample_rate_hz: Sample rate in Hz
        bandwidth_hz: RF bandwidth in Hz
        is_tx: If True, configure for TX; if False, configure for RX
    """
    sdr.sample_rate = int(sample_rate_hz)
    
    if is_tx:
        sdr.tx_rf_bandwidth = int(bandwidth_hz)
        sdr.tx_lo = int(center_frequency_hz)
    else:
        sdr.rx_rf_bandwidth = int(bandwidth_hz)
        sdr.rx_lo = int(center_frequency_hz)


def computePowerSpectrum(
    samples: np.ndarray,
    sampleRateHz: float,
    centerFrequencyHz: float,
):
    """
    Compute a windowed power spectrum.

    Returns:
        frequenciesHz:
            Absolute RF frequency for each FFT bin.

        powerDb:
            Relative power in dB.
    """

    numSamples = len(samples)

    window = np.hanning(
        numSamples
    )

    windowedSamples = (
        samples * window
    )

    spectrum = np.fft.fftshift(
        np.fft.fft(
            windowedSamples
        )
    )

    power = (
        np.abs(spectrum) ** 2
    )

    powerDb = (
        10
        * np.log10(
            power + 1e-12
        )
    )

    frequencyOffsetsHz = np.fft.fftshift(
        np.fft.fftfreq(
            numSamples,
            d=1 / sampleRateHz,
        )
    )

    frequenciesHz = (
        frequencyOffsetsHz
        + centerFrequencyHz
    )

    return frequenciesHz, powerDb