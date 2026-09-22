from dataclasses import dataclass

@dataclass
class SpectrumConfig:
    center_frequency_hz: float  # Center frequency of wideband

    total_bandwidth_hz: float   # total frequency bandwidth of wideband

    channel_bandwidth_hz: float # Channel frequency width 
    channel_spacing_hz: float   # Gap between 2 channels

    sample_rate_hz: float       # sampling rate of data

@dataclass
class Channel:
    index: int                  # Channel no.

    center_frequency_hz: float  # Center frequency of channel

    lower_frequency_hz: float   # lower bound of frequency
    upper_frequency_hz: float   # upper bound of frequency

@dataclass
class SecondaryUserConfig:
    uri:                    str         
    center_frequency_hz:    float
    sample_rate_hz:         float
    bandwidth_hz:           float
    buffer_size:            int = 65536