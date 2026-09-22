import math
from config import SpectrumConfig, Channel

class ChannelPlan:

    def __init__(self, config: SpectrumConfig):
        self.config = config
        self.channels = self._create_channels()

    def get_channel(self, ch_idx) -> Channel:
        return self.channels[ch_idx]

    def _create_channels(self) -> list[Channel]:

        cfg = self.config

        num_channels = math.floor(
            cfg.total_bandwidth_hz /
            cfg.channel_spacing_hz
        )

        spectrum_start = (
            cfg.center_frequency_hz
            - cfg.total_bandwidth_hz / 2
        )

        channels = []

        for i in range(num_channels):

            channel_center = (
                spectrum_start
                + cfg.channel_spacing_hz / 2
                + i * cfg.channel_spacing_hz
            )

            lower = (
                channel_center
                - cfg.channel_bandwidth_hz / 2
            )

            upper = (
                channel_center
                + cfg.channel_bandwidth_hz / 2
            )

            channels.append(
                Channel(
                    index=i,
                    center_frequency_hz=channel_center,
                    lower_frequency_hz=lower,
                    upper_frequency_hz=upper,
                )
            )

        return channels


def validate_channel_indices(channel_plan: ChannelPlan, channel_indices: list[int]):
    if not channel_indices:
        raise ValueError("At least one channel must be selected")

    if channel_indices != list(range(channel_indices[0], channel_indices[-1] + 1)):
        raise ValueError("Selected channels must be contiguous")

    for channel_index in channel_indices:
        channel_plan.get_channel(channel_index)


def get_channel_group(
    channel_plan: ChannelPlan,
    channel_indices: list[int],
    require_contiguous: bool = True,
):
    if require_contiguous:
        validate_channel_indices(channel_plan, channel_indices)
    elif not channel_indices:
        raise ValueError("At least one channel must be selected")

    channels = [channel_plan.get_channel(channel_index) for channel_index in channel_indices]
    center_frequency_hz = (
        channels[0].center_frequency_hz + channels[-1].center_frequency_hz
    ) / 2
    bandwidth_hz = channels[-1].upper_frequency_hz - channels[0].lower_frequency_hz
    return channels, center_frequency_hz, bandwidth_hz