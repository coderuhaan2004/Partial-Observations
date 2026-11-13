function [detected, snrDb] = detect_local_channels(samples, plan, thresholdDb)
%DETECT_LOCAL_CHANNELS Apply the SU energy detector to one IQ frame.

n = numel(samples);
window = 0.5 - 0.5 * cos(2 * pi * (0:n - 1).' / n);
spectrum = fftshift(fft(samples .* window));
powerDb = 10 * log10(abs(spectrum).^2 + 1e-12);
frequencyHz = fftshift((-floor(n / 2):ceil(n / 2) - 1) * ...
    plan.sample_rate_hz / n);

channelPowerDb = zeros(1, plan.channel_count);
for channelIndex = 1:plan.channel_count
    channelCenter = plan.offset_hz(channelIndex);
    channelMask = abs(frequencyHz - channelCenter) < ...
        plan.channel_bandwidth_hz / 2;
    if ~any(channelMask)
        error("detect_local_channels:NoBins", ...
            "No FFT bins found for channel %d.", channelIndex - 1);
    end
    channelPowerDb(channelIndex) = max(powerDb(channelMask));
end

available = channelPowerDb(channelPowerDb <= thresholdDb);
if isempty(available)
    noisePowerDb = prctile(channelPowerDb, 25);
else
    noisePowerDb = mean(available);
end

detected = channelPowerDb > thresholdDb;
snrDb = -inf(1, plan.channel_count);
for channelIndex = find(detected)
    signalPower = max(10^(channelPowerDb(channelIndex) / 10) - ...
        10^(noisePowerDb / 10), 0);
    if signalPower > 0
        snrDb(channelIndex) = 10 * log10(signalPower / ...
            10^(noisePowerDb / 10));
    end
end
end
