function [samples, occupied, snrDb] = receive_frame( ...
    plan, pu, states, links, su, frameSize, sampleRate)
%RECEIVE_FRAME Sum PU waveforms through fixed PU-SU links and add AWGN.

% A calibrated complex-noise floor; gain is applied as a linear amplitude
% factor, while the SU hardware gain remains a reproducible experiment input.
noisePower = 10^((-90 + 10 * log10(max(su.rx_hardwaregain_chan0 + 1, 1))) / 10);
samples = sqrt(noisePower / 2) * ...
    (randn(frameSize, 1) + 1i * randn(frameSize, 1));
occupied = false(1, plan.channel_count);
snrDb = nan(1, plan.channel_count);

for puIndex = 1:numel(pu)
    activeChannels = find(states(puIndex, :));
    if isempty(activeChannels)
        continue;
    end
    occupied(activeChannels) = true;
    tx = generate_pu_signal(plan, activeChannels, ...
        frameSize, sampleRate);
    txGain = 10^(pu(puIndex).tx_hardwaregain_chan0 / 20);
    samples = samples + links(puIndex).complex_gain * txGain * tx;
end

noiseFloor = max(noisePower, realmin);
for channelIndex = find(occupied)
    channelSamples = samples .* exp(-1i * 2 * pi * ...
        plan.offset_hz(channelIndex) * (0:frameSize - 1).' / sampleRate);
    signalPower = max(mean(abs(channelSamples).^2) - noiseFloor, realmin);
    snrDb(channelIndex) = 10 * log10(signalPower / noiseFloor);
end
end
