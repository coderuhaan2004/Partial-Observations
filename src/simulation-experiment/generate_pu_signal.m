function signal = generate_pu_signal(plan, activeChannels, n, fs)
%GENERATE_PU_SIGNAL Create the complex IQ passed directly to adi.Pluto.tx().

t = (0:n - 1).' / fs;
signal = zeros(n, 1);
for channelIndex = activeChannels
    component = exp(1i * 2 * pi * plan.offset_hz(channelIndex) * t);
    signal = signal + component;
end
signal = single(2^13 * signal / numel(activeChannels));
end
