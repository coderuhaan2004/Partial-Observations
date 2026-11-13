function signal = generate_pu_signal(plan, activeChannels, n, fs)
%GENERATE_PU_SIGNAL Create simple pulse-shaped random PU channel signals.
%
% Each occupied channel carries an independent BPSK symbol stream. The
% square-root raised-cosine pulse gives each channel a finite bandwidth and
% realistic sidelobes, rather than producing a deterministic carrier tone.

t = (0:n - 1).' / fs;
signal = zeros(n, 1);
samplesPerSymbol = fs / plan.channel_bandwidth_hz;
spanSymbols = 8;
if samplesPerSymbol < 1 || samplesPerSymbol ~= floor(samplesPerSymbol)
    error("generate_pu_signal:SymbolRate", ...
        "Sample rate must be an integer multiple of channel bandwidth.");
end

for channelIndex = activeChannels
    symbolCount = ceil(n / samplesPerSymbol) + spanSymbols;
    symbols = 2 * randi([0, 1], symbolCount, 1) - 1;
    baseband = pulse_shape(symbols, samplesPerSymbol, spanSymbols);
    filterDelay = spanSymbols * samplesPerSymbol / 2;
    baseband = baseband(filterDelay + 1:filterDelay + n);
    component = baseband .* exp(1i * 2 * pi * ...
        plan.offset_hz(channelIndex) * t);
    signal = signal + component;
end

if ~isempty(activeChannels)
    signal = signal / sqrt(numel(activeChannels));
end
signal = single(2^13 * signal);
end

function output = pulse_shape(symbols, samplesPerSymbol, spanSymbols)
%PULSE_SHAPE Apply a compact square-root raised-cosine pulse.

rolloff = 0.35;
filter = rcosdesign(rolloff, spanSymbols, samplesPerSymbol, "sqrt");
upsampled = zeros(numel(symbols) * samplesPerSymbol, 1);
upsampled(1:samplesPerSymbol:end) = symbols;
output = conv(upsampled, filter);
end
