function inspect_dataset(datasetPath, suId, frameIndex, sampleRateHz)
%INSPECT_DATASET Plot a generated SU frame so PU signals are visible.

arguments
    datasetPath (1,1) string
    suId (1,1) string = "su-001"
    frameIndex (1,1) double {mustBeInteger, mustBePositive} = 1
    sampleRateHz (1,1) double {mustBePositive} = 10e6
end

payload = jsondecode(fileread(fullfile(datasetPath, suId + ".json")));
if frameIndex > numel(payload)
    error("inspect_dataset:Frame", "Frame index exceeds the generated dataset.");
end
frame = payload(frameIndex);
iq = frame.iq(:, 1) + 1i * frame.iq(:, 2);
n = numel(iq);
window = 0.5 - 0.5 * cos(2 * pi * (0:n - 1).' / max(n - 1, 1));
spectrum = fftshift(fft(iq .* window));
powerDb = 10 * log10(abs(spectrum).^2 / n + realmin);
frequencyMHz = fftshift((-floor(n / 2):ceil(n / 2) - 1) * ...
    sampleRateHz / n) / 1e6;

figure("Name", "Simulated SU spectrum");
plot(frequencyMHz, powerDb, "LineWidth", 1);
grid on;
xlabel("Frequency offset (MHz)");
ylabel("Power (dB, relative)");
title(sprintf("%s frame %d; occupied channels: %s", suId, frameIndex, ...
    mat2str(find(frame.channels) - 1)));
end
