function main(configPath, outputPath, durationSeconds, randomSeed, frameRateHz)
%MAIN Run the three-PU/three-SU hardware-equivalent simulation.

arguments
    configPath (1,1) string = ""
    outputPath (1,1) string = ""
    durationSeconds (1,1) double {mustBePositive} = 12 * 60 * 60
    randomSeed (1,1) double = 7
    frameRateHz (1,1) double {mustBePositive} = 9.18
end

root = fileparts(mfilename("fullpath"));
if configPath == ""
    configPath = fullfile(root, "experiment-config.yaml");
end
if outputPath == ""
    outputPath = fullfile(root, "data");
end
fprintf("[SIM] Config: %s\n", configPath);
fprintf("[SIM] Output: %s\n", outputPath);
fprintf("[SIM] Duration: %.2f hours (%.0f seconds)\n", ...
    durationSeconds / 3600, durationSeconds);
fprintf("[SIM] Persistence rate: %.2f frames/sec\n", frameRateHz);
rng(randomSeed, "twister");
cfg = read_experiment_config(configPath);
[su, pu] = expand_devices(cfg, 3);
layout = create_layout();
channelPlan = create_channel_plan(cfg);
validate_simulation_config(channelPlan, su, pu);

sampleRate = su(1).sample_rate_hz;
frameSize = su(1).buffer_size;
frameCount = floor(durationSeconds * frameRateHz);
if frameCount < 1
    error("main:Duration", "Duration must produce at least one frame.");
end
fprintf("[SIM] Devices: %d SUs, %d PUs\n", numel(su), numel(pu));
fprintf("[SIM] Channels: %d; samples/frame: %d; total frames/SU: %d\n", ...
    channelPlan.channel_count, frameSize, frameCount);
fprintf("[SIM] SDR sample rate: %.0f samples/sec\n", sampleRate);

prepare_output_directory(outputPath);
schedules = create_occupancy_schedules(durationSeconds, ...
    channelPlan.channel_count, {pu.pu_id});
write_pu_datasets(outputPath, schedules);
fprintf("[SIM] PU schedules written.\n");

links = create_link_gains(layout, su, pu);
baseTimestamp = datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
progressStep = max(1, ceil(frameCount / 10));
for suIndex = 1:numel(su)
    outputFile = fullfile(outputPath, su(suIndex).su_id + ".jsonl");
    fid = fopen(outputFile, "w");
    if fid < 0
        error("main:OutputOpenFailed", "Unable to open %s.", outputFile);
    end
    fprintf("[SIM] Writing %s (%d/%d): %s\n", ...
        su(suIndex).su_id, suIndex, numel(su), outputFile);
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    for frameIndex = 1:frameCount
        frameTime = (frameIndex - 1) / frameRateHz;
        states = states_at_time(schedules, frameTime, ...
            channelPlan.channel_count);
        [samples, occupied, localChannels, snrDb] = receive_frame( ...
            channelPlan, pu, states, links(suIndex, :), su(suIndex), ...
            frameSize, sampleRate);
        frame.timestamp = char(baseTimestamp + seconds(frameTime));
        frame.frame_id = frameIndex - 1;
        iq = round([real(samples), imag(samples)]);
        iq = max(min(iq, double(intmax("int16"))), ...
            double(intmin("int16")));
        frame.iq = int16(iq);
        frame.channels = occupied;
        frame.local_channels = localChannels;
        frame.snr_db = snrDb;
        fprintf(fid, "%s\n", jsonencode(frame));
        if mod(frameIndex, progressStep) == 0 || frameIndex == frameCount
            fprintf("[SIM] %s: %d/%d frames (%.0f%%)\n", ...
                su(suIndex).su_id, frameIndex, frameCount, ...
                100 * frameIndex / frameCount);
        end
    end
    clear cleanup;
    fileInfo = dir(outputFile);
    fprintf("[SIM] Finished %s: %.2f GB\n", ...
        su(suIndex).su_id, fileInfo.bytes / 1e9);
end

fprintf("Generated %d frames for 3 SUs and 3 PUs in %s\n", ...
    frameCount, outputPath);
fprintf("[SIM] Complete.\n");
end
