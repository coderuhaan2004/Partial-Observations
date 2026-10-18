function main(configPath, outputPath, durationSeconds, randomSeed)
%MAIN Run the three-PU/three-SU hardware-equivalent simulation.

arguments
    configPath (1,1) string = ""
    outputPath (1,1) string = ""
    durationSeconds (1,1) double {mustBePositive} = 2
    randomSeed (1,1) double = 7
end

root = fileparts(mfilename("fullpath"));
if configPath == ""
    configPath = fullfile(root, "..", "..", "experiment-config.yaml");
end
if outputPath == ""
    outputPath = fullfile(root, "data");
end
rng(randomSeed, "twister");
cfg = read_experiment_config(configPath);
[su, pu] = expand_devices(cfg, 3);
layout = create_layout();
channelPlan = create_channel_plan(cfg);
validate_simulation_config(channelPlan, su, pu);

sampleRate = su(1).sample_rate_hz;
frameSize = su(1).buffer_size;
frameCount = floor(durationSeconds * sampleRate / frameSize);
if frameCount < 1
    error("main:Duration", "Duration must produce at least one frame.");
end

prepare_output_directory(outputPath);
schedules = create_occupancy_schedules(durationSeconds, ...
    channelPlan.channel_count, {pu.pu_id});
write_pu_datasets(outputPath, schedules);

links = create_link_gains(layout, su, pu);
baseTimestamp = datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
for suIndex = 1:numel(su)
    frames = repmat(struct(), 1, frameCount);
    for frameIndex = 1:frameCount
        frameTime = (frameIndex - 1) * frameSize / sampleRate;
        states = states_at_time(schedules, frameTime, ...
            channelPlan.channel_count);
        [samples, occupied, snrDb] = receive_frame( ...
            channelPlan, pu, states, links(suIndex, :), su(suIndex), ...
            frameSize, sampleRate);
        frames(frameIndex).timestamp = char(baseTimestamp + seconds(frameTime));
        frames(frameIndex).frame_id = frameIndex - 1;
        frames(frameIndex).iq = [real(samples), imag(samples)];
        frames(frameIndex).channels = occupied;
        frames(frameIndex).snr_db = snrDb;
    end
    write_json(fullfile(outputPath, su(suIndex).su_id + ".json"), frames);
end

fprintf("Generated %d frames for 3 SUs and 3 PUs in %s\n", ...
    frameCount, outputPath);
end
