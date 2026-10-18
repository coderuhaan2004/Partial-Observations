function validate_simulation_config(plan, su, pu)
%VALIDATE_SIMULATION_CONFIG Check assumptions before generating data.

if plan.channel_count ~= 10
    error("validate_simulation_config:ChannelCount", ...
        "Expected 10 channels from experiment-config.yaml.");
end
requiredSu = ["sample_rate_hz", "buffer_size", "rx_hardwaregain_chan0", "su_id"];
requiredPu = ["sample_rate_hz", "tx_hardwaregain_chan0", "pu_id"];
for field = requiredSu
    if ~isfield(su, field)
        error("validate_simulation_config:SUField", "Missing SU field %s.", field);
    end
end
for field = requiredPu
    if ~isfield(pu, field)
        error("validate_simulation_config:PUField", "Missing PU field %s.", field);
    end
end
end
