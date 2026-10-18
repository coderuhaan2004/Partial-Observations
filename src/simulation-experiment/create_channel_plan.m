function plan = create_channel_plan(cfg)
%CREATE_CHANNEL_PLAN Derive the ten 1 MHz channels from experiment-config.yaml.

plan = cfg.channelPlan;
plan.channel_count = floor(plan.total_bandwidth_hz / plan.channel_bandwidth_hz);
plan.offset_hz = ((1:plan.channel_count) - (plan.channel_count + 1) / 2) ...
    * plan.channel_spacing_hz;
end
