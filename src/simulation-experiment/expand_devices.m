function [su, pu] = expand_devices(cfg, count)
%EXPAND_DEVICES Use configured devices and explicitly create three logical radios.

if isempty(cfg.secondary) || isempty(cfg.primary)
    error("expand_devices:MissingDevices", "The YAML must define SU and PU devices.");
end
if numel(cfg.secondary) < count || numel(cfg.primary) < count
    warning("expand_devices:TemplateExpansion", ...
        "Missing logical device settings are copied from the first configured template.");
end
su = repmat(cfg.secondary(1), 1, count);
pu = repmat(cfg.primary(1), 1, count);
for index = 1:count
    su(index) = cfg.secondary(min(index, numel(cfg.secondary)));
    pu(index) = cfg.primary(min(index, numel(cfg.primary)));
    su(index).su_id = string(sprintf("su-%03d", index));
    pu(index).pu_id = string(sprintf("pu-%03d", index));
end
end
