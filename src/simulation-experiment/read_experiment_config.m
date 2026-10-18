function cfg = read_experiment_config(path)
%READ_EXPERIMENT_CONFIG Read the scalar YAML used by the hardware experiment.

lines = splitlines(string(fileread(path)));
cfg.secondary = struct([]);
cfg.primary = struct([]);
cfg.channelPlan = struct();
section = "";
for rawLine = lines.'
    line = strip_yaml_comment(strtrim(rawLine));
    if line == ""
        continue;
    end
    if endsWith(line, ":") && ~contains(line, '"')
        section = erase(line, ":");
        continue;
    end
    device = regexp(line, '^"([^"]+)":\s*$', "tokens", "once");
    if ~isempty(device)
        if section == "Secondary_User"
            cfg.secondary(end + 1).source_key = string(device{1}); %#ok<AGROW>
        elseif section == "Primary_User"
            cfg.primary(end + 1).source_key = string(device{1}); %#ok<AGROW>
        end
        continue;
    end
    keyValue = regexp(line, '^"?([^":]+)"?\s*:\s*(.+)$', "tokens", "once");
    if isempty(keyValue)
        continue;
    end
    key = matlab.lang.makeValidName(strtrim(keyValue{1}));
    value = parse_yaml_scalar(strtrim(keyValue{2}));
    switch section
        case "ChannelPlan"
            cfg.channelPlan.(key) = value;
        case "Secondary_User"
            cfg.secondary(end).(key) = value;
        case "Primary_User"
            cfg.primary(end).(key) = value;
    end
end

required = ["center_frequency_hz", "total_bandwidth_hz", ...
    "channel_bandwidth_hz", "channel_spacing_hz", "sample_rate_hz"];
for key = required
    if ~isfield(cfg.channelPlan, key)
        error("read_experiment_config:MissingField", ...
            "Missing ChannelPlan.%s in %s.", key, path);
    end
end
end

function line = strip_yaml_comment(line)
index = strfind(line, "#");
if ~isempty(index)
    line = strtrim(extractBefore(line, index(1)));
end
end

function value = parse_yaml_scalar(rawValue)
if startsWith(rawValue, '"') && endsWith(rawValue, '"')
    value = erase(rawValue, '"');
elseif strcmpi(rawValue, "true")
    value = true;
elseif strcmpi(rawValue, "false")
    value = false;
else
    numeric = str2double(rawValue);
    if isnan(numeric)
        value = string(rawValue);
    else
        value = numeric;
    end
end
end
