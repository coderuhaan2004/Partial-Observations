function schedules = create_occupancy_schedules(durationSeconds, channelCount, puIds)
%CREATE_OCCUPANCY_SCHEDULES Generate hardware-compatible PU event logs.

template = struct("timestamp", "", "channel_occupation", false(1, channelCount), ...
    "channel_indices", [], "duration_s", 0);
schedules = repmat(struct("pu_id", "", "events", template([])), 1, numel(puIds));
baseTimestamp = datetime("now", "TimeZone", "UTC", ...
    "Format", "yyyy-MM-dd'T'HH:mm:ss.SSSXXX");
for puIndex = 1:numel(puIds)
    elapsed = 0;
    events = template([]);
    while elapsed < durationSeconds
        eventDuration = min(10 + 50 * rand(), durationSeconds - elapsed);
        activeCount = randi([0, min(2, channelCount)]);
        active = false(1, channelCount);
        if activeCount > 0
            active(randperm(channelCount, activeCount)) = true;
        end
        event = template;
        event.timestamp = char(baseTimestamp + seconds(elapsed)); %#ok<AGROW>
        event.channel_occupation = active; %#ok<AGROW>
        event.channel_indices = find(active) - 1; %#ok<AGROW>
        event.duration_s = eventDuration; %#ok<AGROW>
        events(end + 1) = event; %#ok<AGROW>
        elapsed = elapsed + eventDuration;
    end
    schedules(puIndex).pu_id = puIds{puIndex};
    schedules(puIndex).events = events;
end
end
