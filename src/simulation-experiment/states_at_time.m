function states = states_at_time(schedules, timeSeconds, channelCount)
%STATES_AT_TIME Return each PU's occupancy state at a simulation time.

states = false(numel(schedules), channelCount);
for puIndex = 1:numel(schedules)
    elapsed = 0;
    events = schedules(puIndex).events;
    for eventIndex = 1:numel(events)
        elapsed = elapsed + events(eventIndex).duration_s;
        if timeSeconds < elapsed || eventIndex == numel(events)
            states(puIndex, :) = events(eventIndex).channel_occupation;
            break;
        end
    end
end
end
