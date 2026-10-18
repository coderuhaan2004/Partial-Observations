function write_pu_datasets(outputPath, schedules)
%WRITE_PU_DATASETS Persist PU events in the hardware experiment format.

for puIndex = 1:numel(schedules)
    write_json(fullfile(outputPath, schedules(puIndex).pu_id + ".json"), ...
        schedules(puIndex).events);
end
end
