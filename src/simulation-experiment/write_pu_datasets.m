function write_pu_datasets(outputPath, schedules)
%WRITE_PU_DATASETS Persist PU events in the hardware experiment format.

for puIndex = 1:numel(schedules)
    outputFile = fullfile(outputPath, schedules(puIndex).pu_id + ".jsonl");
    fid = fopen(outputFile, "w");
    if fid < 0
        error("write_pu_datasets:OutputOpenFailed", ...
            "Unable to open %s.", outputFile);
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    for eventIndex = 1:numel(schedules(puIndex).events)
        fprintf(fid, "%s\n", jsonencode(schedules(puIndex).events(eventIndex)));
    end
    clear cleanup;
end
end
