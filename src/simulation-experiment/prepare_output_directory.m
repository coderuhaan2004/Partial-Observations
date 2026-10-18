function prepare_output_directory(outputPath)
%PREPARE_OUTPUT_DIRECTORY Remove only generated SU/PU JSON files.

if ~isfolder(outputPath)
    mkdir(outputPath);
    return;
end
for pattern = ["su-*.json", "pu-*.json"]
    files = dir(fullfile(outputPath, pattern));
    for index = 1:numel(files)
        delete(fullfile(outputPath, files(index).name));
    end
end
end
