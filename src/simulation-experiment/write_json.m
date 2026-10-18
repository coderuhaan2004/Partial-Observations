function write_json(path, value)
%WRITE_JSON Write compact JSON compatible with the Python fusion server.

fid = fopen(path, "w");
if fid < 0
    error("write_json:OpenFailed", "Unable to open %s.", path);
end
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, "%s", jsonencode(value, "PrettyPrint", false));
end
