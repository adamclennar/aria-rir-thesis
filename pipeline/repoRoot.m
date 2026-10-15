function root = repoRoot()
% The repository's root folder, for any file in pipeline/ or its subfolders.
root = fileparts(fileparts(mfilename('fullpath')));
end
