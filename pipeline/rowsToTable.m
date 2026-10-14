function T = rowsToTable(rows)
% evalReference rows (a cell of structs, possibly nested one level) as one
% table, with each per-band vector field spread over one column per octave
% band (<field>_125 ..., <field>_8000), ready for writetable.
bands = [125 250 500 1000 2000 4000 8000];
rows = [rows{:}];
if iscell(rows); rows = [rows{:}]; end
T = struct2table(rows, 'AsArray', true);
for v = {'refCohBands','refFiltVarBands','t20Err','rmsAbsErr'}
    if ~ismember(v{1}, T.Properties.VariableNames); continue; end
    M = T.(v{1}); T.(v{1}) = [];
    T = [T, array2table(M, 'VariableNames', arrayfun(@(f) sprintf('%s_%d', v{1}, f), bands, 'uni', 0))]; %#ok<AGROW>
end
end
