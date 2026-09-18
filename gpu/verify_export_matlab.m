root=fileparts(fileparts(mfilename('fullpath')));
reference=load(fullfile(root,'crc_results','gpu_oracle.mat'));
candidate=load(fullfile(root,'crc_results','gpu_tracks_3980164.mat'));
names={'SR_Localizations','tracks','adjacency_tracks','A'};
for k=1:numel(names)
    name=names{k};
    assert(isequaln(candidate.(name),reference.(name)),['Mismatch: ' name]);
    fprintf('EXACT_MATLAB_MATCH %s\n',name);
end
report=struct('passed',true,'matlab',version,'fields',{names},'tracks',numel(candidate.tracks),'links',nnz(candidate.A));
fid=fopen(fullfile(root,'crc_results','matlab_gpu_export_verification.json'),'w');fprintf(fid,'%s\n',jsonencode(report));fclose(fid);
fprintf('MATLAB_EXPORT_VERIFIED\n');
