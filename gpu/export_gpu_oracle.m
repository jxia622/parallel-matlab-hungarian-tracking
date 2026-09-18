root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'baseline'));
load(fullfile(root,'fixture_bloc1_track.mat'),'SR_Localizations');
out=fullfile(root,'crc_results');
c=parcluster('local'); c.JobStorageLocation=fullfile(out,['oracle_pool_' getenv('SLURM_JOB_ID')]);
parpool(c,4);
profile on;
for i=1:8
    hungarianlinker(SR_Localizations{i},SR_Localizations{i+1},8);
end
profile off;
p=profile('info'); save(fullfile(out,'matlab_profile.mat'),'p');
fprintf('PROFILE_READY\n');
t=tic;
[tracks,adjacency_tracks,A]=simpletracker_fast(SR_Localizations,'MaxLinkingDistance',8,'MaxGapClosing',1,'Debug',false);
seconds=toc(t);
save(fullfile(out,'gpu_oracle.mat'),'SR_Localizations','tracks','adjacency_tracks','A','seconds','-v7');
fprintf('ORACLE_READY %.3f seconds\n',seconds);
% Linker edge cases, including zero distances, duplicate points, threshold ties.
rng(42); fixtures={}; expected={};
fixtures{end+1}={zeros(0,2),zeros(0,2)};
fixtures{end+1}={zeros(0,2),[1 1]};
fixtures{end+1}={[1 1],zeros(0,2)};
fixtures{end+1}={[0 0;0 0],[0 0;0 0]};
fixtures{end+1}={[0 0;2 0],[1 1;1 -1]};
fixtures{end+1}={[0 0;100 100],[8 0;108+eps(108) 100]};
for k=1:100
    fixtures{end+1}={round(rand(randi([1 35]),2)*50)/2,round(rand(randi([1 35]),2)*50)/2};
end
for k=1:numel(fixtures)
    expected{k}=hungarianlinker(fixtures{k}{1},fixtures{k}{2},8);
end
save(fullfile(out,'gpu_edge_oracle.mat'),'fixtures','expected','-v7');
delete(gcp('nocreate'));
