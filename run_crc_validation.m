root = fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'baseline'),fullfile(root,'tests'));
out = fullfile(root,'crc_results');
if ~isfolder(out), mkdir(out); end
fprintf('MATLAB %s on %s\n',version,getenv('HOSTNAME'));
disp(ver('parallel'));
fprintf('Parallel toolbox license: %d\n',license('test','Distrib_Computing_Toolbox'));
t = tic;
c = parcluster('local');
c.JobStorageLocation = fullfile(out,['pool_' getenv('SLURM_JOB_ID')]);
if ~isfolder(c.JobStorageLocation), mkdir(c.JobStorageLocation); end
parpool(c,4);
fprintf('Pool startup: %.3f seconds\n',toc(t));
if ~isfile(fullfile(out,'equivalence_tests.mat'))
    serialTests = test_tracking_equivalence(false);
    parallelTests = test_tracking_equivalence(true);
    save(fullfile(out,'equivalence_tests.mat'),'serialTests','parallelTests');
else
    load(fullfile(out,'equivalence_tests.mat'),'serialTests','parallelTests');
    assert(serialTests.passed && parallelTests.passed);
end
fprintf('EQUIVALENCE_TESTS_PASSED\n');
data = load(fullfile(root,'fixture_bloc1_track.mat'),'SR_Localizations');
SR_Localizations = data.SR_Localizations(1:8);
save(fullfile(out,'excerpt8.mat'),'SR_Localizations');
quick = benchmark_tracking(fullfile(out,'excerpt8.mat'),4,3);
save(fullfile(out,'quick_benchmark.mat'),'quick');
fid=fopen(fullfile(out,'quick_benchmark.json'),'w'); fprintf(fid,'%s\n',jsonencode(quick)); fclose(fid);
fprintf('QUICK_BENCHMARK_PASSED\n');
full = benchmark_tracking(fullfile(root,'fixture_bloc1_track.mat'),4,1,false,false);
save(fullfile(out,'full_benchmark.mat'),'full');
fid=fopen(fullfile(out,'full_benchmark.json'),'w'); fprintf(fid,'%s\n',jsonencode(full)); fclose(fid);
fprintf('FULL_BENCHMARK_PASSED\n');
delete(gcp('nocreate'));
