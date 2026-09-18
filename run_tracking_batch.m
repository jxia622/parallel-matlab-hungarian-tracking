function report = run_tracking_batch(inputFiles, outputDir, mode, verify)
% RUN_TRACKING_BATCH Track existing SR_Localizations without overwriting input.
% mode: 'blocks' (parallel files), 'frames' (parallel frame pairs), 'serial'.
% verify defaults true: run independent serial reference and require exact
% equality of tracks, adjacency_tracks, and A before writing each output.
% Create a pool explicitly to control workers/memory before calling this.
if nargin < 3, mode = 'blocks'; end
if nargin < 4, verify = true; end
mode = validatestring(mode, {'blocks','frames','serial'});
inputFiles = cellstr(inputFiles);
assert(~isempty(inputFiles), 'No input files supplied.');
assert(~isfolder(outputDir), 'Use a new output directory to prevent overwrites.');
root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'baseline'), fullfile(root, 'tests'), root);
for k = 1:numel(inputFiles)
    assert(isfile(inputFiles{k}), 'Missing input: %s', inputFiles{k});
end
mkdir(outputDir);
worker_limit = 0;
if strcmp(mode, 'blocks'), worker_limit = Inf; end
report = cell(numel(inputFiles), 1);
parfor (k = 1:numel(inputFiles), worker_limit)
    report{k} = process_one(inputFiles{k}, outputDir, k, mode, verify);
end
save(fullfile(outputDir, 'benchmark_report.mat'), 'report', 'mode', 'verify');
end

function info = process_one(inputFile, outputDir, index, mode, verify)
data = load(inputFile, 'SR_Localizations');
assert(isfield(data, 'SR_Localizations'), 'Missing SR_Localizations in %s', inputFile);
SR_Localizations = data.SR_Localizations;
args = {'MaxLinkingDistance',8,'MaxGapClosing',1,'Debug',false};
t = tic;
[tracks, adjacency_tracks, A] = simpletracker_fast(SR_Localizations, ...
    args{:}, 'UseParallel', strcmp(mode, 'frames'));
info = struct('inputFile',inputFile,'frames',numel(SR_Localizations), ...
    'points',sum(cellfun(@(x) size(x,1), SR_Localizations)), ...
    'optimizedSeconds',toc(t),'referenceSeconds',NaN,'exactMatch',false);
if verify
    t = tic;
    [refTracks, refAdj, refA] = simpletracker_serial_reference(SR_Localizations,args{:});
    info.referenceSeconds = toc(t);
    assert(isequaln(tracks,refTracks) && isequaln(adjacency_tracks,refAdj) ...
        && isequaln(A,refA), 'Tracking output differs for %s',inputFile);
    info.exactMatch = true;
end
% Sequential file numbering preserves caller-supplied block order.
save(fullfile(outputDir,sprintf('block_%04d_tracked.mat',index)), ...
    'SR_Localizations','tracks','adjacency_tracks','info','-v7.3');
end
