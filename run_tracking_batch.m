function report = run_tracking_batch(inputFiles, outputDir, mode, verify, pointsVariable, maxLinkingDistance, maxGapClosing)
% RUN_TRACKING_BATCH Track MATLAB point-cell arrays without overwriting input.
% mode: 'blocks' (parallel files), 'frames' (parallel frame pairs), 'serial'.
% verify defaults true: run independent serial reference and require exact
% equality of tracks, adjacency_tracks, and A before writing each output.
% Create a pool explicitly to control workers/memory before calling this.
if nargin < 3, mode = 'blocks'; end
if nargin < 4, verify = true; end
if nargin < 5, pointsVariable = 'SR_Localizations'; end
if nargin < 6, maxLinkingDistance = 8; end
if nargin < 7, maxGapClosing = 1; end
assert(isvarname(pointsVariable), 'pointsVariable must be a MATLAB variable name.');
assert(isfinite(maxLinkingDistance) && maxLinkingDistance >= 0, ...
    'maxLinkingDistance must be finite and nonnegative.');
assert(isfinite(maxGapClosing) && maxGapClosing >= 1 && fix(maxGapClosing) == maxGapClosing, ...
    'maxGapClosing must be a positive integer.');
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
    report{k} = process_one(inputFiles{k}, outputDir, k, mode, verify, ...
        pointsVariable, maxLinkingDistance, maxGapClosing);
end
save(fullfile(outputDir, 'benchmark_report.mat'), 'report', 'mode', 'verify', ...
    'pointsVariable', 'maxLinkingDistance', 'maxGapClosing');
end

function info = process_one(inputFile, outputDir, index, mode, verify, ...
    pointsVariable, maxLinkingDistance, maxGapClosing)
data = load(inputFile, pointsVariable);
assert(isfield(data, pointsVariable), 'Missing %s in %s', pointsVariable, inputFile);
points = data.(pointsVariable);
args = {'MaxLinkingDistance',maxLinkingDistance, ...
    'MaxGapClosing',maxGapClosing,'Debug',false};
t = tic;
[tracks, adjacency_tracks, A] = simpletracker_fast(points, ...
    args{:}, 'UseParallel', strcmp(mode, 'frames'));
info = struct('inputFile',inputFile,'pointsVariable',pointsVariable, ...
    'maxLinkingDistance',maxLinkingDistance,'maxGapClosing',maxGapClosing, ...
    'frames',numel(points),'points',sum(cellfun(@(x) size(x,1), points)), ...
    'optimizedSeconds',toc(t),'referenceSeconds',NaN,'exactMatch',false);
if verify
    t = tic;
    [refTracks, refAdj, refA] = simpletracker_serial_reference(points,args{:});
    info.referenceSeconds = toc(t);
    assert(isequaln(tracks,refTracks) && isequaln(adjacency_tracks,refAdj) ...
        && isequaln(A,refA), 'Tracking output differs for %s',inputFile);
    info.exactMatch = true;
end
% Sequential file numbering preserves caller-supplied block order.
output = struct();
output.(pointsVariable) = points;
output.tracks = tracks;
output.adjacency_tracks = adjacency_tracks;
output.info = info;
save(fullfile(outputDir,sprintf('block_%04d_tracked.mat',index)), ...
    '-struct','output','-v7.3');
end
