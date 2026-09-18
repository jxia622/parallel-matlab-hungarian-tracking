function report = benchmark_tracking(inputFile, workers, repeats, includeSerial, warmup)
% Compare against the actual already-parallel baseline with exact checks.
% includeSerial=false omits a potentially expensive serial timing.
% warmup=false is for a quick single-pass comparison, not steady-state timing.
if nargin < 2, workers = 4; end
if nargin < 3, repeats = 3; end
if nargin < 4, includeSerial = true; end
if nargin < 5, warmup = true; end
root = fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'baseline'),fullfile(root,'tests'));
d = load(inputFile,'SR_Localizations');
points = d.SR_Localizations;
t = tic;
pool = gcp('nocreate');
if isempty(pool), pool = parpool('local',workers); end
assert(pool.NumWorkers == workers, 'Existing pool has a different worker count.');
report.poolSetupSeconds = toc(t);
report.matlabVersion = version;
report.workers = pool.NumWorkers;
report.inputFile = inputFile;
report.frames = numel(points);
report.points = sum(cellfun(@(x) size(x,1),points));
report.warmup = warmup;
args = {'MaxLinkingDistance',8,'MaxGapClosing',1,'Debug',false};
methods = {@() simpletracker(points,args{:}), ...
    @() simpletracker_fast(points,args{:},'UseParallel',true)};
report.methodNames = {'original_parallel','optimized_parallel'};
if includeSerial
    methods{end+1} = @() simpletracker_fast(points,args{:},'UseParallel',false);
    report.methodNames{end+1} = 'optimized_serial';
end
report.seconds = zeros(repeats,numel(methods));
for r = (1-double(warmup)):repeats
    order = 1:numel(methods);
    if r > 1, order = circshift(order,[0,mod(r-1,numel(methods))]); end
    for m = order
        fprintf('Benchmark %d frames, round %d, %s starting\n',numel(points),r,report.methodNames{m});
        t = tic;
        [x,y,X] = methods{m}();
        elapsed = toc(t);
        if r == 1-double(warmup) && m == 1
            a=x; b=y; A=X;
        end
        assert(isequaln(a,x) && isequaln(b,y) && isequaln(A,X),'Tracking output mismatch.');
        if r > 0, report.seconds(r,m) = elapsed; end
        fprintf('Finished in %.3f s; exact output match\n',elapsed);
    end
end
report.medianSeconds = median(report.seconds,1);
report.speedupVsOriginal = report.medianSeconds(1)./report.medianSeconds;
report.exactMatch = true;
disp(report);
end
