function results = test_tracking_equivalence(useParallel)
% Exact comparisons, including IDs/order/NaNs/sparse links, not just counts.
if nargin < 1, useParallel = false; end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'baseline'),fullfile(root,'tests'));
rng(42);
cases = { ...
    {zeros(0,2),zeros(0,2),zeros(0,2)}, ...
    {[1 1;2 2]}, ...
    {[0 0;0 0;1 1],[0 0;0 0;1 1],[0 0;1 1]}, ...
    {[0 0],zeros(0,2),[1 0],zeros(0,2),[2 0]}, ...
    {[0 0;100 100],[8 0;108+eps(108) 100],[16 0]}, ...
    {[0 0;2 0],[1 1;1 -1],[0 0;2 0]} };
for trial = 1:12
    points = cell(1,20);
    for frame = 1:20
        points{frame} = round(rand(randi([0 25]),2)*25)/2;
    end
    cases{end+1} = points;
end
methods = {'Hungarian','NearestNeighbor'};
n = 0;
for c = 1:numel(cases)
    for orientation = 1:2
        points = cases{c};
        if orientation == 2, points = points.'; end
        for gap = [1 2 3]
            for m = 1:numel(methods)
                args = {'MaxLinkingDistance',8,'MaxGapClosing',gap, ...
                    'Method',methods{m},'Debug',false};
                [a,b,A] = simpletracker_serial_reference(points,args{:});
                [x,y,X] = simpletracker_fast(points,args{:},'UseParallel',useParallel);
                assert(isequaln(a,x) && isequaln(b,y) && isequaln(A,X), ...
                    'Mismatch: case %d, orientation %d, gap %d, method %s', ...
                    c,orientation,gap,methods{m});
                n = n + 1;
            end
        end
    end
end
results = struct('comparisons',n,'passed',true,'parallel',useParallel);
disp(results);
end
