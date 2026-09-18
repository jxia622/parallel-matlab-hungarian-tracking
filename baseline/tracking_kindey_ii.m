addpath(genpath("C:\Users\BAS448\Desktop\Research\SimpleTracker"));

for ii = 1:18 %:26
    tic
    fname =  fullfile('G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking',['bloc_dias' num2str(ii) '_track']);
    load(fname);

    max_linking_distance=8;
    max_gap_closing=1;
    [tracks,adjacency_tracks] = simpletracker(SR_Localizations,...
    'MaxLinkingDistance', max_linking_distance, ...
    'MaxGapClosing', max_gap_closing, ...
    'Debug', true);
    
    filename_save = fullfile('G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking',['bloc_dias' num2str(ii) '_tracked.mat']);

    save(filename_save,'adjacency_tracks','SR_Localizations');
    clear SR_Localizations]
    disp(['bloc' num2str(ii) 'done']);
end