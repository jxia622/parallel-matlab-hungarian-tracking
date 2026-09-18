clc;
clear;
%close all;
mat= matfile('G:\20260603_Mouse5_SSMH+AAT_9978\results\3_0.5per_01ml\20260603_Mouse5_SSMH+AAT_9978_3_0.5per_0.1ml_IQ.mat');
%load('20250219_mouse 3_AA_9491_1_2khz_IQ.mat');
load('G:\20260603_Mouse5_SSMH+AAT_9978\3_0.5per_0.1ml\param.mat');

SoundSpeed = 1540;%Resource.Parameters.speedOfSound;
%dataDepth = size(BmodeData,1);
dataDepth = 800;
fs = Receive(1).decimSampleRate;    % In MHz
lat = Trans.ElementPos(:,1)'*(SoundSpeed*1e3)/(Trans.frequency*1e6); % Center 128 channel
axial = ((1:(dataDepth))*SoundSpeed/(fs*1e6)/2+Receive(1).startDepth/Trans.frequency*SoundSpeed/1e6)*1e3;

%% Consolidating chosen frames without motion
stableFrames= [185,684,685,1184,1185,1684,1685,2184,2185,2684,2685,3184,3185,3684,3685,4184,4185,4584,4585,4690,5010,5509,5510,5690,5995,6494,6495,6994,6995,7494,7495,7994,7995,8494,8495,8920,9255,9754,9755,10000];

%9978
%1_01- [1,500,501,1000,1001,1500,1501,2000,2001,2500,2501,2710,3015,3330,3595,4094,4095,4594,4595,5094,5095,5225,5505,6004,6005,6504,6505,7004,7005,7504,7505,8004,8005,8504,8505,9004,9005,9504,9505,9610,9890,10000];

%9975 
% 2_01- [80,579,580,1079,1080,1579,1580,1865,2185,2684,2685,3184,3185,3684,3685,3910,4195,4694,4695,5194,5195,5694,5695,6194,6195,6694,6695,6810,7105,7604,7605,7750,8060,8559,8560,9059,9060,9559,9560,10000];

% 9977
% 2_01- [1,250,495,994,995,1210,1470,1969,1970,2380,2670,3105,3360,3759,3760,3990,4270,4770,5035,5534,5535,5935,6180,6679,6680,7010,7265,7764,7765,7970,8225,8660,8890,9289,9290,9575,9835,10000];
% 3_01- [1,435,705,1175,1430,1929,1930,2329,2330,2460,2735,3230,3470,3969,3970,4395,4625,5024,5025,5190,5445,5890,6130,6629,6630,6970,7245,7644,7645,7775,8000,8385,8640,9139,9140,9450,9695,10000];

%[1,180,400,805,1015,1464,1465,1740,1950,2349,2350,2660,2895,3110,3345,3744,3745,4130,4355,4754,4755,5110,5330,5815,6035,6434,6435,6665,6875,7350,7545,7944,7945,8220,8445,8820,9040,9290,9510,9965];
%SSMH+AAT- 
%1_01 - [1,355,540,939,940,1115,1315,1675,1850,2350,2545,2950,3155,3200,3400,3740,3940,4210,4435,4910,5120,5519,5520,5690,5900,6395,6595,7025,7230,7470,7660,8059,8060,8170,8380,8800,9020,9510,9720,9970];
%3_01 - [1,70,290,540,725,985,1195,1660,1855,2215,2420,2660,2845,3200,3385,3740,3925,4165,4370,4470,4665,4920,5140,5365,5770,6180,6360,6695,6935,7165,7345,7605,7820,8075,8260,8530,8715,8985,9170,9360,9520,9785];
%2_01 - [210,515,800,1065,1315,1570,1805,2230,2475,2660,2880,3165,3360,3855,4245,4395,4615,4885,5105,5425,5623,5960,6165,6530,6715,6835,7075,7495,7735,7935,8130,8430,8700,8955,9185,9655,9870,10000];

No_blocs = length(stableFrames)/2;
no_frames = 0;
for i = 1:No_blocs
    no_frames = no_frames + stableFrames(i*2)-stableFrames(i*2-1)+1;
    frames(i) = stableFrames(i*2)-stableFrames(i*2-1)+1;
end

for i = 1:No_blocs
    Beamformed_IQ = mat.BmodeData(1:800,:,stableFrames(i*2-1):stableFrames(i*2));
    fname =  fullfile('G:\20260603_Mouse5_SSMH+AAT_9978\results\3_0.5per_01ml',['bloc' num2str(i) '_bf']);
    save(fname,'Beamformed_IQ');
end

%% Motion Correction and SVD filtering
disp('Motion Correction and SVD filtering')
[optimizer,metric]=imregconfig('monomodal');
registration.optimizer.MaximumIterations = 200;
registration.optimizer.MaximumStepLength = 0.001;
for i=20
    fname =  fullfile('G:\20260603_Mouse5_SSMH+AAT_9978\results\3_0.5per_01ml',['bloc' num2str(i) '_bf']);
    load(fname);
    
    [size_z,size_x,N_frames]=size(Beamformed_IQ); 
    Casorati = reshape(Beamformed_IQ,[size_z*size_x,N_frames]); 
    [U,S,V]=svd(Casorati,'econ');
    enS = diag(S);
    %figure; plot(20*log10(enS)); drawnow
    
    % %perform motion correction
    % Cutoff=20;
    % S([Cutoff:end],[Cutoff:end])=0;
    % Casorati_f=U*S*V';
    % Bmode_tissue=reshape(Casorati_f,[size_z,size_x,N_frames]);
    % %a=abs(Bmode_tissue(1:800,:,i));
    % % figure;
    % % imagesc(20*log10(a/max(a(:))),[-40 0]); colormap(gray); colorbar;
    % %Bmode_samp = Bmode_tissue(200:400,45:85,:);
    % Bmode_samp = Bmode_tissue(183:302,63:87,:);
    % %Bmode_samp = Bmode_tissue(305:585, 37:85,:);
    % %this is kinda like the ROI sample instead of the whole image used as a reference
    % %110:700: These are the depth (Z) rows. You should move these to where the kidney looks clearest (e.g., 300:500).
    % %15:115: These are the lateral (X) columns. You want this to center on the kidney's "meat
    % %an important point - this doesn't work well if you take the whole
    % %image - for kidney, focus on the center which will change during
    % %motion
    % 
    % for ii=1:size(Bmode_samp,3)
    %    tform=imregtform(abs(Bmode_samp(:,:,ii)),abs(Bmode_samp(:,:,110)),'translation',optimizer,metric);
    %    X(ii)=tform.T(3,1);
    %    Z(ii)=tform.T(3,2);
    %    Beamformed_IQ_new(:,:,ii)=imwarp(Beamformed_IQ(:,:,ii),tform,'OutputView',imref2d([size(Bmode_tissue,1) size(Bmode_tissue,2)]));
    % end

    % %perform SVD
    % [size_z,size_x,N_frames]=size(Beamformed_IQ_new); 
    % Casorati = reshape(Beamformed_IQ_new,[size_z*size_x, N_frames]); 
    % [U,S,V]=svd(Casorati,'econ');
    % % enS = diag(S);
    % % figure; plot(20*log10(enS)); drawnow

    for k=1:size(U,2)-1
        spatial_corr(k)= corr(abs(U(:,k)),abs(U(:,k+1)));
    end
    y= movmean(spatial_corr,10);

    Cuts= [12,14,17,19];
    %Cutoff2 = find(y < 0.66, 1, 'first'); %observe the spatial correlation graph and see where it sharply drops - you can also use the spatial vectors to make a decision 
    for j=4
        Cutoff1=Cuts(j);
        S([1:Cutoff1],[1:Cutoff1])=0;
        %S([Cutoff2:end],[Cutoff2:end])=0;
        Casorati_f=U*S*V';
        FilteredData=reshape(Casorati_f,[size_z,size_x,N_frames]);
        
        figure; imagesc(mean(abs(FilteredData),3)); colormap hot
        title(['SVD - ',num2str(Cutoff1)]);
        
        fname= fullfile('G:\20260603_Mouse5_SSMH+AAT_9978\results\3_0.5per_01ml',['bloc' num2str(i) '_filtered']);
        %save(fname,'FilteredData');    
        disp(['bloc' num2str(i) 'done']);
    end
    clear U S V spatial_corr Casorati Casorati_f slope y dy d2y Bmode_samp Bmode_tissue Beamformed_IQ Beamformed_IQ_new;
end

%% Localization
addpath(genpath("D:\The Loser's Hub\PhD\MATLAB\Research\GRU-MT\SimpleTracker"));
disp('Microbubble Localization and Tracking')

Threshold=0.05; %0.08;
m=linspace(1,800,800);
n0=linspace(1,128,128);
n=linspace(1,128,1024);
addpath(genpath('SimpleTracker'))

%No_blocs=21;

for i=21
    tic
    fname =  fullfile('G:\20260603_Mouse5_SSMH+AAT_9978\results\1_0.5per_01ml',['bloc' num2str(i) '_filtered']);
    load(fname);
    for ii = 1:size(FilteredData,3)
        prediction(:,:,ii) = abs(interp2(n0,m',abs(FilteredData(:,:,ii)),n,m','spline'));
    end
    %clear FilteredData
    [size_z,size_x,N_frames]=size(prediction(:,:,:)); 
    [Pix_X,Pix_Z]=meshgrid([-2:2],[-2:2]);
    
    % if ismember(i,[2,4,5,6,7,8,10,11,12,14,16,18])
    %     Threshold= 0.07;
    % end

    for nn=1:N_frames
        Im=squeeze(prediction(:,:,nn));
        Im = Im/max(Im(:));
        Reg_max=Im.*imregionalmax(Im);
        Reg_max(Reg_max<Threshold)=0;
        [Ind_z,Ind_x]=ind2sub(size(Reg_max),find(Reg_max));    
        kk = 1;
        for i_loc=1:length(Ind_z)
            Z_init=Ind_z(i_loc);
            X_init=Ind_x(i_loc);
            if Z_init>2 && Z_init<799 && X_init>2 && X_init<1023
                Im_MB=Im(Z_init-2:Z_init+2,X_init-2:X_init+2);
                dz=sum(sum(Im_MB.*Pix_Z))/sum(sum(Im_MB));
                dx=sum(sum(Im_MB.*Pix_X))/sum(sum(Im_MB));
                Z_waverage=Z_init+dz+0.5;
                X_waverage=X_init+dx+0.5;
                SR_Localizations{nn}(kk,:)=[Z_waverage X_waverage];
                kk = kk+1;
            end
        end
    end


    Points=cell2mat(SR_Localizations');
    ind=sub2ind([size_z,size_x],round(Points(:,1)),round(Points(:,2)));
    Imfinal=zeros([size_z, size_x]);
    for pp=1:length(ind)
        Imfinal(ind(pp))=Imfinal(ind(pp))+1;
    end

    figure
    imagesc(Imfinal)
    colormap gray
    axis image,axis off
    caxis([0 2])
    %drawnow;
    
    % filename_save = fullfile('G:\20260603_Mouse5_SSMH+AAT_9978\results\1_0.5per_01ml',['bloc_dias' num2str(i) '_track.mat']);
    % save(filename_save,"SR_Localizations");

    % Tracking
    % max_linking_distance=8;
    % max_gap_closing=1;
    % [tracks,adjacency_tracks] = simpletracker(SR_Localizations,...
    % 'MaxLinkingDistance', max_linking_distance, ...
    % 'MaxGapClosing', max_gap_closing, ...
    % 'Debug', false);
    % 
    % filename_save = fullfile('G:\20260602_Mouse1_SSMH_9974\results\1_3per_01ml\0.07',['bloc' num2str(i) '_track.mat']);
    % save(filename_save,'adjacency_tracks','SR_Localizations');
    % clear SR_Localizations prediction
    disp(['bloc' num2str(i) 'done']);
    clear SR_Localizations prediction
    toc
end

%%
for i=1:15
    fname =  fullfile('H:\Kidney Data\20250807_Mouse3_SSMH+AAT_9703\2_01ml\results',['bloc_dias' num2str(i) '_track']);
    load(fname);

    Points=cell2mat(SR_Localizations');
    ind=sub2ind([size_z,size_x],round(Points(:,1)),round(Points(:,2)));
    Imfinal=zeros([size_z, size_x]);
    for pp=1:length(ind)
        Imfinal(ind(pp))=Imfinal(ind(pp))+1;
    end

    figure
    imagesc(Imfinal)
    colormap gray
    axis image,axis off
    caxis([0 2]);
    clear SR_Localizations
end