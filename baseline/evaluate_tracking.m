clc;
clear;

size_z = 800;
size_x = 1033;
min_length = 15;
ii = 1;
for i = 1:18
    filename =  fullfile('G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking',['bloc_dias' num2str(i) '_tracked']);
    load(filename);
    n_tracks = numel(adjacency_tracks);
    all_points = vertcat(SR_Localizations{:});
    size(all_points,1);
    for i_track = 1 : n_tracks  
        tmp = adjacency_tracks{i_track};
        if length(tmp)>min_length
            Tracks{ii} = all_points(tmp, :);  
            ii=ii+1;
        end
    end
end

% % Apply Kalman Filter
for nn = 1:length(Tracks)
    Traj_raw = Tracks{nn};
    Tracks_post{nn} = Kalman_func(Traj_raw);
end
% Interpolation of the tracks
n_tracks = numel(Tracks);
for i_track=1:n_tracks
    z_track=Tracks_post{i_track}(:,1);
    x_track=Tracks_post{i_track}(:,2);    
    z_track_interp = interp1(1:length(z_track),z_track,1:0.1:length(z_track));
    x_track_interp = interp1(1:length(x_track),x_track,1:0.1:length(x_track));
    x_final=round(x_track_interp(1:end));
    z_final=round(z_track_interp(1:end));    
    [~,ixu]=unique(x_final);[~,izu]=unique(z_final);
    ifin=union(izu,ixu);
    Tracks_Final{i_track}(:,1)=z_final(ifin);
    Tracks_Final{i_track}(:,2)=x_final(ifin);
end

FinalPoints=cell2mat(Tracks_Final');
%FinalPoints=cell2mat(Tracks');
%FinalPoints = round(FinalPoints);
FinalPoints(FinalPoints<1) = 1;
for nn = 1:size(FinalPoints,1)
    if FinalPoints(nn,1)>800
        FinalPoints(nn,1) = 800;
    end
    if FinalPoints(nn,2)>1033
        FinalPoints(nn,2) = 1033;
    end
end
ind=sub2ind([size_z, size_x],FinalPoints(:,1),FinalPoints(:,2));
Imfinal=zeros([size_z, size_x]);
for pp=1:length(ind)
    Imfinal(ind(pp))=Imfinal(ind(pp))+1;
end

figure;
imagesc(Imfinal.^0.5); clim([0,8])
colormap(hot); colorbar; axis image;

save("G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking\section_1.mat","Imfinal");

%% Velocity Map
VelocityMap_final = zeros(800,1024);
V_index = zeros(800,1024);
pixel_value = 1540/15.625/1e6/8*1e3; %mm
n_tracks = numel(Tracks);
nn = 1;
for i_track=1:n_tracks
    z_track=Tracks_post{i_track}(:,1);
    x_track=Tracks_post{i_track}(:,2);
    velocity = zeros(length(z_track)-1,1);
    for j = 1:length(z_track)-1
        %calculation velocity
        line1=z_track(j+1) - z_track(j);
        line2=x_track(j+1) - x_track(j);
        flow_direction=z_track(j+1)-z_track(j);
        dis=sqrt(line1^2+line2^2) * pixel_value;
        velocity(j)=dis/(1/500)*sign(-flow_direction);
        %velocity(j)=dis/(1/400)*sign(-flow_direction);
    end
    z_track_interp = interp1(1:length(z_track),z_track,1:0.1:length(z_track));
    x_track_interp = interp1(1:length(x_track),x_track,1:0.1:length(x_track));
    x_final=round(x_track_interp(1:end));
    z_final=round(z_track_interp(1:end));    
    [~,ixu]=unique(x_final);[~,izu]=unique(z_final);
    ifin=union(izu,ixu);
    for j = 1:length(ifin)
        if x_final(ifin(j))<1 
            x_final(ifin(j)) = 1;
        end
        if z_final(ifin(j))> 800 
            z_final(ifin(j)) = 800;
        end
        if x_final(ifin(j))>1024
            x_final(ifin(j)) = 1024;
        end
        VelocityMap_final(z_final(ifin(j)),x_final(ifin(j))) = velocity(floor(abs((ifin(j)-2))/10)+1);
        %VelocityMap(z_final(ifin(j)),x_final(ifin(j))) = VelocityMap(z_final(ifin(j)),x_final(ifin(j))) + velocity(floor(abs((ifin(j)-2))/10)+1);
        %V_index(z_final(ifin(j)),x_final(ifin(j))) = V_index(z_final(ifin(j)),x_final(ifin(j))) + 1;
    end
    %i_track
end

%VelocityMap_final = VelocityMap ./ V_index;
load('G:\20260602_Mouse1_SSMH_9974/MyColormap');
VelocityMap_final(isnan(VelocityMap_final))=0;
figure
imagesc(VelocityMap_final)
colormap(cmap); colorbar;
clim([-25,25])
axis image,axis off

save('G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking\section_1vmap.mat',"VelocityMap_final");

%% original coordinate
%load('H:\Kidney Data\20250807_Mouse2_SSMH+AAT_9701\1_01ml\param.mat');
clc;
clear;
%close all;

bmat= matfile('G:\20250807_Mouse4_SSMH+AAT_9704\results\20250807_Mouse4_SSMH+AAT_9704_05percent_3_01ml_IQ.mat');
%mat= matfile('H:\Kidney Data\20250807_Mouse3_SSMH+AAT_9703\1_01ml\20250807_Mouse2_SSMH+AAT_9701_01ml_05percent_3_01ml_IQ.mat');
%load('20250219_mouse 3_AA_9491_1_2khz_IQ.mat');
load('G:\20260602_Mouse1_SSMH_9974\2_0.5per_0.1ml\param.mat');
load('G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking\section_1.mat');
load('G:\20250807_Mouse4_SSMH+AAT_9704\results\3_01ml\results - tracking\section_1vmap.mat');

SoundSpeed = 1540;
dataDepth = 800;
fs = Receive(1).decimSampleRate;    % In MHz
lat = Trans.ElementPos(:,1)'*(SoundSpeed*1e3)/(Trans.frequency*1e6); 
axial = ((1:(dataDepth))*SoundSpeed/(fs*1e6)/2+Receive(1).startDepth/Trans.frequency*SoundSpeed/1e6)*1e3; %mm % US Pulse-echo case
 
m = axial(1):0.0123:axial(800);
n = lat(1):0.0123:lat(end);

%overlay SR on B-Mode
figure(8);
ax1 = axes;
image_filtered = imagesc(n,m,Imfinal.^0.5);
colormap(ax1,hot);axis image
set(image_filtered,'AlphaData',1);
hold on
figure(8);
ax2=axes;
bmode=abs(bmat.BmodeData(1:800,:,200));
image_bmode=imagesc(lat,axial,20*log10(bmode/max(bmode(:))),[-40 0]);
colormap(ax2,gray);axis image;
set(image_bmode,'AlphaData',0.4);
set(ax2,'visible','off');

%overlay velocity on B-Mode
figure(9)
ax4=axes;
load('G:\20260602_Mouse1_SSMH_9974/MyColormap');
VelocityMap_final(isnan(VelocityMap_final))=0;
vmap_filtered= imagesc(n,m,VelocityMap_final); colormap(ax4,cmap); axis image; %colorbar;
clim([-25,25])
set(vmap_filtered,'AlphaData',1);
hold on;
figure(9);
ax3=axes;
bmode=abs(bmat.BmodeData(1:800,:,200));
image_bmode=imagesc(lat,axial,20*log10(bmode/max(bmode(:))),[-40 0]);
colormap(ax3,gray);axis image;
set(image_bmode,'AlphaData',0.4);
set(ax3,'visible','off');
