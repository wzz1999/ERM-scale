
%% number of ROI
nROI = zeros(6,1);
i = 0;
for isubject = [{'201106'},{'201116'},{'201117'},{'201125'},{'201126'},{'210106'}]
    i = i + 1;
    dir_nut = getFishPath('activity',isubject{1});
    load([dir_nut,'\Judge2.mat'],'Judge2');
    nROI(i) = sum(Judge2);
    if strcmp(isubject, '201125')
        nROI(i) = nROI(i) - 2;
    end
end
mean(nROI)
std(nROI)

%% recording time
recording_time = zeros(6,1);
i = 0;
for isubject = [{'201106'},{'201116'},{'201117'},{'201125'},{'201126'},{'210106'}]
    i = i + 1;
    dir_nut = getFishPath('activity',isubject{1});
    load([dir_nut,'\sequences.mat']);
    recording_time(i) = sequences(end,end);
end

mean(recording_time)/10/60
std(recording_time)/10/60

%% ROI size 
% data generated from C:\Users\Public\code\Fish-Brain-Behavior-Analysis\code\data_analysis_Weihao\spike_inference\ROI_size_compute.m
ROI_size_all = [];
i = 0;
for isubject = [{'201106'},{'201116'},{'201117'},{'201125'},{'201126'},{'210106'}]
    data_dir = getFishPath('activity',isubject{1});
    ROI_size = load(fullfile(data_dir, 'ROI_size.mat'), 'ROI_size');
    ROI_size_all = [ROI_size_all; ROI_size.ROI_size];
end

mean(ROI_size_all*2) % 1 voxel is 1*1*2 \mu m^3

std(ROI_size_all*2)

%% ROI diameter
diameter_all = [];
for isubject = [{'201106'},{'201116'},{'201117'},{'201125'},{'201126'},{'210106'}]
    data_dir = getFishPath('activity',isubject{1});
    load(fullfile(data_dir, 'diameter_ROI.mat'), 'diameter');
    diameter_fish = mean(diameter);
    diameter_all = [diameter_all; diameter];
end

mean(diameter_all*2) % 1 voxel is 1*1*2 \mu m^3

std(diameter_all*2)

%% area coverage
ifish = '201126';
standard_load_fish;
nc_area = zeros(58,1);
for i = 1:58
    i_name_area = anatomy{i,2};
    ROI_ls = area_ROI_extract(areaName, i_name_area);
    nc_area(i) = length(ROI_ls);
end

sum(nc_area > 2)/58
results = [0.6552, 0.5690, 0.7931, 0.7586, 0.6552];
mean_results = mean(results)
std_results = std(results)
