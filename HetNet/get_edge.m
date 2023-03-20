clc
close all
clear all
loaddir = "./datasets/MSD/train/mask/";
savedir = "./datasets/MSD/train/edge/";
test = 0;

fig = uifigure;
d = uiprogressdlg(fig, 'Title', 'Processing Edge Maps');

imgs = dir(fullfile(loaddir, '*.png'));
for i = 1:numel(imgs);
    filename = imgs(i).name;
    %sprintf('%s', filename)
    temp_filename = strcat(loaddir,filename);
    gt = imread(temp_filename);
    gt = (gt > 0);
    gt = double(gt);
    %figure('Name', 'gt > 0')
    %imshow(gt)
    [gy, gx] = gradient(gt);
    temp_edge = gy.*gy + gx.*gx;
    temp_edge(temp_edge~=0)=1;
    bound = uint8(temp_edge*255);
    output_path = fullfile(savedir, filename);
    imwrite(bound, output_path);
    d.Value = i/numel(imgs);
    %figure('Name', 'bound')
    %imshow(bound)
end
    