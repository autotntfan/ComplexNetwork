addpath('~/WORK/Field_II/');
clearvars -except n;close all;clc;
n = 1;
for iid = [1:16:1600]
    id = iid + (n-1)*1600
    BF_Sim_Large;
end
