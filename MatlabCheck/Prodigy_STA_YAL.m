clear
close all


str = './IdealPSF/IdealPSF_full_Frame';
load([str, '1_EG1.mat'], 'EG_para', 'Overall_para', 'Event_para_hdr');

beamspacing = 2;
upsampling = 8;
Nelements = Overall_para.Ele_num;
f0 = EG_para.Freq*1e6;
fs = upsampling*Overall_para.fs*1e6;
soundv = Overall_para.soundv;
dz0 = soundv/fs/2;
f_num = 2;
FOV   = 1*Nelements; % field of view, the x-direction size of beamformed RF data.
iter = Overall_para.Frames;
Noffset = Event_para_hdr.Toffset;
Noffset = upsampling*round(Noffset*fs);
N_dep_min = EG_para.MinDepth_sample;
N_dep = EG_para.MaxDepth_sample;
lambda_in_sample = fs/f0;   
Nsample = N_dep-N_dep_min+1;
pitch = Overall_para.Pitch;

bp = fir1(48, [0.1, 0.9]).';
clear EG_para Overall_para Event_para_hdr;
full_dataset = zeros(Nsample, Nelements, Nelements);

for idx = 1:iter
    load([str,int2str(idx),'_EG1.mat'], 'EG1');
    tmp = double(EG1);
    for idx1 = 1:128
        for idx2 = 1:size(EG1, 3)
            tmp(:, :, idx2, idx1) = conv2(tmp(:, :, idx2, idx1), bp, 'same');
        end
    end
    tmp = double(reshape(mean(tmp, 3), [size(tmp, 1), size(tmp, 2), size(tmp, 4)]));

    full_dataset = full_dataset+tmp(1:Nsample, :, :);

    clear EG1;
end
tmp = full_dataset/iter;
[xx, zz] = meshgrid(1:Nelements,1:Nsample);
[xq, zq] = meshgrid(1:Nelements,linspace(1,Nsample,upsampling*Nsample));
full_dataset = zeros(upsampling*Nsample,Nelements,Nelements);
for ii = 1:Nelements
    full_dataset(:,:,ii) = interp2(xx,zz,tmp(:,:,ii),xq,zq,'cubic');
end
Nsample = size(full_dataset,1);
%% Delay Profile
x = randn(1, Nelements);
bt = 0.5;
span = 8;
sps = 16;
h = gaussdesign(bt, span, sps);
delay_curve_in_time = conv(x, h, 'same');
delay_curve_in_time = (delay_curve_in_time - min(delay_curve_in_time));
delay_curve_in_time = delay_curve_in_time / max(delay_curve_in_time);
delay_curve_in_time = delay_curve_in_time - 0.5;
figure;
plot(delay_curve_in_time);
title('delay curve');

% STA

% ------ prepare for beamformed RF data ------ 
% Coordinate setup
% x_elem: x location at each element's center, length = Nelements
% x_range/x_aline: x location at each aline's center in beam buffer (beam space), length = beamspacing*Nelements

x_elem   = (-(Nelements-1)/2:(Nelements-1)/2)*pitch; % element central location
x_range  = ((FOV-1)/2 + 1/beamspacing*(beamspacing-1)/2)*pitch;
x_aline  = -x_range:pitch/beamspacing:x_range; % aline central location
% 3D element's x location, dimension = (copy the i-th element's location (Nsamples), copy the i-th element's location (FOV including beamspacing), i-th element location)
xx_elem  = repelem(reshape(x_elem, 1,1,[]), Nsample, FOV*beamspacing, 1); % size = (Nsample, FOV*beamspacing, Nelements)
% 3D element's x location, dimension = (copy the i-th aline's location (Nsamples), i-th aline location, copy the i-th aline's location (Nelements))
xx_aline = repmat(reshape(x_aline, [1,FOV*beamspacing,1]), [Nsample, 1, Nelements]); % size = (Nsample, FOV*beamspacing, Nelements)
z        = dz0*(Noffset + (0:Nsample-1)'); % depth axis, size = (Nsample,1)
% delay: the distance btw the i-th element to all aline
% xx_aline - xx_elem: x-direction distance btw the i-th element and the n-th aline, (distance at different depth, distance btw n-th aline to the i-th element, the i-th element location)
% abs(X+1j*Z): a way to calculate the Euclidean distance d(x,z) = =....
distance    = abs(xx_aline - xx_elem + 1j*repmat(z,[1,FOV*beamspacing, Nelements])); % size = (Nsample, # of aline, # of element), unit in "distance" instead of time or sample.
% in order to convert the array index to general index
% e.g. value at (2,2,2) in a 3*3*3 matrix, its general index is 2*(3*3*3) + 1*3 + 2 = 59, i.e. the 59-th element in this array
index3D  = repelem((0:Nelements-1)*(Nsample+1), Nsample,1,Nelements) + ... % 2-nd dim index
           repelem(reshape((Nsample+1)*Nelements*(0:Nelements-1), 1,1,[]), Nsample, Nelements, 1); % 3-rd dim index
% ------ Apply delay profile ------
maxdelay = [0, 1, 1.5, 2]/4; % (*pi)
k = 1;
% for k = 1:4
    aberratedchanneldata  = zeros(Nsample+1,Nelements,Nelements);
    delay_curve_in_sample = round(delay_curve_in_time * maxdelay(k) * lambda_in_sample); % delay how many samples
    for itx = 1:Nelements 
        % full_dataset(:, :, itx): one channel data, obtained by the itx-th transmitting element and all received Nelements.
        % delay_curve1(itx): delay applied to only one tx element
        % delay_curve1: delay applied to all Nelements
        aberratedchanneldata(1:end-1, :, itx) = Apply_Delay(full_dataset(:, :, itx), delay_curve_in_sample+delay_curve_in_sample(itx)); 
    end
    % ------ STA beamforming ------
    RFdata = zeros(Nsample, FOV);
    for Nline = 1:beamspacing*FOV % beamform an aline in one time
        Nline
        % f # mask: 2* means abs(x_aline(Nline) - x_elem) only consider one side of aperture
        % f_num_mask_rx: where the aperture larger than f # (how many tx elements used, (aperture for different depth, which rx element used, copy all rx element's mask along tx element)
        % f_num_mask_tx: where the aperture larger than f # (how many rx elements used, (aperture for different depth, copy the i-th tx element's mask along rx element, which tx element used)
        f_num_mask_rx = double(z./(2*abs(x_aline(Nline) - x_elem)) > f_num); % size = (Nsample, Nelements) = (Nsample, Nelements, 1)
        f_num_mask_tx = reshape(f_num_mask_rx, Nsample, 1, Nelements);      % size = (Nsample, 1, Nelements)
        Nlinedistance   = distance(:,Nline,:) + squeeze(distance(:,Nline,:));         % distance btw the Nline-th aline and tx + rx
        % delay(:,Nline,:): tx delay, (delay at different depth, copy the same delay for each tx element, delay btw the Nline-th aline and i-th element)
        % squeeze(delay(:,Nline,:)): rx delay, (delay at different depth, delay btw all aline and i-th element, copy the same delay for each tx element)
        channel_ind = ceil(Nlinedistance/soundv*fs - Noffset); % convert delay in time to in sample
        channel_ind(channel_ind > Nsample) = Nsample + 1; % limit delay index
        channel_ind(channel_ind < 1) = 1; % limit delay index
        channel_ind = channel_ind + index3D; % convert the array index to general index
        % sum the channel data from the K tx elements and K rx elements,where K is related to the f # mask.
        % dimension means (Nsample, channel data from rx element, channel data from tx element)
        RFdata(:,Nline) = sum(f_num_mask_rx.*f_num_mask_tx.*aberratedchanneldata(channel_ind), [2,3]); 
    end
    lpf = fir1(48, 0.8*f0/(fs/2))'; % cutoff freq. at 0.8*(bw*f0) Hz
    BBdata = conv2(RFdata.*exp(-1j*2*pi*f0*(0:size(RFdata, 1)-1)'/fs), lpf, 'same');
    % -------- PSF Preprocessing ---------
    envelope = abs(BBdata);
    envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);
    gain = 60;
    DR = 60;
    figure;
    image(x_aline*1e3, z*1e3, envelope_dB+gain);
    colormap(gray(DR));colorbar;
    axis image;

toc