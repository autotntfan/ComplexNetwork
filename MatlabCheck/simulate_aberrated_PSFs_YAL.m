% Generate simulated RF/BB point spread function and its associated RF/BB speckle
% rng('shuffle');
addpath('./Field2');
savepath = './simulation_data1';
if ~exist(savepath, 'dir')
    mkdir(savepath)
end
% --------- parameters ---------
Npt         = 5;                    % number of simulated PSFs for one set
rngseed     = 7414;                 % random number generator seed
Npx         = 257;                  % final RF/baseband data size after resize for x direction
Npz         = 257;                  % final RF/baseband data size after resize for z direction
fs          = 100e6;                % sampling rate 100 MHz
soundv      = 1540;                 % sound velocity [m/s]
height      = 5e-3;                 % = 5 mm
pitch       = 3e-4;                 % = 0.3 mm
kerf        = 0;
Nelements   = 128;
FOV         = 0.5*Nelements;        % field of view, the x-direction size of beamformed RF data.
focus       = [0 0 1e3]/1000;       % initial electronic focus at 1000 mm
f_num       = 2;                    % f#
beamspacing = 4;                    % how many pixel in one beam (beam width)
gain        = 60;                   % image gain
DR          = 60;                   % dynamic range
psfsegx     = 32/2;                 % x-direction segmentation size, 16 lambda for left and right side from central point
psfsegz     = 16/2;                 % z-direction segmentation size, 8 lambda for top and bottom side from central point

rng(rngseed)

% --------- parameters ---------
f0     = (3 + 4.5*rand(1))*1e6; % center freq. 3~7.5 MHz
bw     = 0.5 + 0.3*rand(1);     % fractional bandwidth
lambda = soundv/f0;             % = [205.3,513.3] [um]
dz0    = soundv/fs/2;           % original depth (time) interval, i.e. dz, 2dz/soundv=1/fs
% lambda = soundv/f0; dt = lambda/soundv = 1/f0 (one-way wavelength);
% convert length to # of pixels　?　1/f0*fs = fs/f0
lambda_in_sample = fs/f0;   % lambda length represented in sample. [13.333 33.3333]
% ------ delay profile ------
% the phase aberration profile follows the near field phase screen model
% and assumes its correlation length to be 5mm.
x    = randn(1, Nelements); % Nelements delay
bt   = 0.5; % the 3-dB bandwidth-symbol time product
sps  = 16;  % samples per symbol
span = 8;   % filter spans 8 symbols
% total 8*16 = 128 + 1 sample
h    = gaussdesign(bt, span, sps);
delay_curve = conv(x, h, 'same');
% limit delay_curve in [-0.5,0.5]
delay_curve = (delay_curve - min(delay_curve));  % shift to all positive values
delay_curve = delay_curve / max(delay_curve);    % limited in [0,1]
delay_curve = delay_curve - 0.5;         % limited in [-0.5,0.5]
% ------ generate full dataset using Field II ------
field_init(0)
set_field('fs',fs);
set_field('c', soundv);
% 1 tx and all 128 rx in one time, total 128 elements so there are 128*128 acquisitions
% xdc_linear_array(# of element, width, height, kerf, # of sub x, # of sub y, focus)
Th  = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus); % pt to the transmitted aperture
Th2 = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus); % pt to the received aperture
tc  = gauspuls('cutoff', f0, bw, -6, -80);
tp  = -tc:1/fs:tc; % pulse time axis
% system impulse response
impulse_response = gauspuls(tp, f0, bw, -6);
impulse_response = impulse_response.*hanning(length(impulse_response))';
xdc_impulse(Th, impulse_response); 
xdc_impulse(Th2, impulse_response);
% ecitation pulse
excitation                = 10*sin(2*pi*f0*(0:1/fs:2/f0)); % a sine wave
excitation(excitation>1)  = 1;   % limit value under 1
excitation(excitation<-1) = -1;  % excitation range [-1,1]
excitation = excitation.*hanning(length(excitation))'; % like a rectangular wave
xdc_excitation(Th, excitation);
% point source location
ptloc      = 1/1e3;          % first point location: 1 mm
ptinterval = 9/1e3;          % adjacent point interval: 9 mm
shiftrange = ptinterval/3;   % 3 mm
positions  = zeros(2+Npt,3);
positions(1,3) = ptloc;
% positions  = [0 0 1;0 0 10;0 0 19;0 0 28;0 0 37;0 0 46;0 0 55]/1e3; % first position determines Noffset.
for ipt = 1:Npt+1
    positions(1+ipt,3) = ptloc + ipt*ptinterval;
end
zshift = 2*shiftrange*rand(1) - shiftrange; % randomly and simultaneously shift point source depth in the range [-3,3]
positions(2:end-1,3) = positions(2:end-1,3) + zshift;
amp      = ones(size(positions, 1), 1);
amp(1)   = eps;
amp(end) = eps;
% calc_scat_all (Th1, Th2, pts' position, pts' amplitudes, dec_factor)
[v, t]   = calc_scat_all(Th, Th2, positions, amp, 1); 
% v's dim = (Nsample, Nelements*Nelements), where Nsample is related to the sampling rate 
xdc_free(Th);
field_end
full_dataset = reshape(v, [], Nelements, Nelements); % size = (Nsample, received by the i-th element, emitted by the i-th element)
Noffset = round((t-2*tc)*fs); % t*fs: the time for the first sample in v (fieldII), round(2*tc*fs): half pulse length offset
Nsample = size(full_dataset, 1); % size(v, 1) depending on sampling rate

% ----- scat_dist: scatterer location (x,z) -----
% scatterer distribution
den         = 0.05 + 0.45*rand(1);                    % scatterer's density, [0.05,0.5]
Nscat       = round(den*Npz*Npx);                   % number of scatterers with in a patch
scat_space  = zeros(Npz,Npx);                       % distribution space size = (Npz,Npx)
scat_indice = randi(numel(scat_space),[1,Nscat]);   % scatterer's location at which index
scat_space(scat_indice) = randn(Nscat,1);           % scatterer's amplitude (random normal)
scat_space = reshape(scat_space, [Npz, Npx]);


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

for k = 1:4
    aberratedchanneldata  = zeros(Nsample+1,Nelements,Nelements);
    delay_curve_in_sample = round(delay_curve * maxdelay(k) * lambda_in_sample); % delay how many samples
    for itx = 1:Nelements 
        % full_dataset(:, :, itx): one channel data, obtained by the itx-th transmitting element and all received Nelements.
        % delay_curve1(itx): delay applied to only one tx element
        % delay_curve1: delay applied to all Nelements
        aberratedchanneldata(1:end-1, :, itx) = Apply_Delay(full_dataset(:, :, itx), delay_curve_in_sample+delay_curve_in_sample(itx)); 
    end
    % ------ STA beamforming ------
    RFdata = zeros(Nsample, FOV);
    for Nline = 1:beamspacing*FOV % beamform an aline in one time
        % f # mask: 2* means abs(x_aline(Nline) - x_elem) only consider one side of aperture
        % f_num_mask_rx: where the aperture larger than f # (how many tx elements used, (aperture for different depth, which rx element used, copy all rx element's mask along tx element)
        % f_num_mask_tx: where the aperture larger than f # (how many rx elements used, (aperture for different depth, copy the i-th tx element's mask along rx element, which tx element used)
        f_num_mask_rx = double(z./(2*abs(x_aline(Nline) - x_elem)) > f_num); % size = (Nsample, Nelements) = (Nsample, Nelements, 1)
        f_num_mask_tx = reshape(f_num_mask_rx, Nsample, 1, Nelements);      % size = (Nsample, 1, Nelements)
        Nlinedistance = distance(:,Nline,:) + squeeze(distance(:,Nline,:));         % distance btw the Nline-th aline and tx + rx
        % delay(:,Nline,:): tx delay, (delay at different depth, copy the same delay for each tx element, delay btw the Nline-th aline and i-th element)
        % squeeze(delay(:,Nline,:)): rx delay, (delay at different depth, delay btw all aline and i-th element, copy the same delay for each tx element)
        channel_ind = ceil(Nlinedistance/soundv*fs - Noffset); % convert delay in time to in sample
        channel_ind(channel_ind > Nsample) = Nsample + 1; % limit delay index
        channel_ind(channel_ind < 1) = Nsample + 1; % received signal before t=0
        channel_ind = channel_ind + index3D; % convert array index to general index
        % sum the channel data from the K tx elements and K rx elements,where K is related to the f # mask.
        % dimension means (Nsample, channel data from rx element, channel data from tx element)
        RFdata(:,Nline) = sum(f_num_mask_rx.*f_num_mask_tx.*aberratedchanneldata(channel_ind), [2,3]); 

    end
    lpf = fir1(48, f0/(fs/2))'; % cutoff freq. at 0.8*(bw*f0) Hz
    BBdata = conv2(RFdata.*exp(-1j*2*pi*f0*(0:size(RFdata, 1)-1)'/fs), lpf, 'same');
    % -------- PSF Preprocessing ---------
    envelope = abs(BBdata);
    envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);

    figure;
    image(x_aline*1e3, z*1e3, envelope_dB+gain);
    colormap(gray(DR));colorbar;
    xlabel('Lateral position (mm)')
    ylabel('Depth (mm)')
    axis image;

%     % check delay channel data
%     tmp = aberratedchanneldata(channel_ind);
%     figure
%     for ii = 1:size(tmp,2)
%         plot(z*1e3,1e-26*ii+tmp(:,ii,128)), hold on
%     end

    for ipt = 2:size(positions,1)-1 % start from the second to the next to last point location.
        [~, z_pt_start_ind] = min(abs(z - (positions(ipt,3) - psfsegz*lambda) )); % start index of a PSF region
        [~, z_pt_end_ind] = min(abs(z - (positions(ipt,3) + psfsegz*lambda) ));   % end index of a PSF region
        [~, x_pt_end_ind] = min(abs(x_aline - psfsegx*lambda)); % start index of a PSF region
        [~, x_pt_start_ind] = min(abs(x_aline + psfsegx*lambda)); % start index of a PSF region
        newz = interp1(z_pt_start_ind:z_pt_end_ind, z(z_pt_start_ind:z_pt_end_ind), linspace(z_pt_start_ind,z_pt_end_ind,Npz));
        newx = interp1(x_pt_start_ind:x_pt_end_ind, x_aline(x_pt_start_ind:x_pt_end_ind), linspace(x_pt_start_ind,x_pt_end_ind,Npx));
        depth = z(z_pt_start_ind);
        dx = newx(2) - newx(1);
        dz = newz(2) - newz(1);
        newfs = soundv/2/dz;
        lpf = fir1(48, f0/(newfs/2))';
        
        psf_rf = RFdata(z_pt_start_ind:z_pt_end_ind,x_pt_start_ind:x_pt_end_ind);
        psf_rf = imresize(psf_rf, [Npz,Npx], 'bilinear');
        psf_bb = conv2(psf_rf.*exp(-1j*2*pi*f0*(0:size(psf_rf,1)-1)'./newfs), lpf, 'same');
%         psf_bb = conv2(psf_rf.*exp(-1j*2*pi*f0*(2*newz'/soundv)), lpf, 'same');
        envelope = abs(psf_bb);
        envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);
        fig = figure('visible','off');
        subplot(121)
        image(newx*1e3, newz*1e3, envelope_dB+gain);
        colormap(gray(DR));colorbar;
        xlabel('Lateral position (mm)')
        ylabel('Depth (mm)')
        axis image;

        % produce speckle
        speckle_rf = conv2(scat_space, psf_rf, 'same'); % psf is a filter
        speckle_bb = conv2(speckle_rf.*exp(-1j*2*pi*f0*(0:size(speckle_rf,1)-1)'./newfs), lpf, 'same');
%         speckle_bb = conv2(speckle_rf.*exp(-1j*2*pi*f0*(2*newz'/soundv)), lpf, 'same');
        envelope = abs(speckle_bb);
        envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);

        subplot(122)
        image(newx*1e3, newz*1e3, envelope_dB+gain);
        colormap(gray(DR));colorbar;
        xlabel('Lateral position (mm)')
        ylabel('Depth (mm)')
        axis image;


        save(fullfile(savepath,['Data_', num2str(ipt), '_delay_', num2str(k) '.mat']), 'psf_rf', 'psf_bb', 'speckle_rf', 'speckle_bb', 'dx', 'dz', 'depth', 'f0', 'k', 'bw', 'delay_curve');
        saveas(fig, fullfile(savepath,['Data_', num2str(ipt), '_delay_', num2str(k), '.png']));

    end
end


