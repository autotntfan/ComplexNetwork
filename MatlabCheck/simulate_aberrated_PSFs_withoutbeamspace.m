tic
addpath('./Field2');
% --------- parameters ---------
fc = (3 + 4.5*rand(1))*1e6; % center freq. 3~7.5 MHz
fs = 100e6; % sampling rate 100 MHz
bw = 0.5 + 0.3*rand(1);
soundv = 1540; % sound velocity [m/s]
lambda = soundv/fc;  % = [205.3,513.3] [um]
height = 5e-3;       % = 5 mm
pitch = 3e-4;        % = 0.3 mm
kerf = 0;
Nelements = 128;
focus = [0 0 1e3]/1000;% initial electronic focus, 1000 mm 
f_num = 2;
dz_orig = soundv/fs/2; % depth interval, i.e. dz
% lambda = c/f0; dz = c/fs;
% 1 lambda (length to # of pixel) = fs/f0 samples;
T_sample = fs/fc; % [13.333 33.3333]
Upsample = 1;
beamspace = 1;

% PSF size
Kx_dec = 16; 
Kz_dec = 16;
Kx_range = 16;
Kz_range = 16;

z_len = round(2.1*Kz_range*(T_sample)); % 448 ~ 1120 mm
x_len = round(1.1*Kx_range*lambda/(pitch/4)); % 48 ~ 120 mm

% scatterer distribution
den = 0.05 + 0.45*rand(1); % density, 0.05 ~ 0.5
N = round(den*(2*Kz_range*Kz_dec+1)*(2*Kx_range*Kx_dec+1)); % number of scatterers, 13158 ~ 131580

% ----- scat_dist: scatterer location (x,z) -----
scat_dist = zeros((2*Kz_range*Kz_dec+1)*(2*Kx_range*Kx_dec+1),1); % distribution size (513,513)
scat_index = randi((2*Kz_range*Kz_dec+1)*(2*Kx_range*Kx_dec+1),[1,N]); % scatterer index for 2D matrix
scat_dist(scat_index) = randn(N,1); % scatterer amp
scat_dist = reshape(scat_dist,[2*Kz_range*Kz_dec+1,2*Kx_range*Kx_dec+1]);

% ------ delay profile ------
x = randn(1, Nelements);
bt = 0.5; % the 3-dB bandwidth-symbol time product
span = 8; % 
sps = 16; % total 8*16 = 128 + 1 sample
h = gaussdesign(bt, span, sps);
delay_curve = conv(x, h, 'same');
% limit delay_curve to be positive then [-0.5,0.5]
delay_curve = (delay_curve - min(delay_curve)); % positive value
delay_curve = delay_curve / max(delay_curve); % [0,1]
delay_curve_in_time = delay_curve - 0.5; % [-0.5,0.5]
% ------ Synthetic transmit aperture Field II ------
field_init(0)
set_field('fs',fs);
set_field('c', soundv);
% 1 tx and 128 rx in one time, total 128 elements so 128*128 acquisition
% xdc_linear_array(# of element, width, height, kerf, # of sub x, # of sub
% y, focus)
Th = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus); % pt to the transmitting aperture
Th2 = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus); % pt to the receiving aperture
tc = gauspuls('cutoff', fc, bw, -6, -80);
t = -tc:1/fs:tc;
% system impulse response
impulse_response = gauspuls(t, fc, bw, -6);
impulse_response = impulse_response.*hanning(length(impulse_response))';
xdc_impulse(Th, impulse_response); 
xdc_impulse(Th2, impulse_response);
% ecitation pulse
excitation = 10*sin(2*pi*fc*(0:1/fs:2/fc)); % like a rectangular wave?
excitation(excitation>1) = 1;
excitation(excitation<-1) = -1; % excitation range [-1,1]
excitation = excitation.*hanning(length(excitation))';
xdc_excitation(Th, excitation); 
% point sources location
positions = [0 0 1;0 0 7;0 0 14;0 0 21;0 0 28;0 0 60]/1000;
% original depth + random depth : [1, 10]/1e3 + [3.3,8.2]/1e3 + 2/1e3
positions(2:end-1, 3) = positions(2:end-1, 3) + (rand(1)*10)*1e-3 + (Kz_range)*soundv/fc + 2e-3; 
amp = ones(size(positions, 1), 1);
amp(1) = eps;
amp(end) = eps;
% calc_scat_all (Th1, Th2, pts' position, pts' amplitudes, dec_factor)
% recieved channel data v's dim = (?,Nelement*Nelement), where ? is related to
% sampling rate
[v, t] = calc_scat_all(Th, Th2, positions, amp, 1); 
xdc_free(Th);
field_end
full_dataset = reshape(v, [], Nelements, Nelements); % size = (?, received by the i-th element, emitted by the i-th element)
Noffset = (t*soundv/2)/dz_orig; % interval offset for the first pt source at 1mm, i.e. 1mm ~= ? interval (not exact 1mm/dz_orig)
Nsample = size(full_dataset, 1); % size(v, 1) depending on sampling rate


% Aline coordinate
% each element has "beamspace" Alines, e.g. 4 Alines consist of one element
% coordinate is at the center of Aline, the first right Aline is at 
% 1/beamspace*(beamspace-1)/2 and the last right Aline is at
% Nelement/2 - 1/beamspace*(beamspace-1)/2

% element coordinate without considering interpolation, i.e. beamspace = 1
% (Nelement-1)/2: element central location
% 1/beamspace*(beamspace-1)/2: the last aline*Nelements;
FOV = 1*Nelements;
x_elem = (-(Nelements-1)/2:(Nelements-1)/2)*pitch; % element central location
x_range = ((FOV-1)/2 + 1/beamspace*(beamspace-1)/2)*pitch;
x_aline = -x_range:pitch/beamspace:x_range; % aline x location
xx_elem = repelem(reshape(x_elem, 1,1,[]), Nsample, FOV*beamspace, 1);
xx_aline = repmat(reshape(x_aline, [1,FOV*beamspace,1]), [Nsample, 1, Nelements]); % (Nsample, aline)
z = dz_orig/Upsample*(Upsample*Noffset + (0:Upsample*Nsample-1)'); % depth axis
delay = abs(xx_aline - xx_elem + 1j*repmat(z,[1,FOV*beamspace, Nelements])); % (Nsample, aline, element)
% delay_idx = ceil(Upsample*delay_time/soundv*fs - Upsample*Noffset);
% delay_idx(delay_idx > Upsample*Nsample) = Upsample*Nsample + 1;



f_num_mask = double(repmat(z,[1,FOV*beamspace,Nelements])./abs(xx_aline - xx_elem)./2 > f_num); % f# = d/D



k = 1;


% ------ Apply delay profile ------
delay_max = [0, 1, 1.5, 2]; % in pi/4
% 
% for k = 1:4
    aberratedchanneldata = zeros(Nsample+1,Nelements,Nelements);
    delay_curve_in_sample = round(delay_curve_in_time * T_sample/4 * delay_max(k)); % convert [0,1/4]*lambda to # of samples
    % this loop spends about 1.68s
    for itx = 1:Nelements 
        % full_dataset(:, :, itx): one channel data, obtained by itx-th
        % transmitting element and all received Nelements.
        % delay_curve1(itx): tx delay for only one tx element
        % delay_curve1: rx delay for all Nelements
        aberratedchanneldata(1:end-1, :, itx) = Apply_Delay(full_dataset(:, :, itx), delay_curve_in_sample+delay_curve_in_sample(itx)); 
    end
    
    RFdata = zeros(Nsample, FOV); % field of views, x: -0.75*Nelement to 0.75*Nelement;
    for Nline = 1:FOV
        f_num_mask_tx = double(repmat(z,[1,Nelements,Nelements])./squeeze(xx_aline(:,Nline,:) - xx_elem(:,Nline,:))./2 > f_num);
        f_num_mask_rx = repmat(squeeze(delay(:,Nline,:)), [1,1,Nelements]);
        delay_iline = repelem(delay(:,Nline,:), 1, Nelements, 1) + repmat(squeeze(delay(:,Nline,:)), [1,1,Nelements]); % tx + rx
        delay_idx = ceil(Upsample*delay_iline/soundv*fs - Upsample*Noffset);
        delay_idx(delay_idx > Upsample*Nsample) = Upsample*Nsample + 1;
        channel_ind = delay_idx + ...
                      repelem((0:Nelements-1)*(Nsample+1), Nsample,1,Nelements) + ... % 2-nd dim index
                      repelem(reshape((Nsample+1)*Nelements*(0:Nelements-1), 1,1,[]), Nsample, Nelements, 1); % 3-rd dim index
        RFdata(:,Nline) = sum(f_num_mask_tx.*f_num_mask_rx.*aberratedchanneldata(channel_ind), [2,3]);
        if Nline == 64
            break
        end
        Nline
    end
    lpf = fir1(48, 0.8*bw*fc/(fs/2)).';
    bb_data = conv2(RFdata.*exp(-1i*2*pi*fc*(0:size(RFdata, 1)-1)'/fs), lpf, 'same');
    % -------- PSF Preprocessing ---------
    envelope = abs(bb_data);
    envelope_dB = 20*log10(envelope/max(envelope, [], 'all'));
    DR = 60;
    figure;
    image((-(size(bb_data,2)-1)/2:(size(bb_data,2)-1)/2)*pitch*1e3, (Noffset+(0:Nsample-1))*dz_orig*1e3, envelope_dB+DR);
    colormap(parula(DR));colormap;colorbar;
    axis image;

toc