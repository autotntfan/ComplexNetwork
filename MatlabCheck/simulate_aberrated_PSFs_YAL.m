clear
close all
addpath('./Field2');
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

% PSF size
beamspace = 4; % unit in samples (pixels), i.e. Aline owns how many samples
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
% f0 = z; <<<<<<<<<<<<
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
% ------ Apply delay profile ------
delay_max = [0, 1, 1.5, 2]; % in pi/4

% Aline coordinate
% each element has "beamspace" Alines, e.g. 4 Alines consist of one element
% coordinate is at the center of Aline, the first right Aline is at 
% 1/beamspace*(beamspace-1)/2 and the last right Aline is at
% Nelement/2 - 1/beamspace*(beamspace-1)/2
x_range = (Nelements-1)/2 + 1/beamspace*(beamspace-1)/2; 

x_bf = (-x_range:1/beamspace:x_range)*pitch;
% beamformed Aline, total Aline = beanspace*Nelement
x_ibf = reshape(x_bf,beamspace,[]);  % size = (beamspace, Nelement)
% element coordinate without considering interpolation, i.e. beamspace = 1
x_ref = (-(Nelements-1)/2:(Nelements-1)/2)*pitch;

z = dz_orig/Upsample*(Upsample*Noffset + (0:Upsample*Nsample-1)');
k = 1;
% 
% for k = 1:4
    delay_channel_data = zeros(Nsample, Nelements, Nelements);
    delay_curve_in_sample = round(delay_curve_in_time * T_sample/4 * delay_max(k)); % convert [0,1/4]*lambda to # of samples
    % this loop spends about 1.68s
    for itx = 1:Nelements 
        % full_dataset(:, :, itx): one channel data, obtained by itx-th
        % transmitting element and all received Nelements.
        % delay_curve1(itx): tx delay for only one tx element
        % delay_curve1: rx delay for all Nelements
        delay_channel_data(:, :, itx) = Apply_Delay(full_dataset(:, :, itx), delay_curve_in_sample+delay_curve_in_sample(itx)); 
    end
    % ------ calculate delay ------
    tic
    % delay_set size = [Nsample, beamspace*tx Nelement]
    % time delay for each Aline to different depth, it can be viewed as the
    % pure transmitted and received delay
    delay_set = abs(repmat(x_bf,[Nsample,1])+1j*repmat(z,[1,beamspace*Nelements]));

    % delay_tmp size = [Nsample, each element has "beamspace" beam, rx element], only the reason is
    % convenient to do further and the second dim is beamspace due to the x_bf order.
    delay_tmp = reshape(delay_set, [Nsample,beamspace,Nelements]);
    % delay_time = delay_tx + delay_rx;
    % IF WE CHECK BEANSPACE 1.2.3.4, EACH HAS [Nsample, rx Nelements]
    % delay_rx: for the i-th beamspace
    % the "rx Nelements" means received elements which have the common (only one) transmitted element, so
    % it repeats "Nelements" tx to calcuate the delay_rx.
    % delay_tx = repmat(delay_set,[1,1,Nelements]);

    % delay_rx: for the i-th beamspace, the different transmitted elements
    % have the common received elements. It means that the rx elements need
    % to be copy by "Nelements (tx)" times along beamspace (second dim) direction.
    % delay_rx = repmat(delay_tmp,[1,Nelements,1]);

    delay_time = repmat(delay_set,[1,1,Nelements]) + repmat(delay_tmp, [1,Nelements,1]); % [Nsample,rx Nelements*beamspace,tx Nelements]
    clear delay_tmp delay_set
    delay_time = permute(delay_time,[1 3 2]); % [Nsample,tx Nelements,rx Nelements*beamspace]

    x_element = reshape(repmat(x_ref',[1, beamspace*Nelements]),1,Nelements,[]); % i-th element x location
    x_beam = reshape(repmat(x_bf,[Nelements,1]),1,Nelements,[]); % i-th beam (aline) x location 
    % x_element and x_beam are of size [Nelement, beamspace*Nelement]
    % f # mask ensures lateral resolution at diff. depth to be almost the
    % same
    f_num_mask_tx = double(abs(repmat(z,[1,Nelements,beamspace*Nelements])./repmat(x_beam - x_element,[Nsample,1,1]))/2 > f_num); % f# = d/D
    for ii = 1:beamspace*Nelements
        f_num_mask_tx(:,:,ii) = f_num_mask_tx(:,:,ii)./sum(f_num_mask_tx(:,:,ii),2); % f# = d/D
    end
    toc

    delay_idx = round(Upsample*delay_time/soundv*fs - Upsample*Noffset); % convert delay time to i-th sample
    delay_idx(delay_idx > Upsample*Nsample) = Upsample*Nsample + 1; % all the i-th sample over Nsample belongs to the Nsample+1-th sample
    delay_idx = delay_idx + repmat([0:Upsample*Nsample+1:(Upsample*Nsample+1)*(Nelements-1)], [Upsample*Nsample,1, beamspace*Nelements]); % convert to array index
    append_elements = 0.25*Nelements;
    append_chan = Nelements + 2*append_elements;
    half_chan = Nelements/2; % 64
    

    delay_channel_data = [zeros(Nsample, append_elements, Nelements) delay_channel_data zeros(Nsample, append_elements, Nelements)]; 
    RF_data = zeros(Upsample*Nsample, Nelements, 1.5*beamspace*Nelements); % (?, 128, 192*4)

    for itx = 1:Nelements % 1 to 128
        tmp = zeros(Upsample*Nsample+1, 1.5*Nelements);
        tmp(1:end-1,:) = delay_channel_data(:,:,itx); % size = (?*Upsample+1, 1.5*Nelement)
        for irx = 1:beamspace*append_chan % 1 to 4*192
            
            if abs((itx+0.25*Nelements)-ceil(irx/beamspace)) < half_chan % the itx-th transmitted element being the center selects the left and right half channel elements as the received elements
                if ceil(irx/beamspace) < half_chan + 1 % left one-third elements -> half_chan - ceil(Nlines/beamspace) + 1 > 0
                    RF_data(:, half_chan-ceil(irx/beamspace)+2:end, irx) = RF_data(:, half_chan-ceil(irx/beamspace)+2:end, irx) +...
                        f_num_mask_tx(:, half_chan-ceil(irx/beamspace)+2:end, beamspace*(itx+0.25*Nelements)-(irx-beamspace*half_chan)+1)...
                        .*tmp(delay_idx(:, half_chan-ceil(irx/beamspace)+2:end, beamspace*(itx+0.25*Nelements)-(irx-beamspace*half_chan)+1) + ((ceil(irx/beamspace)-half_chan))*(Upsample*Nsample+1));
                elseif ceil(irx/beamspace) > (append_chan - half_chan) % right one-third elements
                    RF_data(:, 1:half_chan+(append_chan-ceil(irx/beamspace)), irx) = RF_data(:, 1:half_chan+(append_chan-ceil(irx/beamspace)), irx) +...
                        f_num_mask_tx(:, 1:half_chan+(append_chan-ceil(irx/beamspace)), beamspace*(itx+0.25*Nelements)-(irx-beamspace*half_chan)+1)...
                        .*tmp(delay_idx(:, 1:half_chan+(append_chan-ceil(irx/beamspace)), beamspace*(itx+0.25*Nelements)-(irx-beamspace*half_chan)+1) + ((ceil(irx/beamspace)-half_chan))*(Upsample*Nsample+1));
                else  
                    % middle one-third elements, Nlines > beamspace*half_chan
                    % tmp(delay(...)): size = [Nsample,Nelement]
                    RF_data(:, :, irx) = RF_data(:, :, irx) + ...
                        f_num_mask_tx(:, :, beamspace*(itx+0.25*Nelements)-(irx-beamspace*half_chan)+1) ...
                        .*tmp(delay_idx(:, :, beamspace*(itx+0.25*Nelements)-(irx-beamspace*half_chan)+1) + ((ceil(irx/beamspace)-half_chan))*(Upsample*Nsample+1));
                end
            end
        end
    end

    f_num_mask_rx = double(abs(repmat(z,[1, Nelements])./repmat(x_ref, [Upsample*Nsample, 1])/2) > f_num);
    f_num_mask_rx = f_num_mask_rx./sum(f_num_mask_rx, 2);
    
    rf_data = zeros(Upsample*Nsample, beamspace*append_chan);
    for idx = 1:beamspace*append_chan
        rf_data(:, idx) = sum(f_num_mask_rx.*RF_data(:, :, idx), 2);
    end
    rf_data = rf_data(1:Upsample:end, :);
    % end
    lpf = fir1(48, 0.8*bw*fc/(fs/2)).';
    bb_data = conv2(rf_data.*exp(-1i*2*pi*fc*(0:size(rf_data, 1)-1)'/fs), lpf, 'same');
    % -------- PSF Preprocessing ---------
    envelope = abs(bb_data);
    envelope_dB = 20*log10(envelope/max(envelope, [], 'all'));
    DR = 60;
    figure;
    image((-(size(bb_data,2)-1)/2:(size(bb_data,2)-1)/2)*pitch/beamspace*1e3, (Noffset+(0:Nsample-1))*dz_orig*1e3, envelope_dB+DR);
    colormap(parula(DR));colormap;colorbar;
    axis image;

