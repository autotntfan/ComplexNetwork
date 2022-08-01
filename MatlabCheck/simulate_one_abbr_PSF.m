addpath('../Field_II');

%% Imaging Parameters
f0 = (3 + 4.5*rand(1))*1e6; % 3 ~ 7.5 MHz
fs = 100e6;
bw = 0.5 + 0.3*rand(1); % 0.5 ~ 0.8
soundv = 1540;
lambda = soundv/f0;
height = 5e-3;
pitch = 3e-4;
% kerf = 0.01e-4;
kerf = 0;
Nelements = 128;
focus = [0 0 1e3]/1000;
f_num = 2;
dz_orig = soundv/fs/2;

%% PSF Size
Kx_dec = 8;
Kz_dec = 16;
Kx_range = 16;
Kz_range = 8;

% don't know what the fuck is this
% Kx_dec = 16;
% Kz_range = 16;

z_len = round(2.1*Kz_range*(fs/f0));
x_len = round(1.1*Kx_range*lambda/(pitch/4));

%% Generate scatterer distribution
scat_dist = zeros(2*Kz_range*Kz_dec+1, 2*Kx_range*Kx_dec+1);
den = 0;
while ~(den < 0.5 && den > 0.05)
    den = randn(1);
end
% disp(den)
N = round(den*(2*Kz_range*Kz_dec+1)*(2*Kx_range*Kx_dec+1));

scat_dist_z = round((2*Kz_range*Kz_dec+1)*rand(1, N));
while ~isempty(scat_dist_z(scat_dist_z<1))
    scat_dist_z(scat_dist_z<1) =  round((2*Kz_range*Kz_dec+1)*rand(1, length(scat_dist_z(scat_dist_z<1))));
end

scat_dist_x = round((2*Kx_range*Kx_dec+1)*rand(1, N));
while ~isempty(scat_dist_x(scat_dist_x<1))
    scat_dist_x(scat_dist_x<1) =  round((2*Kx_range*Kx_dec+1)*rand(1, length(scat_dist_x(scat_dist_x<1))));
end
for idx = 1:N
    scat_dist(scat_dist_z(idx), scat_dist_x(idx)) = randn(1);
end

%% Generate delay curve
x = randn(1, Nelements);
bt = 0.5;
span = 8;
sps = 16;
h = gaussdesign(bt, span, sps);
delay_curve = conv(x, h, 'same');
delay_curve = (delay_curve - min(delay_curve));
delay_curve = delay_curve / max(delay_curve);
delay_curve = delay_curve - 0.5;
T_sample = fs/f0;

% figure;
% plot(delay_curve);
% title('delay curve');

%% STA Field II
field_init(0)

set_field('fs',fs);
set_field('c', soundv);

Th = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus);
Th2 = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus);

tc = gauspuls('cutoff', f0, bw, -6, -80);
t = -tc:1/fs:tc;
impulse_response = gauspuls(t, f0, bw, -6);
impulse_response = impulse_response.*hanning(length(impulse_response)).';

xdc_impulse(Th, impulse_response);
xdc_impulse(Th2, impulse_response);

excitation = 10*sin(2*pi*f0*[0:1/fs:2/f0]);
excitation(excitation>1) = 1;
excitation(excitation<-1) = -1;
% bp=fir1(48, [0.66*f0/(fs/2) 1.34*f0/(fs/2)] , 'bandpass');
% excitation = conv(excitation, bp, 'same');
excitation = excitation.*hanning(length(excitation)).';

xdc_excitation(Th, excitation);

% point source location
positions = [...
    0 0 1;...
%     -15 0 0;...
%     -15 0 5;...
%     -15 0 10;...
%     -15 0 15;...

%     -8 0 0;...
%     -8 0 5;...
%     -8 0 10;...
%     -8 0 15;...

%     0 0 7;...
%     0 0 14;...
%     0 0 21;...
%     0 0 28;...
    0, 0, 7+rand(1)*28; ...
%     0, 0, 21+rand(1)*14; ...

%     8 0 0;...
%     8 0 5;...
%     8 0 10;...
%     8 0 15;...

%     15 0 0;...
%     15 0 5;...
%     15 0 10;...
%     15 0 15;...   

%     0 0 20;...
%     0 0 25;...
%     0 0 30;...
%     0 0 35;...
%     0 0 40;...
%     0 0 45;...
    0 0 60;...
%     0 0 55;...
%     0 0 60;...
%     0 0 65;...
    ]/1000;
% positions(2:end-1, 3) = positions(2:end-1, 3) + (rand(1)*10)*1e-3 + (Kz_range)*soundv/f0 + 2e-3;
positions(2:end-1, 3) = positions(2:end-1, 3);
% % positions(2:end-1, 3) = positions(2:end-1, 3) + (10)*1e-3 + (Kz_range)*soundv/f0 + 1e-3;
% positions(2:end-1, 1) = (rand(4,1)*16-8)/1000;

amp = ones(size(positions, 1), 1);
amp(1) = eps;
amp(end) = eps;
[v, t] = calc_scat_all(Th, Th2, positions, amp, 1);


xdc_free(Th);
field_end
full_dataset = reshape(v, [size(v, 1), Nelements, Nelements]);
Noffset = (t*soundv/2)/dz_orig;
Nsample = size(full_dataset, 1);

delay_max = [0, 1, 1.5, 2]; % in pi/4
for k = 1:4
    %% Apply Delay
    phase_abb_amount = delay_max(k);
    
    full_dataset1 = zeros(Nsample, Nelements, Nelements);
    if k > 0
        delay_curve1 = round(delay_curve * T_sample/4 * phase_abb_amount);
        for tx = 1:Nelements
            tmp = full_dataset(:, :, tx);
            tmp = Apply_Delay(tmp, delay_curve1+delay_curve1(tx));
            full_dataset1(:, :, tx) = tmp;
        end
    else
        full_dataset1 = full_dataset;
    end
    
    phase_abb_amount = phase_abb_amount * pi /4;

    %% STA Beamforming
    Chan = 64;
%     x = [-(Nelements-1)/2:(Nelements-1)/2]*pitch;
    x_range = (Chan-1)/2+1/4*1.5;

    x_bf = [-x_range:1/4:x_range]*pitch;
    x_ibf = zeros(4, length(x_bf)/4);
    x_ibf(1, :) = x_bf(1:4:end);
    x_ibf(2, :) = x_bf(2:4:end);
    x_ibf(3, :) = x_bf(3:4:end);
    x_ibf(4, :) = x_bf(4:4:end);

    x_ref = [-(Chan-1)/2:(Chan-1)/2]*pitch;


    %% Tx 
    delay = zeros(Nsample, Chan, 4*Chan);

    f_num_mask_tx = zeros(Nsample, Chan, 4*Chan);
    for delay_tx = 1:Chan
        for ibf = 1:4
            delay(:, :, 4*(delay_tx-1)+ibf) = ...
                abs(repmat(x_ibf(ibf, :), [Nsample, 1]) + 1i*dz_orig*repmat(Noffset+[0:Nsample-1].', [1, Chan])) +...    % rx
                abs(repmat(x_ibf(ibf, delay_tx), [Nsample, Chan]) + 1i*dz_orig*repmat(Noffset+[0:Nsample-1].', [1, Chan]));   % tx        
            pos1_tx = repmat(x_ref-x_ibf(ibf, delay_tx), [Nsample, 1]) + 1i*dz_orig*repmat(Noffset+[0:Nsample-1].', [1, Chan]);
            f_num_mask_tx(:, :, 4*(delay_tx-1)+ibf) = abs(imag(pos1_tx)./real(pos1_tx))/2 > f_num;
        end
    end
    delay_idx = round(delay/soundv*fs - Noffset);
    delay_idx(delay_idx > Nsample) = Nsample+1;
    delay_idx = delay_idx + repmat([0:Nsample+1:(Nsample+1)*(Chan-1)], [Nsample,1, 4*Chan]);

    Chan_half = Chan/2;
    chan_data = zeros(Nsample, Chan, 4*Nelements);    
    for tx = 1:Nelements/2
%         tx
        tmp1 = zeros(size(full_dataset1, 1), size(full_dataset1, 2)/2);
        tmp2 = zeros(size(full_dataset1, 1), size(full_dataset1, 2)/2);

        tmp1 = full_dataset1(:, :, Nelements/2+tx);
        tmp2 = full_dataset1(:, end:-1:1, Nelements/2+1-tx); 

        tmp1 = [tmp1; zeros(1, Nelements)];
        tmp2 = [tmp2; zeros(1, Nelements)];

        for Nlines = 4*Chan_half+1:4*Nelements
            if abs(tx+Nelements/2-ceil(Nlines/4)) < Chan_half
                if ceil(Nlines/4) > (Nelements - Chan_half)
                    chan_data(:, 1:Chan_half+(Nelements-ceil(Nlines/4)), Nlines) = chan_data(:, 1:Chan_half+(Nelements-ceil(Nlines/4)), Nlines) +...
                        f_num_mask_tx(:, 1:Chan_half+(Nelements-ceil(Nlines/4)), 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1).*tmp1(delay_idx(:, 1:Chan_half+(Nelements-ceil(Nlines/4)), 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1) + ((ceil(Nlines/4)-Chan_half)-1)*(Nsample+1));
                    chan_data(:, end:-1:end+1-(Chan_half+(Nelements-ceil(Nlines/4))), end-Nlines+1) = chan_data(:, end:-1:end+1-(Chan_half+(Nelements-ceil(Nlines/4))), end-Nlines+1) +...
                        f_num_mask_tx(:, 1:Chan_half+(Nelements-ceil(Nlines/4)), 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1).*tmp2(delay_idx(:, 1:Chan_half+(Nelements-ceil(Nlines/4)), 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1) + ((ceil(Nlines/4)-Chan_half)-1)*(Nsample+1));
                else
                    chan_data(:, :, Nlines) = chan_data(:, :, Nlines) +...
                        f_num_mask_tx(:, :, 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1).*tmp1(delay_idx(:, :, 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1) + ((ceil(Nlines/4)-Chan_half)-1)*(Nsample+1));
                    chan_data(:, end:-1:1, end-Nlines+1) = chan_data(:, end:-1:1, end-Nlines+1) +...
                        f_num_mask_tx(:, :, 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1).*tmp2(delay_idx(:, :, 4*(tx+Nelements/2)-(Nlines-4*Chan_half+4*1)+1) + ((ceil(Nlines/4)-Chan_half)-1)*(Nsample+1));
                end
            end
        end

    end


    pos_rx = repmat(x_ref, [Nsample, 1]) + 1i*dz_orig*repmat(Noffset+[0:Nsample-1].', [1, Chan]);
    f_num_mask_rx = double(abs(imag(pos_rx)./real(pos_rx))/2 > f_num);
    % for irx1 = 1:size(f_num_mask_rx)
    %     f_num_mask_rx(irx1, (f_num_mask_rx(irx1, :)>0)) = f_num_mask_rx(irx1, (f_num_mask_rx(irx1, :)>0)).*hanning(sum(f_num_mask_rx(irx1, :))).';
    % end
    f_num_mask_rx_weight = sum(f_num_mask_rx, 2);

    rf_data = zeros(size(sum(f_num_mask_rx.*chan_data(:, :, 1), 2), 1), 4*Nelements);

    for idx = 1:4*Nelements
        rf_data(:, idx) = sum(f_num_mask_rx.*chan_data(:, :, idx), 2);
    %     rf_data(:, idx) = rf_data(:, idx)./f_num_mask_rx_weight;
    end

    lpf =fir1(48, 0.5*bw*f0/(fs/2)).';
    bb_data = conv2(rf_data.*exp(-sqrt(-1)*2*pi*f0/fs*[0:size(rf_data, 1)-1].'), lpf, 'same');
    %% PSF Preprocessing
%     rf_data(abs(bb_data)/max(abs(bb_data(:))) < 1e-2) = 0;
%     bb_data(abs(bb_data)/max(abs(bb_data(:))) < 1e-2) = 0;
    envelope = abs(bb_data);
    rf_data1 = rf_data;
%     rf_data1 = imresize(rf_data, [size(rf_data, 1), 4*size(rf_data, 2)]);
    bb_data1 = conv2(rf_data1.*exp(-sqrt(-1)*2*pi*f0/fs*[0:size(rf_data1, 1)-1].'), lpf, 'same');
    envelope1 = abs(bb_data1);
    envelope1_dB = 20*log10(envelope1/max(envelope1, [], 'all'));
    
%     DR = 60;
%     figure;
%     image([-(size(bb_data1,2)-1)/2:(size(bb_data1,2)-1)/2]*pitch/4*1e3, (Noffset+[0:Nsample-1])*dz_orig*1e3,envelope1_dB+DR);
%     colormap(parula(DR));colormap;colorbar;
%     axis image;

    Ndep = round((positions(:,3) - Noffset*dz_orig)/dz_orig);
    dN = round((Ndep(2) - Ndep(1))/2);

    % idz = zeros(size(positions, 1)-2, 1);
    % idx = zeros(size(positions, 1)-2, 1);
    % fs = Kz_dec/2*f0;
    dx = lambda/Kx_dec;
    dz = lambda/Kz_dec;
    lpf = fir1(48, bw/(Kz_dec/2/2)).';
    % psf_rf_all = zeros(2*Kz_range*Kz_dec+1, 2*Kx_range*Kx_dec+1, size(positions, 1)-2);
    % psf_bb_all = zeros(2*Kz_range*Kz_dec+1, 2*Kx_range*Kx_dec+1, size(positions, 1)-2);

    for I = 1:size(positions, 1)-2
        [idz, idx] = find(envelope == max(max(envelope(Ndep(I+1)-dN:Ndep(I+1)+dN, :))));
        tmp = rf_data(idz-z_len:idz+z_len, idx-x_len:idx+x_len);
    %     tmp = rf_data(idz-z_len:idz+z_len, (size(rf_data, 2)+1)/2-x_len:(size(rf_data, 2)+1)/2+x_len);
    %     tmp = rf_data(idz-z_len:idz+z_len, :);

        tmp = interp2_rat(tmp, Kz_dec/2*f0/fs, Kx_dec*pitch/4/lambda);
        [size1, size2] = size(tmp);   
        cent1 = round((size1+1)/2);
        cent2 = round((size2+1)/2);
        psf_rf = tmp(cent1-Kz_range*Kz_dec:cent1+Kz_range*Kz_dec, cent2-Kx_range*Kx_dec:cent2+Kx_range*Kx_dec);
        psf_bb = conv2(psf_rf.*(exp(-1i*2*pi*1/(Kz_dec/2)).^([0:size(psf_rf,1)-1]).'), lpf, 'same');
        depth = positions(I+1, 3);


        envelope_dB = 20*log10(abs(psf_bb)/max(max(abs(psf_bb))));
        speckle_rf = conv2(scat_dist, psf_rf, 'same');
        speckle_bb = conv2(speckle_rf.*(exp(-1i*2*pi*1/(Kz_dec/2)).^([0:size(speckle_rf,1)-1]).'), lpf, 'same');
        data_envelope_dB = 20*log10(abs(speckle_bb)/max(max(abs(speckle_bb))));

        % plot and save figure
        DR = 60;
        fig = figure;
        subplot(1,2,1)
        image([-Kx_range*Kx_dec:Kx_range*Kx_dec]*dx*1e3, [-Kz_range*Kz_dec:Kz_range*Kz_dec]*dz*1e3+(positions(I+1, 3)*1e3), envelope_dB+DR);
        colormap(gray(DR));colorbar;
        axis image;
        xlabel('Lateral position (mm)');
        ylabel('Depth (mm)');
        subplot(1,2,2)
        image([-Kx_range*Kx_dec:Kx_range*Kx_dec]*dx*1e3, [-Kz_range*Kz_dec:Kz_range*Kz_dec]*dz*1e3, data_envelope_dB+DR);
        colormap(gray(DR));colorbar;
        axis image;
        xlabel('Lateral position (mm)');
        ylabel('Depth (mm)');
        
        out_fig_fpath = char(sprintf('./simulation_data2/Data_%d_delay_%g.png', data_id, k));
        saveas(fig, out_fig_fpath);
        
        close(fig)

        % save simulated data
        out_fpath = char(sprintf('./simulation_data2/Data_%d_delay_%g.mat', data_id, k));
        save(out_fpath, 'psf_rf', 'psf_bb', 'speckle_rf', 'speckle_bb', 'dx', 'dz', 'depth', 'f0', 'k', 'bw', 'delay_curve', 'phase_abb_amount');
    end
end
% end
