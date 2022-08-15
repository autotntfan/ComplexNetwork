% clearvars -except id n iid;
% % close all;
% rng('shuffle');
% 
% 
% %% Model Parameter
% f0 = 0;
% while ~(f0 < 7.5 && f0 > 3)
%     f0 = rand(1)*7.5;
% end
% f0 
% f0 = f0*1e6;
% fs = 100e6;
% 
% bw = 0;
% while ~(bw < 0.8 && bw > 0.5)
%     bw = rand(1);
% end
% bw
% % bw = 0.68;
% 
% 
% soundv = 1540;
% lambda = soundv/f0;
% height = 5e-3;
% pitch = 3e-4;
% % kerf = 0.01e-4;
% kerf = 0;
% Nelements = 128;
% focus = [0 0 1e3]/1000;
% f_num = 2;
% dz_orig = soundv/fs/2;
% 
% %%
% 
% %% PSF Size
% Upsample = 1;
% beamspace = 4;
% Kx_dec = 8;
% Kz_dec = 16;
% Kx_range = 16;
% Kz_range = 8;
% 
% z_len = round(2.1*Kz_range*(fs/f0));
% x_len = round(1.1*Kx_range*lambda/(pitch/beamspace));
% 
% %%
% 
% %% Scatterer Distribution
% scat_dist = zeros(2*Kz_range*Kz_dec+1, 2*Kx_range*Kx_dec+1);
% den = 0;
% while ~(den < 0.5 && den > 0.05)
%     den = randn(1);
% end
% den
% N = round(den*(2*Kz_range*Kz_dec+1)*(2*Kx_range*Kx_dec+1));
% 
% scat_dist_z = round((2*Kz_range*Kz_dec+1)*rand(1, N));
% while length(scat_dist_z(scat_dist_z<1)) > 0
%     scat_dist_z(scat_dist_z<1) =  round((2*Kz_range*Kz_dec+1)*rand(1, length(scat_dist_z(scat_dist_z<1))));
% end
% 
% scat_dist_x = round((2*Kx_range*Kx_dec+1)*rand(1, N));
% while length(scat_dist_x(scat_dist_x<1)) > 0
%     scat_dist_x(scat_dist_x<1) =  round((2*Kx_range*Kx_dec+1)*rand(1, length(scat_dist_x(scat_dist_x<1))));
% end
% for idx = 1:N
%     scat_dist(scat_dist_z(idx), scat_dist_x(idx)) = randn(1);
% end
% 
% %%
% 
% %% Delay Profile
% x = randn(1, Nelements);
% bt = 0.5;
% span = 8;
% sps = 16;
% h = gaussdesign(bt, span, sps);
% delay_curve = conv(x, h, 'same');
% delay_curve = (delay_curve - min(delay_curve));
% delay_curve = delay_curve / max(delay_curve);
% delay_curve = delay_curve - 0.5;
% figure;
% plot(delay_curve);
% title('delay curve');
% T_sample = fs/f0;
% 
% 
% %%
% 
% %% STA Field II
% field_init(0)
% 
% set_field('fs',fs);
% set_field('c', soundv);
% 
% 
% Th = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus);
% Th2 = xdc_linear_array(Nelements, pitch, height, kerf, 1, 1, focus);
% 
% 
% tc = gauspuls('cutoff', f0, bw, -6, -80);
% t = -tc:1/fs:tc;
% impulse_response = gauspuls(t, f0, bw, -6);
% impulse_response = impulse_response.*hanning(length(impulse_response)).';
% 
% xdc_impulse(Th, impulse_response);
% xdc_impulse(Th2, impulse_response);
% 
% 
% excitation = 10*sin(2*pi*f0*[0:1/fs:2/f0]);
% excitation(excitation>1) = 1;
% excitation(excitation<-1) = -1;
% % bp=fir1(48, [0.66*f0/(fs/2) 1.34*f0/(fs/2)] , 'bandpass');
% % excitation = conv(excitation, bp, 'same');
% excitation = excitation.*hanning(length(excitation)).';
% 
% xdc_excitation(Th, excitation);
% 
% 
% positions = [...
%     0 0 1;...
% %     -15 0 0;...
% %     -15 0 5;...
% %     -15 0 10;...
% %     -15 0 15;...
%     
% %     -8 0 0;...
% %     -8 0 5;...
% %     -8 0 10;...
% %     -8 0 15;...
%     
%     0 0 7;...
%     0 0 14;...
%     0 0 21;...
%     0 0 28;...
%     
% %     8 0 0;...
% %     8 0 5;...
% %     8 0 10;...
% %     8 0 15;...
%     
% %     15 0 0;...
% %     15 0 5;...
% %     15 0 10;...
% %     15 0 15;...   
%  
% %     0 0 20;...
% %     0 0 25;...
% %     0 0 30;...
% %     0 0 35;...
% %     0 0 40;...
% %     0 0 45;...
%     0 0 50;...
% %     0 0 55;...
% %     0 0 60;...
% %     0 0 65;...
%     ]/1000;
% positions(2:end-1, 3) = positions(2:end-1, 3) + (rand(1)*10)*1e-3 + (Kz_range)*soundv/f0 + 2e-3;
% % positions(2:end-1, 3) = positions(2:end-1, 3) + (10)*1e-3 + (Kz_range)*soundv/f0 + 1e-3;
% positions(2:end-1, 1) = (rand(4,1)*32-16)/1000;
% 
% amp = ones(size(positions, 1), 1);
% amp(1) = eps;
% amp(end) = eps;
% [v, t] = calc_scat_all(Th, Th2, positions, amp, 1);
% 
% 
% xdc_free(Th);
% field_end
% full_dataset = reshape(v, [size(v, 1), Nelements, Nelements]);
% Noffset = (t*soundv/2)/dz_orig;
% Nsample = size(full_dataset, 1);

%%

%% STA Beamforming

Chan = Nelements;
%     x = [-(Nelements-1)/2:(Nelements-1)/2]*pitch;
x_range = (Chan-1)/2+1/beamspace*(beamspace-1)/2;

x_bf = [-x_range:1/beamspace:x_range]*pitch;
x_ibf = zeros(beamspace, length(x_bf)/beamspace);

for ibs = 1:beamspace
    x_ibf(ibs, :) = x_bf(ibs:beamspace:end);
end

x_ref = [-(Chan-1)/2:(Chan-1)/2]*pitch;


%%
k = 1;
%% Apply Delay
delay_max = [0, 1, 1.5, 2];
% for k = 1:4

    full_dataset1 = zeros(Nsample, Nelements, Nelements);
    if k > 1
        delay_curve1 = round(delay_curve * T_sample/4 * delay_max(k));
        for tx = 1:Nelements
            tmp = full_dataset(:, :, tx);
            tmp = Apply_Delay(tmp, delay_curve1+delay_curve1(tx));
            full_dataset1(:, :, tx) = tmp;
        end
    else
        full_dataset1 = full_dataset;
    end
    % size = [Nsample, 1.5*Nelements, Nelements], may mean that other
    % virtual elements exist to do "draw circle" well. i.e. append zeros
    full_dataset1 = [zeros(Nsample, Nelements*0.25, Nelements) full_dataset1  zeros(Nsample, Nelements*0.25, Nelements)];
    NChan = 1.5*Nelements;
    %% Tx 
    delay = zeros(Upsample*Nsample, Chan, beamspace*Chan);

    f_num_mask_tx = zeros(Upsample*Nsample, Chan, beamspace*Chan);
    for delay_tx = 1:Chan
        for ibf = 1:beamspace
            delay(:, :, beamspace*(delay_tx-1)+ibf) = ...
                abs(repmat(x_ibf(ibf, :), [Upsample*Nsample, 1]) + 1i*dz_orig/Upsample*repmat(Upsample*Noffset+[0:Upsample*Nsample-1].', [1, Chan])) +...    % rx
                abs(repmat(x_ibf(ibf, delay_tx), [Upsample*Nsample, Chan]) + 1i*dz_orig/Upsample*repmat(Upsample*Noffset+[0:Upsample*Nsample-1].', [1, Chan]));   % tx        
            pos1_tx = repmat(x_ref-x_ibf(ibf, delay_tx), [Upsample*Nsample, 1]) + 1i*dz_orig/Upsample*repmat(Upsample*Noffset+[0:Upsample*Nsample-1].', [1, Chan]);
            f_num_mask_tx(:, :, beamspace*(delay_tx-1)+ibf) = abs(imag(pos1_tx)./real(pos1_tx))/2 > f_num;
            f_num_mask_tx(:, :, beamspace*(delay_tx-1)+ibf) = f_num_mask_tx(:, :, beamspace*(delay_tx-1)+ibf)./sum(f_num_mask_tx(:, :, beamspace*(delay_tx-1)+ibf), 2);

        end
    end
    % ------ time to index ------
    delay_idx = round(Upsample*delay/soundv*fs - Upsample*Noffset); % convert delay time to i-th sample
    % delay_idx size = (?,128,512)
    delay_idx(delay_idx > Upsample*Nsample) = Upsample*Nsample+1; % all the i-th sample over Nsample belongs to the Nsample+1-th sample
    delay_idx = delay_idx + repmat([0:Upsample*Nsample+1:(Upsample*Nsample+1)*(Chan-1)], [Upsample*Nsample,1, beamspace*Chan]); % convert to array index

    Chan_half = Chan/2; % 64
    chan_data = zeros(Upsample*Nsample, Chan, beamspace*NChan); % (?, 128, 192*4)

    for tx = 1:Nelements % 1 to 128
        tx
        tmp = zeros(Upsample*size(full_dataset1, 1), size(full_dataset1, 2)); % size = (?, Nelement)
        parfor ch = 1:size(full_dataset1, 2)
            tmp(:, ch) = interp(full_dataset1(:, ch, tx), Upsample);
        end
        tmp = [tmp; zeros(1, NChan)]; % size = (?*Upsample+1, 1.5*Nelement)

        for Nlines = 1:beamspace*NChan % 1 to 4*192
            % itx + 0.25*Nelements: adjust to the new index of transmitted element
            % ceil(Nlines/beamspace): the i-th received element
            if abs((tx+0.25*Nelements)-ceil(Nlines/beamspace)) < Chan_half % the itx-th transmitted element being the center selects the left and right half channel elements as the received elements
                if ceil(Nlines/beamspace) < Chan_half+1 % left one-third elements -> half_chan - ceil(Nlines/beamspace) + 1 > 0
                    % half_chan - ceil(Nlines/beamspace) + 2:end -> at least 2 to end
                    % beamspace*(itx+0.25*Nelements+half_chan) - Nlines + 1
                    chan_data(:, Chan_half-ceil(Nlines/beamspace)+2:end, Nlines) = chan_data(:, Chan_half-ceil(Nlines/beamspace)+2:end, Nlines) +...
                        f_num_mask_tx(:, Chan_half-ceil(Nlines/beamspace)+2:end, beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1)...
                        .*tmp(delay_idx(:, Chan_half-ceil(Nlines/beamspace)+2:end, beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1) + ((ceil(Nlines/beamspace)-Chan_half))*(Upsample*Nsample+1));
                elseif ceil(Nlines/beamspace) > (NChan - Chan_half) % right one-third elements
                    chan_data(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), Nlines) = chan_data(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), Nlines) +...
                        f_num_mask_tx(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1)...
                        .*tmp(delay_idx(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1) + ((ceil(Nlines/beamspace)-Chan_half))*(Upsample*Nsample+1));
                else 
                    % middle one-third elements, Nlines > beamspace*half_chan
                    % tmp(delay(...)): size = [Nsample,Nelement]
                    chan_data(:, :, Nlines) = chan_data(:, :, Nlines) +...
                        f_num_mask_tx(:, :, beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1)...
                        .*tmp(delay_idx(:, :, beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1) + ((ceil(Nlines/beamspace)-Chan_half))*(Upsample*Nsample+1));
                end
            end
        end

    end    

    pos_rx = repmat(x_ref, [Upsample*Nsample, 1]) + 1i*dz_orig/Upsample*repmat(Upsample*Noffset+[0:Upsample*Nsample-1].', [1, Chan]);
    f_num_mask_rx = double(abs(imag(pos_rx)./real(pos_rx))/2 > f_num);
    % for irx1 = 1:size(f_num_mask_rx)
    %     f_num_mask_rx(irx1, (f_num_mask_rx(irx1, :)>0)) = f_num_mask_rx(irx1, (f_num_mask_rx(irx1, :)>0)).*hanning(sum(f_num_mask_rx(irx1, :))).';
    % end
    f_num_mask_rx = f_num_mask_rx./sum(f_num_mask_rx, 2);

    rf_data = zeros(Upsample*Nsample, beamspace*NChan);
    for idx = 1:beamspace*NChan
        rf_data(:, idx) = sum(f_num_mask_rx.*chan_data(:, :, idx), 2);
    end
    rf_data = rf_data(1:Upsample:end, :);
    lpf =fir1(48, 0.8*bw*f0/(fs/2)).';
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
    DR = 60;
    figure;
    image([-(size(bb_data1,2)-1)/2:(size(bb_data1,2)-1)/2]*pitch/beamspace*1e3, (Noffset+[0:Nsample-1])*dz_orig*1e3,envelope1_dB+DR);
    colormap(parula(DR));colormap;colorbar;
    axis image;
% %{
    %%

%     Ndep = round((positions(:,3) - Noffset*dz_orig)/dz_orig);
%     dN = round((Ndep(2) - Ndep(1))/2);
%     %%
% 
% 
%     % idz = zeros(size(positions, 1)-2, 1);
%     % idx = zeros(size(positions, 1)-2, 1);
%     % fs = Kz_dec/2*f0;
%     dx = lambda/Kx_dec;
%     dz = lambda/Kz_dec;
%     lpf = fir1(48, bw/(Kz_dec/2/2)).';
%     % psf_rf_all = zeros(2*Kz_range*Kz_dec+1, 2*Kx_range*Kx_dec+1, size(positions, 1)-2);
%     % psf_bb_all = zeros(2*Kz_range*Kz_dec+1, 2*Kx_range*Kx_dec+1, size(positions, 1)-2);
% 
%     for I = 1:size(positions, 1)-2
%         [idz, idx] = find(envelope == max(max(envelope(Ndep(I+1)-dN/2:Ndep(I+1)+dN/2, :))));
%         tmp = rf_data(idz-z_len:idz+z_len, idx-x_len:idx+x_len);
%     %     tmp = rf_data(idz-z_len:idz+z_len, (size(rf_data, 2)+1)/2-x_len:(size(rf_data, 2)+1)/2+x_len);
%     %     tmp = rf_data(idz-z_len:idz+z_len, :);
% 
%         tmp = interp2_rat(tmp, Kz_dec/2*f0/fs, Kx_dec*pitch/beamspace/lambda);
%         [size1, size2] = size(tmp);   
%         cent1 = round((size1+1)/2);
%         cent2 = round((size2+1)/2);
%         psf_rf = tmp(cent1-Kz_range*Kz_dec:cent1+Kz_range*Kz_dec, cent2-Kx_range*Kx_dec:cent2+Kx_range*Kx_dec);
%         psf_bb = conv2(psf_rf.*(exp(-1i*2*pi*1/(Kz_dec/2)).^([0:size(psf_rf,1)-1]).'), lpf, 'same');
%         depth = positions(I+1, 3);
% 
% 
%         
% 
%         envelope_dB = 20*log10(abs(psf_bb)/max(max(abs(psf_bb))));
%         speckle_rf = conv2(scat_dist, psf_rf, 'same');
%         speckle_bb = conv2(speckle_rf.*(exp(-1i*2*pi*1/(Kz_dec/2)).^([0:size(speckle_rf,1)-1]).'), lpf, 'same');
%         data_envelope_dB = 20*log10(abs(speckle_bb)/max(max(abs(speckle_bb))));
% 
%         fig = figure;
%         subplot(1,2,1)
%         image([-Kx_range*Kx_dec:Kx_range*Kx_dec]*dx*1e3, [-Kz_range*Kz_dec:Kz_range*Kz_dec]*dz*1e3+(positions(I+1, 3)*1e3), envelope_dB+DR);
%         colormap(parula(DR));colorbar;
%         axis image;
%         xlabel('Lateral position (mm)');
%         ylabel('Depth (mm)');
%         subplot(1,2,2)
%         image([-Kx_range*Kx_dec:Kx_range*Kx_dec]*dx*1e3, [-Kz_range*Kz_dec:Kz_range*Kz_dec]*dz*1e3, data_envelope_dB+DR);
%         colormap(parula(DR));colorbar;
%         axis image;
%         xlabel('Lateral position (mm)');
%         ylabel('Depth (mm)');
%         K = delay_max(k);
%         save(['Data_', num2str((k-1)*2*2+id+I-1), '.mat'], 'psf_rf', 'psf_bb', 'speckle_rf', 'speckle_bb', 'dx', 'dz', 'depth', 'f0', 'K', 'bw');
%         saveas(fig, ['Data_', num2str((k-1)*2*2+id+I-1), '.png']);
%     end
% %     %}
% end
