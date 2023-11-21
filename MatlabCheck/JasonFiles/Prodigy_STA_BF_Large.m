clear;clc;%close all;

% str = 'IdealPSF/IdealPSF_full_Frame';
str = './IdealPSF/IdealPSF_full_Frame';
load([str, '1_EG1.mat'], 'EG_para', 'Overall_para', 'Event_para_hdr');

Nelements = Overall_para.Ele_num;
f0 = EG_para.Freq*1e6;
fs = Overall_para.fs*1e6;
soundv = Overall_para.soundv;
dz_orig = soundv/fs/2;
f_num = 2;

iter = Overall_para.Frames;
Noffset = Event_para_hdr.Toffset;
Noffset = round(Noffset*fs);
N_dep_min = EG_para.MinDepth_sample;
N_dep = EG_para.MaxDepth_sample;

Nsample = N_dep-N_dep_min+1;
pitch = Overall_para.Pitch;
bp = fir1(48, [0.1, 0.9]).';
clear EG_para Overall_para Event_para_hdr;
full_dataset = zeros(Nsample, 128, 128);

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
full_dataset = full_dataset/iter;
%%

%% Delay Profile
x = randn(1, Nelements);
bt = 0.5;
span = 8;
sps = 16;
h = gaussdesign(bt, span, sps);
delay_curve = conv(x, h, 'same');
delay_curve = (delay_curve - min(delay_curve));
delay_curve = delay_curve / max(delay_curve);
delay_curve = delay_curve - 0.5;
figure;
plot(delay_curve);
title('delay curve');
T_sample = fs/f0;
Nsample = size(full_dataset, 1);

%  full_dataset1 = zeros(Nsample, Nelements, Nelements);
%%
% k = 2;
% delay_curve1 = round(delay_curve * T_sample/4 * k);
% for tx = 1:Nelements
%     tmp = full_dataset(:, :, tx);
%     tmp = Apply_Delay(tmp, delay_curve1+delay_curve1(tx));
%     full_dataset(:, :, tx) = tmp;
% end

full_dataset1 = [zeros(Nsample, Nelements*0.25, Nelements) full_dataset  zeros(Nsample, Nelements*0.25, Nelements)];
NChan = 1.5*Nelements;
%% STA
beamspace = 2;
Upsample = 8;

Chan = Nelements;
%     x = [-(Nelements-1)/2:(Nelements-1)/2]*pitch;
x_range = (Chan-1)/2+1/beamspace*(beamspace-1)/2;

x_bf = [-x_range:1/beamspace:x_range]*pitch;
x_ibf = zeros(beamspace, length(x_bf)/beamspace);

for ibs = 1:beamspace
    x_ibf(ibs, :) = x_bf(ibs:beamspace:end);
end

x_ref = [-(Chan-1)/2:(Chan-1)/2]*pitch;



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
delay_idx = round(Upsample*delay/soundv*fs - Upsample*Noffset);
delay_idx(delay_idx > Upsample*Nsample) = Upsample*Nsample+1;
delay_idx = delay_idx + repmat([0:Upsample*Nsample+1:(Upsample*Nsample+1)*(Chan-1)], [Upsample*Nsample,1, beamspace*Chan]);

Chan_half = Chan/2;
chan_data = zeros(Upsample*Nsample, Chan, beamspace*NChan);    
 
for tx = 1:Nelements
    tx
    tmp = zeros(Upsample*size(full_dataset1, 1), size(full_dataset1, 2));
    for ch = 1:size(full_dataset1, 2)
        tmp(:, ch) = interp(full_dataset1(:, ch, tx), Upsample);
    end
    tmp = [tmp; zeros(1, NChan)];

    for Nlines = 1:beamspace*NChan
        if abs((tx+0.25*Nelements)-ceil(Nlines/beamspace)) < Chan_half
            if ceil(Nlines/beamspace) < Chan_half+1
                chan_data(:, Chan_half-ceil(Nlines/beamspace)+2:end, Nlines) = chan_data(:, Chan_half-ceil(Nlines/beamspace)+2:end, Nlines) +...
                    f_num_mask_tx(:, Chan_half-ceil(Nlines/beamspace)+2:end, beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1)...
                    .*tmp(delay_idx(:, Chan_half-ceil(Nlines/beamspace)+2:end, beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1) + ((ceil(Nlines/beamspace)-Chan_half))*(Upsample*Nsample+1));
            elseif ceil(Nlines/beamspace) > (NChan - Chan_half)
                chan_data(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), Nlines) = chan_data(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), Nlines) +...
                    f_num_mask_tx(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1)...
                    .*tmp(delay_idx(:, 1:Chan_half+(NChan-ceil(Nlines/beamspace)), beamspace*(tx+0.25*Nelements)-(Nlines-beamspace*Chan_half)+1) + ((ceil(Nlines/beamspace)-Chan_half))*(Upsample*Nsample+1));
            else 
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

%%

rf_data_print = imresize(rf_data, [size(full_dataset, 1), NChan*8]);
filter =fir1(48, 0.8*f0/(fs/2));
bb_data = conv2(rf_data_print.*exp(-sqrt(-1)*2*pi*f0/fs*[0:size(rf_data, 1)-1].'), filter.', 'same');
envelope = abs(bb_data);
envelope_dB = 20*log10(envelope/max(envelope(500:end,:), [], 'all'));
DR = 80;
figure;
image([-(size(bb_data,2)-1)/2:(size(bb_data,2)-1)/2]*pitch/8*1e3, (Noffset+[0:Nsample-1])*dz_orig*1e3,envelope_dB+DR);
axis image;
ylim([Noffset*dz_orig*1e3 N_dep*dz_orig*1e3])
% colormap(parula(DR));colormap;colorbar;
colormap(gray(DR));colormap;colorbar;

% function rf_data_delayed = Apply_Delay(rf_data, delay_curve)
%   [Nsample, Nelement] = size(rf_data);
%   rf_data_delayed = zeros(Nsample, Nelement);
%     for idx = 1:Nelement
%        A_line = rf_data(:, idx);
%        K = delay_curve(idx);
%        
%        if K > 0
%            delayed_signal = [zeros(K, 1); A_line(1:end-K)];
%        else
%            delayed_signal = [A_line(abs(K)+1:end, 1); zeros(abs(K), 1)];
%        end
%        rf_data_delayed(:, idx) = delayed_signal;
%         
%     end
% end