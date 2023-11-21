% function pulse = rf_pulse_est(RFdata, Nc)
% % Nc: number of cepstrum data is used.
% RFdata = double(RFdata(:,128));
% figure; hold on;
% 
% y = mean(real(ifft(log(abs(fft(RFdata.*hanning(size(RFdata,1))))))), 2);
% 
% y(2:Nc) = 2*y(2:Nc);
% y(Nc+1:end) = 0;
% pulse = real(ifft(exp(fft(y))));
% 
% pulse = pulse/max(abs(pulse));
% figure
% plot(pulse, 'r', 'LineWidth', 4);
% title('Estimated pulse profile');


% Nc: number of cepstrum data is used.
BBdata = psf_bb(:,128);


% Compute the spectrum of the baseband data
spec = abs((fft(BBdata)));

% % Low-pass filter the spectrum to remove high-frequency noise
% spec_filt = smooth(spec, round(length(spec)/10));

% Compute the log spectrum of the filtered data
log_spec = log(abs(spec));

% Compute the cepstrum of the log spectrum
cepstrum = real(ifft(log_spec));

% Apply the liftering operation
cepstrum(2:Nc) = 2*cepstrum(2:Nc);
cepstrum(Nc+1:end) = 0;

% Compute the pulse profile
pulse = real(ifft(exp(fft(cepstrum))));

% Normalize the pulse profile
pulse = pulse/max(abs(pulse));

% Plot the pulse profile
figure
plot(pulse, 'r', 'LineWidth', 4);
title('Estimated pulse profile');
figure
plot(abs(exp(fft(cepstrum))));