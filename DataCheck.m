clear 
close all

load('D:/ComplexDataset/simulation_straight/Data_101_delay_2.mat')
Aline_rf = speckle_rf(:,256);
Aline_bb = speckle_bb(:,256);
Aline_RF = fftshift(fft(Aline_rf));
Aline_BB = fftshift(fft(Aline_bb));
fs = 1/(2*dz/1540);
freq = linspace(-fs/2,fs/2,513);
figure
plot(freq,abs(Aline_RF))
xlabel('Hz')
figure
plot(freq,abs(Aline_BB))
xlabel('Hz')

DR=40
envelope = abs(hilbert(psf_rf));
envelope_dB = 20*log10(envelope/max(envelope(:))+eps);
figure
image(envelope_dB+DR);
colormap(gray(40))
colorbar;
title('Point targets')
xlabel('Lateral position (mm)')
ylabel('Depth (mm)')
axis image

DR=40
envelope = abs(hilbert(psf_rf));
envelope_dB = 20*log10(envelope/max(envelope(:))+eps);
figure
image(envelope_dB+DR);
colormap(gray(40))
colorbar;
title('Point targets')
xlabel('Lateral position (mm)')
ylabel('Depth (mm)')
axis image

psf_norm = speckle_rf/max(max(speckle_rf));
envelope = abs(hilbert(psf_norm));
envelope_dB = 20*log10(envelope/max(envelope(:))+eps);
figure
image(envelope_dB+DR);
colormap(gray(40))
colorbar;
title('Point targets')
xlabel('Lateral position (mm)')
ylabel('Depth (mm)')
axis image