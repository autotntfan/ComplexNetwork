clear 
close all

load('D:/ComplxDataset/simulation_straight/Data_1_delay_1.mat')
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
