clear 
close all
global DR
DR=60;
% loading data
% load('D:/ComplexDataset/simulation_straight/Data_179_delay_2.mat')
% load('D:/ComplexDataset/simulation_straight/Data_185_delay_1.mat')
% load('D:/ComplexDataset/simulation_straight/Data_396_delay_2.mat')
load('D:/ComplexDataset/simulation_straight/Data_322_delay_1.mat')
envelope = Envelope(psf_rf,DR);
image(envelope)
colormap(gray(DR))
fs = 1/(2*dz/1540);
figure
showfft(psf_rf,fs)
figure
image(20*log10(abs(psf_bb)/max(max(abs(psf_bb)))+eps) + DR)
colormap(gray(DR))
figure
showfft(abs(psf_bb),fs)
figure
showfft(speckle_rf,fs)
figure
showfft(abs(speckle_bb),fs)
figure
image(envelope(1:200,:))
colormap(gray(DR))
axis image
figure
showfft(psf_rf(1:200,:),fs)

pred_complex = readmatrix('D:/ComplexNetwork/complexpred.txt');
pred_complex = reshape(pred_complex,[128,256,2]);
pred_complex = pred_complex(:,:,1) + 1j.*pred_complex(:,:,2);

x_axis = (0:dx:dx*size(psf_bb,2)-dx).*1000;
z_axis = (depth/2 + (0:dz:dz*size(psf_bb,1)-dz))*1000;
envelope_rf = Envelope(psf_rf,DR);
figure
showimg(x_axis,z_axis,envelope_rf)

pred_real = readmatrix('D:/ComplexNetwork/realpred_tanh.txt');
pred_real = reshape(pred_real,[256,256]);



factors = [1 2 4 8];
% for ii = 1:length(factors)
%     factor = factors(ii);
%     bb = psf_bb(1:factor:end,:);
%     rf = psf_rf(1:factor:end,:);
%     fs = 1/(2*factor*dz/1540);
%     x_axis = 0:dx:dx*size(bb,2)-dx;
%     z_axis = depth/2 + (0:factor*dz:factor*dz*size(bb,1)-factor*dz);
%     envelope = abs(hilbert(rf));
%     envelope_dB_rf = 20*log10(envelope/max(envelope(:))+eps) + DR;
%     figure(1)
%     subplot(2,2,ii)
%     showimg(x_axis, z_axis, envelope_dB_rf)
%     title(['psf rf factor ' string(factor)])
%     
%     envelope = abs(bb);
%     envelope_dB_bb = 20*log10(envelope/max(envelope(:))+eps) + DR;
%     figure(2)
%     subplot(2,2,ii)
%     showimg(x_axis, z_axis, envelope_dB_bb)
%     title(['psf bb factor ' string(factor)])
%     
%     figure(3)
%     subplot(2,2,ii)
%     showfft(rf, fs)
%     title(['downsample spectrum rf factor ' string(factor)])
%     figure(4)
%     subplot(2,2,ii)
%     showfft(bb, fs)
%     title(['downsample spectrum bb factor ' string(factor)])
% end
% img_speckle = speckle_bb(2:4:end,2:2:end);
% img_psf = psf_bb(2:4:end,2:2:end); 
% % speckle
% x_axis = (0:2*dx:2*dx*size(img_speckle,2)-2*dx).*1000;
% z_axis = (depth/2 + (0:4*dz:4*dz*size(img_speckle,1)-4*dz))*1000;
% figure
% envelope = abs(img_speckle);
% envelope_dB_bb = 20*log10(envelope/max(envelope(:))+eps) + DR;
% showimg(x_axis, z_axis, envelope_dB_bb)
% title('Speckle')
% % psf
% figure
% envelope_truth = abs(img_psf);
% envelope_truth = envelope_truth/max(envelope_truth(:));
% envelope_dB_bb = 20*log10(envelope_truth+eps) + DR;
% showimg(x_axis, z_axis, envelope_dB_bb)
% title('Ground truth PSF')
% 
% 
% % predicted PSF (RF)
% z_axis = (depth/2 + (0:2*dz:2*dz*size(pred_real,1)-2*dz))*1000;
% figure
% envelope_RF = abs(hilbert(pred_real));
% envelope_RF = envelope_RF/max(envelope_RF(:));
% envelope_RF_dB = 20*log10(envelope_RF+eps) + DR;
% showimg(x_axis, z_axis, envelope_RF_dB)
% title('Predicted PSF (RF)')
% 
% % predicted PSF (BB)
% z_axis = (depth/2 + (0:4*dz:4*dz*size(pred_complex,1)-4*dz))*1000;
% figure
% envelope_BB = abs(pred_complex);
% envelope_BB = envelope_BB/max(envelope_BB(:));
% envelope_BB_dB  = 20*log10(envelope_BB+eps) + DR;
% showimg(x_axis, z_axis, envelope_BB_dB)
% title('Predicted PSF (BB)')
% ssim(envelope_BB,envelope_truth,'DynamicRange',1)
% 
% lateral_project = max(envelope_BB_dB,[],1);
% axial_project = max(envelope_BB_dB,[],2);
% figure
% plot(x_axis,lateral_project)
% xlabel('lateral position (mm)')
% ylabel('intensity (dB)')
% title('lateral projection')
% set(gca,'FontSize',16)
% figure
% plot(z_axis,axial_project)
function showimg(xaxis, zaxis, img)
    global DR
    image(xaxis, zaxis, img);
    colormap(gray(DR))
    colorbar;
    title('speckle targets', 'FontSize', 14)
    xlabel('Lateral position (mm)', 'FontSize', 12)
    ylabel('Depth (mm)', 'FontSize', 12)
    xticks('auto')
    yticks('auto')
    axis square
end

function showfft(img, fs)
    Aline = img(:,256);
    Aline = fftshift(fft(Aline));
    freq = linspace(-fs/2,fs/2,length(Aline))/1e6;
    plot(freq,abs(Aline))
    xlabel('MHz')
end

function Envelope_dB = Envelope(img, DR)
    envelope = abs(hilbert(img));
    Envelope_dB = 20*log10(envelope/max(envelope(:))+eps) + DR;
end