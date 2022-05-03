clear 
close all

global DR
DR = 60;

% loading data
load('D:/ComplexNetwork/simulation_data/Data_30_delay_2.mat')
% load('D:/ComplexDataset/simulation_straight/Data_185_delay_1.mat')
% load('D:/ComplexDataset/simulation_straight/Data_396_delay_2.mat')
% load('D:/ComplexDataset/simulation_straight/Data_322_delay_1.mat')

pred_complex = readmatrix('complexpred.txt');
pred_complex = reshape(pred_complex,[128,256,2]);
pred_complex = pred_complex(:,:,1) + 1j.*pred_complex(:,:,2);
pred_complex = normalization(pred_complex);
true_complex = normalization(psf_bb(2:2:end,2:end));

x_axis = (0:dx:dx*size(true_complex,2)-dx).*1000;
z_axis = (depth/2 + (0:2*dz:2*dz*size(true_complex,1)-2*dz))*1000;

envelope = normalization(abs(psf_bb(2:2:end,2:end)));
envelope_dB = 20*log10(envelope/max(max(envelope))+eps) + DR;
figure
showimg(x_axis,z_axis,envelope_dB)
title('ground truth B-mode image')

pred_envelope = abs(pred_complex);
pred_envelope_dB = 20*log10(pred_envelope/max(max(pred_envelope))+eps) + DR;
figure
showimg(x_axis, z_axis, pred_envelope_dB)
title('predicted B-mode image')
% pred_real = readmatrix('D:/ComplexNetwork/realpred_tanh.txt');
% pred_real = reshape(pred_real,[256,256]);
pred_ang = angle(pred_complex);
true_ang = angle(true_complex);

ssim(pred_envelope,envelope,'DynamicRange',1)
ssim(pred_ang,true_ang,'DynamicRange',2*pi)
ssim(real(pred_complex),real(true_complex),'DynamicRange',2)
ssim(imag(pred_complex),imag(true_complex),'DynamicRange',2)

angle_err = abs(true_ang - pred_ang);
figure
plot(project(pred_envelope_dB,1))
hold on
plot(project(envelope_dB,1))
legend('predict','true')
figure
heatmap(abs(pred_ang), 'Colormap', hot)
title('|angle| distribution')
figure
heatmap(angle_err,'Colormap',hot)
title('angle difference')


figure
imagesc(x_axis, z_axis, angle_err<0.5)
colormap(gray(2))
axis image
title('binary angle difference')

figure
heatmap(abs(true_ang),'Colormap',hot)
title('truth |angle| distribution')
figure
heatmap(true_ang,'Colormap',hot)
title('truth angle distribution')
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
    axis image
end

function showfft(img, fs)
    Aline = img(:,256);
    Aline = fftshift(fft(Aline));
    freq = linspace(-fs/2,fs/2,length(Aline))/1e6;
    plot(freq,abs(Aline))
    xlabel('MHz')
end

