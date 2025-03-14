% Regenerate Speckles which have ONLY ONE kind of scatterer distribution
Npx  = 257;
Npz  = 257;
gain = 60;
DR = 60;
soundv = 1540;
dirs =  './simulation_data2';
files = dir(fullfile(dirs,'/*.mat'));
filename = {files.name};
savepaths = {'Regenerate_4for1sharescatterer'};
% ----- scat_dist: scatterer location (x,z) -----
% scatterer distribution
Nsx = (Npx - 1)*2 + 1;
Nsz = (Npz - 1)*2 + 1;
den         = 0.05 + 0.45*rand(1);                    % scatterer's density, [0.05,0.5]
Nscat       = round(den*Nsz*Nsx);                   % number of scatterers with in a patch
scat_space  = zeros(Nsz,Nsx);                       % distribution space size = (Npz,Npx)
scat_indice = randi(numel(scat_space),[1,Nscat]);   % scatterer's location at which index
scat_space(scat_indice) = randn(Nscat,1);           % scatterer's amplitude (random normal)
scat_space = reshape(scat_space, [Nsz, Nsx]);

for ifile = 1:length(savepaths)

    savepath = savepaths{ifile};
    if ~exist(savepath, 'dir')
        mkdir(savepath) % create dirc 
    end

    parfor ii = 1:length(files)
        close all
        fprintf('Now generating... %d psf\n', ii)

        expand_scale_x = [1,1,1,1];
        expand_scale_z = [1,1,1,1];
        file_name = fullfile(dirs, filename{ii});
        data = load_file(file_name);
        psf_rf = data.psf_rf;
        psf_bb = data.psf_bb;
        dx = data.dx;
        dz = data.dz;
        depth = data.depth;
        f0 = data.f0;
        k = data.k;
        bw = data.bw;
        delay_curve = data.delay_curve;

%         
%         newfs = soundv/2/(dz*expand_scale_z(k));
%         if newfs/f0 < 2
%             warning('Sampling rate is under Nyquist frequency %.4f', newfs/f0)
%         end
%         lpf = fir1(48, f0/(newfs*expand_scale_z(k)/2))';
%     
%     
%     
        speckle_rf = conv2(scat_space, psf_rf, 'same'); % psf is a filter
        speckle_rf = speckle_rf((Nsz+1)/2 - (Npz-1)*expand_scale_z(k)/2:(Nsz+1)/2 + (Npz-1)*expand_scale_z(k)/2, (Nsx+1)/2 - (Npx-1)*expand_scale_x(k)/2:(Nsx+1)/2 + (Npx-1)*expand_scale_x(k)/2);
        speckle_rf = imresize(speckle_rf, [Npz,Npx], 'bicubic');
        speckle_bb = conv2(speckle_rf.*exp(-1j*2*pi*f0*(0:size(speckle_rf,1)-1)'./newfs), lpf, 'same');
    
        newz = depth:dz:(Npz*dz+depth)-dz;
        newx = linspace(-dx*(Npx-1)/2,dx*(Npx-1)/2,Npx);
%         psfsegz     = 16/2;                  % z-direction segmentation size, 8 lambda for top and bottom side from central point, unit in lambda
%         psf_rf = psf_rf((Npz+1)/4:3*(Npz+1)/4,:);
%         psf_rf = imresize(psf_rf, [Npz, Npx]);
%         psf_bb = conv2(speckle_rf.*exp(-1j*2*pi*f0*(0:size(psf_rf,1)-1)'./newfs), lpf, 'same');
%         newz = newz((Npz+1)/4:3*(Npz+1)/4,:);
        envelope = abs(psf_bb);
        envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);
        fig = figure('visible','off');
        subplot(121)
        image(newx*1e3, newz*1e3, envelope_dB+gain);
        colormap(gray(DR));colorbar;
        xlabel('Lateral position (mm)')
        ylabel('Depth (mm)')
        axis image;
    
        fig_name = filename{ii};
        envelope = abs(speckle_bb);
        envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);
        subplot(122)
        image(expand_scale_x(k)*newx*1e3, expand_scale_z(k)*newz*1e3, envelope_dB+gain);
        colormap(gray(DR));colorbar;
        xlabel('Lateral position (mm)')
        ylabel('Depth (mm)')
        axis image;
        parsave(fullfile(savepath,filename{ii}), psf_rf, psf_bb, speckle_rf, speckle_bb, dx', dz, depth, f0, k, bw, delay_curve);
        saveas(fig, fullfile(savepath,[fig_name(1:end-4), '.png']));
    
    end
end