clear
close all
Npx   = 257;               % final RF/baseband data size after resize for x direction
Npz  = 257;               % final RF/baseband data size after resize for z direction
gain = 60;
DR = 60;
soundv = 1540;
dirs =  './simulation_data_SharePSFandScatterer';
files = dir(fullfile(dirs,'/*.mat'));
filename = {files.name};
savepaths = {'Regenerate_4for1SameChannelDataShareScatterer'};

for ifile = 1:length(savepaths)

    savepath = savepaths{ifile};
    if ~exist(savepath, 'dir')
        mkdir(savepath) % create dirc 
    end

    for ii = 1:length(files)
        close all
        fprintf('Now generating... %d psf\n', ii)
        k = mod(ii,4);
        if k == 1
            % generate scatterer distribution for each set
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
            if ~exist('temp', 'dir')
                mkdir('temp') % create dir
            end
            save(fullfile('temp','scat_space.mat'), 'scat_space');
        else
            % use saved scatterer distribution
            load(fullfile('temp','scat_space.mat'));
            if k == 0
                k = 4;
            end
        end
        file_name = ['Data_', num2str(ceil(ii/4)), '_delay_', num2str(k), '.mat'];
        % parameters
        expand_scale_z = [1,1,1,1];
        expand_scale_x = [1,1,1,1];
        data = load_file(fullfile(dirs, file_name));
        psf_rf = data.psf_rf;
        psf_bb = data.psf_bb;
        dx = data.dx;
        dz = data.dz;
        depth = data.depth;
        f0 = data.f0;
        k = data.k;
        bw = data.bw;
        delay_curve = data.delay_curve;

        newfs = soundv/2/(dz*expand_scale_z(k));
        if newfs/f0 < 2
            warning('Sampling rate is under Nyquist frequency %.4f', newfs/f0)
        end
        lpf = fir1(48, f0/(newfs*expand_scale_z(k)/2))';

        speckle_rf = conv2(scat_space, psf_rf, 'same'); % psf is a filter
        speckle_rf = speckle_rf((Nsz+1)/2 - (Npz-1)*expand_scale_z(k)/2:(Nsz+1)/2 + (Npz-1)*expand_scale_z(k)/2, (Nsx+1)/2 - (Npx-1)*expand_scale_x(k)/2:(Nsx+1)/2 + (Npx-1)*expand_scale_x(k)/2);
        speckle_rf = imresize(speckle_rf, [Npz,Npx], 'bilinear');
        speckle_bb = conv2(speckle_rf.*exp(-1j*2*pi*f0*(0:size(speckle_rf,1)-1)'./newfs), lpf, 'same');
    
        newz = depth:dz:(Npz*dz+depth)-dz;
        newx = linspace(-dx*(Npx-1)/2,dx*(Npx-1)/2,Npx);
        envelope = abs(psf_bb);
        envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);
        fig = figure('visible','off');
        subplot(121)
        image(newx*1e3, newz*1e3, envelope_dB+gain);
        colormap(gray(DR));colorbar;
        xlabel('Lateral position (mm)')
        ylabel('Depth (mm)')
        axis image;
    
        envelope = abs(speckle_bb);
        envelope_dB = 20*log10(envelope/max(envelope, [], 'all')+eps);
        subplot(122)
        image(expand_scale_x(k)*newx*1e3, expand_scale_z(k)*newz*1e3, envelope_dB+gain);
        colormap(gray(DR));colorbar;
        xlabel('Lateral position (mm)')
        ylabel('Depth (mm)')
        axis image;
        parsave(fullfile(savepath,file_name), psf_rf, psf_bb, speckle_rf, speckle_bb, dx', dz, depth, f0, k, bw, delay_curve);
        saveas(fig, fullfile(savepath,[file_name(1:end-4), '.png']));
    
    end
end