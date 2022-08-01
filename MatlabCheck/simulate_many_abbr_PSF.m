clear 
close all

iSTART = 1;
iEND = 400;
seed = 0;

rng(seed)

for data_id = iSTART:iEND
    close all
    disp(data_id)
    if ~isfile(char(sprintf('./simulation_data2/Data_%d_delay_1.mat', data_id))) || ...
       ~isfile(char(sprintf('./simulation_data2/Data_%d_delay_2.mat', data_id))) || ...
       ~isfile(char(sprintf('./simulation_data2/Data_%d_delay_3.mat', data_id))) || ...
       ~isfile(char(sprintf('./simulation_data2/Data_%d_delay_4.mat', data_id)))
         simulate_one_abbr_PSF;
    end
end
