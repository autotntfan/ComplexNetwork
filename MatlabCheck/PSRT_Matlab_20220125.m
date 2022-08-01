

clear
close all

% ============================= program start ==================================

eval('PSRT_DataAnalysis_20220113')

addpath('C:\Program Files\Prodigy\pulsesequapp\2.0')

k = EG_num + 1;

if (NameObj(1).EnProcessDataType <= 1)
    FrameNum = 1;
else
    FrameNum = NameObj(1).Frames;
end
Na = NameObj(1).Na;
DownSampleRatio = NameObj(1).DownSampleRatio;

if Overall_para.En2D == 0 %1D array
    BeamNum = (NameObj(2).FOV_E_SC - NameObj(2).FOV_S_SC + 1) * (NameObj(2).BeamRep + 1)*pulse_firing;
else %2D array
    BeamNum = NameObj(2).Focus_Num *(NameObj(2).BeamRep + 1)*pulse_firing;
end
%SampleNum = NameObj(2).SL_modified;  %% Number of samples
if (NameObj(2).BeamFormingMethod == 0)
    DataSize_InByte = BeamNum * Na * (NameObj(2).SL_modified/DownSampleRatio) * 2;
elseif (NameObj(2).BeamFormingMethod == 1)
    DataSize_InByte = BeamNum * 1 * (NameObj(2).SL_modified/DownSampleRatio) * 4 * 2;
elseif (NameObj(2).BeamFormingMethod == 2 || NameObj(2).BeamFormingMethod == 3)
    DataSize_InByte = BeamNum * Na * (NameObj(2).SL_modified/DownSampleRatio) * 4 * 2;
end

if (k > 2)
    for evt = 3:k
        if Overall_para.En2D == 0 %1D array
            BeamNum_next_evt = (NameObj(evt).FOV_E_SC - NameObj(evt).FOV_S_SC + 1) * (NameObj(evt).BeamRep + 1)*pulse_firing;
        else % 2D array
            BeamNum_next_evt = NameObj(evt).Focus_Num * (NameObj(evt).BeamRep + 1)*pulse_firing;
        end
        
        BeamNum = BeamNum + BeamNum_next_evt;
        if (NameObj(evt).BeamFormingMethod == 0)
            DataSize_InByte = DataSize_InByte + BeamNum_next_evt * Na * (NameObj(evt).SL_modified/DownSampleRatio) * 2;
        elseif (NameObj(evt).BeamFormingMethod == 1)
            DataSize_InByte = DataSize_InByte + BeamNum_next_evt * 1 * (NameObj(evt).SL_modified/DownSampleRatio) * 4 * 2;
        elseif (NameObj(evt).BeamFormingMethod == 2 || NameObj(evt).BeamFormingMethod == 3)
            DataSize_InByte = DataSize_InByte + BeamNum_next_evt * Na * (NameObj(evt).SL_modified/DownSampleRatio) * 4 * 2;
        end
    end
end
TotalData = zeros(DataSize_InByte * FrameNum, 1, 'uint8');
ScanStatus = zeros(4, 1, 'uint8');
FrameCount = 0;
EnStopWhenMemFull_Matlab = NameObj(1).EnStopWhenMemFull_Matlab;
disp([' >> ready to Start Scan'])
while (ScanStatus(1) == 0 || ScanStatus(1) == 2)
    if (isOctave == 1)
        Octave_CheckScanStatus(ScanStatus);
    else
        CheckScanStatus(ScanStatus);
    end
end

MaxSampleNum = 4096;
if (FrameCount == 0)
  pause(0.01);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%       Asynthronize-mode       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (NameObj(1).EnProcessDataType == 0)
    if (NameObj(1).Frames > 0)
        while (FrameCount < NameObj(1).Frames)
            if (isOctave == 1)
                Octave_ReceiveData(TotalData, DataSize_InByte, FrameCount, NameObj(1).Frames, NameObj(1).EnProcessDataType);
            else
                ReceiveData(TotalData, DataSize_InByte, FrameCount, NameObj(1).Frames, NameObj(1).EnProcessDataType);
            end
            FrameCount = FrameCount + 1;
            
            DownSampRatio = NameObj(1).DownSampleRatio; % downsampling ratio in baseband beamformed data
            Data_sub_starInd = 1; % data storage start index
            for ij=2:EG_num+1
                %%%%%%%%%% Check Data type (RF-data, Beamformed I-Q Data)
                if (NameObj(ij).BeamFormingMethod == 0)   %-> RF
                    
                    %%%% RF-Data
                    Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                    Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                if Overall_para.En2D == 0 % 1D array
                    slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                else
                    slinesb = EG_para(ij-1).Focus_Num*pulse_firing; % Scan Lines
                end
                    Data_sub_size = ( Sub_EG_SampleNum* Na*slinesb*Sub_EG_BeamRep*2); % Total data size of each sub-EG
                    Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                    RFfloat = typecast(Data_sub, 'int16'); % Byte array to int16
                    RFData =double(reshape(RFfloat, Sub_EG_SampleNum, Na,slinesb,Sub_EG_BeamRep));
                    %%%% RF-Data  storage
                    eval(['EG' num2str(ij-1) '=' 'RFData' ';']);
                    Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                    %                     eval(['save ' 'EG' num2str(ij-1) ' RFData;']);
                    clear RFData
                    
                    %%%%%%    Implement area    %%%%%%


                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                elseif (NameObj(ij).BeamFormingMethod == 1) %-> Line-By-Line Beamformed IQ Data
                    
                    %%%% Beamformed I-Q Data (line by line scna)
                    Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                    Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                    slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing ; % Scan Lines
                    
                    Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                    Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                    IQfloat = typecast(Data_sub, 'single');
                    IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb));
                    I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                    Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                    bbIQData_Line = I1+1i.*Q1;
                    Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                    %%%% Beamformed I-Q Data (line by line scna) storage
                    eval(['EG' num2str(ij-1) '=' 'bbIQData_Line' ';']);
                    %                     eval(['save '  'EG' num2str(ij-1) ' bbIQData_Line;']);
                    
                    %%%%%%    Implement area    %%%%%%
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                    
                else %-> Plane-Wave or Photoacoustic Beamformed IQ Data
                    
                    %%%% Beamformed I-Q Data (PlaneWave)
                    Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                    Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                    slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing ; % Scan Lines
                    
                    if slinesb==1 % plane-wave imaging
                        slinesb = Ne;
                    else
                        slinesb = slinesb;
                    end
                    
                    Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                    Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                    IQfloat = typecast(Data_sub, 'single');
                    IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb,Sub_EG_BeamRep));
                    I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                    Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                    bbIQData_Plane = I1+1i.*Q1;
                    Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                    
                    %%%% Beamformed I-Q Data (PlaneWave) storage
                    eval(['EG' num2str(ij-1) '=' 'bbIQData_Plane' ';']);
                    %                     eval(['save ' 'EG' num2str(ij-1) ' bbIQData_Plane;']);
                    
                    %%%%%%    Implement area    %%%%%%
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                end
            end
            
            if Overall_para.EnSlotSwitch==1 && NameObj(ij).BeamFormingMethod == 0 % Combine Slot switch RF data
                slotData_comb = cat(2,EG1,EG2);
                %                 eval(['save ' 'slotData_comb_Frame' num2str(FrameCount) ' slotData_comb;']);
                
                %%%%%%    Implement area    %%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            end
            
            %%%%%%%%%%%%%%%%%%%%    Cluster start    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if sum(EG_ClusterSet)>=1 && NameObj(ij).BeamFormingMethod>0
                
                clus_start_EG = min(find(EG_ClusterSet==1))-1;
                clus_end_EG = max(find(EG_ClusterSet==1));
                Total_beam = sum((EG_FOV_E_SC(1,clus_start_EG:clus_end_EG)-EG_FOV_S_SC(1,clus_start_EG:clus_end_EG)+1).*EG_BeamRep(1,clus_start_EG:clus_end_EG));
                EG_Cluster_Start2 =  EG_Cluster_Start(1,clus_start_EG:clus_end_EG);
                EG_Cluster_Interval2 =  EG_Cluster_Interval(1,clus_start_EG:clus_end_EG);
                EG_BeamRep2 = EG_BeamRep(1,clus_start_EG:clus_end_EG);
                
                for ij=clus_start_EG:clus_end_EG
                    eval(['Count' num2str(ij) '=EG_BeamRep(1,' num2str(ij) ');' ';']);
                    eval(['Frindex' num2str(ij) '=0;' ]);
                    
                end
                
                accum_frameNum = 0;
                sub_BeamRep_count = zeros(1,clus_end_EG-clus_start_EG+1);
                Iden_index2 = EG_BeamRep2;
                for iEG = 1: EG_BeamRep(clus_start_EG)/EG_Cluster_Interval(clus_start_EG) % check start cycling EG
                    Clus_interval_1 = EG_Cluster_Interval2(1);
                    Iden_index = Clus_interval_1*(iEG-1);
                    tmp_ind = find(Iden_index>=EG_Cluster_Start2); % check cycling EG number
                    
                    for Clus_ind = 1:length(tmp_ind)
                        
                        tmp_start = Clus_ind+clus_start_EG-1;
                        Clus_Rep = EG_Cluster_Interval2(Clus_ind);
                        cout_index = 0;
                        for sub_EG_ind=1:Clus_Rep
                            cout_index = 1;
                            eval(['Sub_EG=EG' num2str(tmp_start) ';' ]);
                            eval(['Frindex' num2str(tmp_start) '=' num2str(cout_index) '+ Frindex' num2str(tmp_start) ';'  ]);
                            eval(['Count' num2str(tmp_start) '=' 'Count' num2str(tmp_start) '-' num2str(cout_index) ';'  ]);
                            eval(['tmp_count=Count' num2str(tmp_start) ';' ]);
                            
                            if tmp_count>=0
                                accum_frameNum=accum_frameNum+1;
                                eval(['subFr_ind=Frindex' num2str(tmp_start) ';' ]);
                                Clus_group(:,:, accum_frameNum) =  Sub_EG(:,:,subFr_ind);
                            end
                            
                        end
                        
                    end
                    
                end
                eval(['Fr_' num2str(1) '_ClusGroup=Clus_group' ';'])
                %                 eval(['save ' 'Fr_' num2str(1) '_ClusGroup' ' Fr_' num2str(1) '_ClusGroup' ';']);
                clear Clus_group;
            end
            %%%%%%%%%%%%%%%%%%%%    Cluster end    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        if (FrameCount == NameObj(1).Frames)
%             if (isOctave == 1)
%                 Octave_StopScan(2);
%             else
%                 StopScan(2);
%             end
%             ScanStatus(1) = 2;
%         end
%         if (isOctave == 1)
%             Octave_CheckScanStatus(ScanStatus);
%         else
%             CheckScanStatus(ScanStatus);
%         end
%         if (ScanStatus(1) == 0)
%             if (isOctave == 1)
%                 Octave_StartScan();
%             else
%                 StartScan();
%             end
%             ScanStatus(1) = 1;
%         end
        end
        
    else % Frame = 0
        while ScanStatus(1) == 1
            if (isOctave == 1)
                Octave_CheckScanStatus(ScanStatus);
            else
                CheckScanStatus(ScanStatus);
            end
            if (ScanStatus(1) == 1)
                if (isOctave == 1)
                    Octave_ReceiveData(TotalData, DataSize_InByte, FrameCount, NameObj(1).Frames, NameObj(1).EnProcessDataType);
                else
                    ReceiveData(TotalData, DataSize_InByte, FrameCount, NameObj(1).Frames, NameObj(1).EnProcessDataType);
                end
                FrameCount = FrameCount + 1;
                
                DownSampRatio = NameObj(1).DownSampleRatio; % downsampling ratio in baseband beamformed data
                Data_sub_starInd = 1; % data storage start index
                for ij=2:EG_num+1
                    %%%%%%%%%% Check Data type (RF-data, Beamformed I-Q Data)
                    if (NameObj(ij).BeamFormingMethod == 0)   %-> RF
                        
                        %%%% RF-Data
                        Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                        Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                if Overall_para.En2D == 0 % 1D array
                    slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                else
                    slinesb = EG_para(ij-1).Focus_Num*pulse_firing; % Scan Lines
                end
                        Data_sub_size = ( Sub_EG_SampleNum* Na*slinesb*Sub_EG_BeamRep*2); % Total data size of each sub-EG
                        Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                        RFfloat = typecast(Data_sub, 'int16'); % Byte array to int16
                        RFData =double(reshape(RFfloat, Sub_EG_SampleNum, Na,slinesb,Sub_EG_BeamRep));
                        %%%% RF-Data  storage
                        eval(['EG' num2str(ij-1) '=' 'RFData' ';']);
                        Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                        %                     eval(['save ' 'EG' num2str(ij-1) ' RFData;']);
                        clear RFData
                        
                        %%%%%%    Implement area    %%%%%%
                        
                        
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                    elseif (NameObj(ij).BeamFormingMethod == 1) %-> Line-By-Line Beamformed IQ Data
                        
                        %%%% Beamformed I-Q Data (line by line scna)
                        Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                        Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                        slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                        
                        Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                        Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                        IQfloat = typecast(Data_sub, 'single');
                        IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb));
                        I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                        Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                        bbIQData_Line = I1+1i.*Q1;
                        Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                        %%%% Beamformed I-Q Data (line by line scna) storage
                        eval(['EG' num2str(ij-1) '=' 'bbIQData_Line' ';']);
                        %                     eval(['save '  'EG' num2str(ij-1) ' bbIQData_Line;']);
                        
                        %%%%%%    Implement area    %%%%%%
                        
                        
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                    else %-> Plane-Wave or Photoacoustic Beamformed IQ Data
                        
                        %%%% Beamformed I-Q Data (PlaneWave)
                        Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                        Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                        slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                        
                        if slinesb==1 % plane-wave imaging
                            slinesb = Ne;
                        else
                            slinesb = slinesb;
                        end
                        
                        Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                        Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                        IQfloat = typecast(Data_sub, 'single');
                        IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb,Sub_EG_BeamRep));
                        I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                        Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                        bbIQData_Plane = I1+1i.*Q1;
                        Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                        
                        %%%% Beamformed I-Q Data (PlaneWave) storage
                        eval(['EG' num2str(ij-1) '=' 'bbIQData_Plane' ';']);
                        %                     eval(['save ' 'EG' num2str(ij-1) ' bbIQData_Plane;']);
                        
                        %%%%%%    Implement area    %%%%%%
                        
                        
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        
                    end
                end
                
                if Overall_para.EnSlotSwitch==1 && NameObj(ij).BeamFormingMethod == 0 % Combine Slot switch RF data
                    slotData_comb = cat(2,EG1,EG2);
                    %                 eval(['save ' 'slotData_comb_Frame' num2str(FrameCount) ' slotData_comb;']);
                    
                    %%%%%%    Implement area    %%%%%%
                    
                    
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    
                end
                
                
                %%%%%%%%%%%%%%%%%%    Cluster start    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                if sum(EG_ClusterSet)>=1 && NameObj(ij).BeamFormingMethod>0
                    
                    clus_start_EG = min(find(EG_ClusterSet==1))-1;
                    clus_end_EG = max(find(EG_ClusterSet==1));
                    Total_beam = sum((EG_FOV_E_SC(1,clus_start_EG:clus_end_EG)-EG_FOV_S_SC(1,clus_start_EG:clus_end_EG)+1).*EG_BeamRep(1,clus_start_EG:clus_end_EG));
                    EG_Cluster_Start2 =  EG_Cluster_Start(1,clus_start_EG:clus_end_EG);
                    EG_Cluster_Interval2 =  EG_Cluster_Interval(1,clus_start_EG:clus_end_EG);
                    EG_BeamRep2 = EG_BeamRep(1,clus_start_EG:clus_end_EG);
                    
                    for ij=clus_start_EG:clus_end_EG
                        eval(['Count' num2str(ij) '=EG_BeamRep(1,' num2str(ij) ');' ';']);
                        eval(['Frindex' num2str(ij) '=0;' ]);
                        
                    end
                    
                    accum_frameNum = 0;
                    sub_BeamRep_count = zeros(1,clus_end_EG-clus_start_EG+1);
                    Iden_index2 = EG_BeamRep2;
                    for iEG = 1: EG_BeamRep(clus_start_EG)/EG_Cluster_Interval(clus_start_EG) % check start cycling EG
                        Clus_interval_1 = EG_Cluster_Interval2(1);
                        Iden_index = Clus_interval_1*(iEG-1);
                        tmp_ind = find(Iden_index>=EG_Cluster_Start2); % check cycling EG number
                        
                        for Clus_ind = 1:length(tmp_ind)
                            
                            tmp_start = Clus_ind+clus_start_EG-1;
                            Clus_Rep = EG_Cluster_Interval2(Clus_ind);
                            cout_index = 0;
                            for sub_EG_ind=1:Clus_Rep
                                cout_index = 1;
                                eval(['Sub_EG=EG' num2str(tmp_start) ';' ]);
                                eval(['Frindex' num2str(tmp_start) '=' num2str(cout_index) '+ Frindex' num2str(tmp_start) ';'  ]);
                                eval(['Count' num2str(tmp_start) '=' 'Count' num2str(tmp_start) '-' num2str(cout_index) ';'  ]);
                                eval(['tmp_count=Count' num2str(tmp_start) ';' ]);
                                
                                if tmp_count>=0
                                    accum_frameNum=accum_frameNum+1;
                                    eval(['subFr_ind=Frindex' num2str(tmp_start) ';' ]);
                                    Clus_group(:,:, accum_frameNum) =  Sub_EG(:,:,subFr_ind);
                                end
                                
                            end
                            
                        end
                        
                    end
                    eval(['Fr_' num2str(1) '_ClusGroup=Clus_group' ';'])
                    %                 eval(['save ' 'Fr_' num2str(1) '_ClusGroup' ' Fr_' num2str(1) '_ClusGroup' ';']);
                    clear Clus_group;
                end
                %%%%%%%%%%%%%%%%%%    Cluster end    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
            
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%     synthronize-mode     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
elseif (NameObj(1).EnProcessDataType == 1)
    while ScanStatus(1) == 1
        if (isOctave == 1)
            Octave_ReceiveData(TotalData, DataSize_InByte, FrameCount, 0, NameObj(1).EnProcessDataType);
        else
            ReceiveData(TotalData, DataSize_InByte, FrameCount, 0, NameObj(1).EnProcessDataType);
        end
        ScanStatus(1) = 0;
        FrameCount = FrameCount + 1;
        
        Data_sub_starInd=1;
        for ij=2:EG_num+1
            
            %%%%%%%%%% Check Data type (RF-data, Beamformed I-Q Data)
            if (NameObj(ij).BeamFormingMethod == 0)   %-> RF
                
                %%%% RF-Data
                Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio);  % Samples
                if Overall_para.En2D == 0 % 1D array
                    slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                else %2D array
                    slinesb = EG_para(ij-1).Focus_Num*pulse_firing; % Scan Lines
                end
                Data_sub_size = (Sub_EG_SampleNum*Na*slinesb*Sub_EG_BeamRep*2); % Total data size of each sub-EG
                Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                RFfloat = typecast(Data_sub, 'int16'); % Byte array to int16
                RFData =double(reshape(RFfloat,Sub_EG_SampleNum, Na,slinesb,Sub_EG_BeamRep));
                Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                %%%% RF-Data  storage
                eval(['EG' num2str(ij-1) '=' 'RFData' ';']);
                %         eval(['save ' 'Fr' num2str(FrameCount) '_' 'EG' num2str(ij-1) ' RFData;']);
                clear RFData;
                
                %%%%%%    Implement area    %%%%%%



                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            elseif (NameObj(ij).BeamFormingMethod == 1) %-> Line-By-Line Beamformed IQ Data
                
                %%%% Beamformed I-Q Data (line by line scna)
                Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing ; % Scan Lines
                
                Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                IQfloat = typecast(Data_sub, 'single');
                IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb));
                I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                bbIQData_Line = I1+1i.*Q1;
                Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                %%%% Beamformed I-Q Data (line by line scna) storage
                eval(['EG' num2str(ij-1) '=' 'bbIQData_Line' ';']);
                %         eval(['save ' 'Fr' num2str(FrameCount) '_' 'EG' num2str(ij-1) ' bbIQData_Line;']);
                
                %%%%%%    Implement area    %%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            else %-> Plane-Wave or Photoacoustic Beamformed IQ Data
                
                %%%% Beamformed I-Q Data (PlaneWave)
                Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                
                if slinesb==1 % plane-wave imaging
                    slinesb = Ne;
                else
                    slinesb = slinesb;
                end
                
                Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                IQfloat = typecast(Data_sub, 'single');
                IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb,Sub_EG_BeamRep));
                I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                bbIQData_Plane = I1+1i.*Q1;
                Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                %%%% Beamformed I-Q Data (PlaneWave) storage
                eval(['EG' num2str(ij-1) '=' 'bbIQData_Plane' ';']);
                %         eval(['save ' 'Fr' num2str(FrameCount) '_' 'EG' num2str(ij-1) ' bbIQData_Plane;']);
                
                %%%%%%    Implement area    %%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%   Cluster start   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if sum(EG_ClusterSet)>=1 && NameObj(ij).BeamFormingMethod>0
            
            clus_start_EG = min(find(EG_ClusterSet==1))-1;
            clus_end_EG = max(find(EG_ClusterSet==1));
            Total_beam = sum((EG_FOV_E_SC(1,clus_start_EG:clus_end_EG)-EG_FOV_S_SC(1,clus_start_EG:clus_end_EG)+1).*EG_BeamRep(1,clus_start_EG:clus_end_EG));
            EG_Cluster_Start2 =  EG_Cluster_Start(1,clus_start_EG:clus_end_EG);
            EG_Cluster_Interval2 =  EG_Cluster_Interval(1,clus_start_EG:clus_end_EG);
            EG_BeamRep2 = EG_BeamRep(1,clus_start_EG:clus_end_EG);
            
            for ij=clus_start_EG:clus_end_EG
                eval(['Count' num2str(ij) '=EG_BeamRep(1,' num2str(ij) ');' ';']);
                eval(['Frindex' num2str(ij) '=0;' ]);
                
            end
            
            accum_frameNum = 0;
            sub_BeamRep_count = zeros(1,clus_end_EG-clus_start_EG+1);
            Iden_index2 = EG_BeamRep2;
            for iEG = 1: EG_BeamRep(clus_start_EG)/EG_Cluster_Interval(clus_start_EG) % check start cycling EG
                Clus_interval_1 = EG_Cluster_Interval2(1);
                Iden_index = Clus_interval_1*(iEG-1);
                tmp_ind = find(Iden_index>=EG_Cluster_Start2); % check cycling EG number
                
                for Clus_ind = 1:length(tmp_ind)
                    
                    tmp_start = Clus_ind+clus_start_EG-1;
                    Clus_Rep = EG_Cluster_Interval2(Clus_ind);
                    cout_index = 0;
                    for sub_EG_ind=1:Clus_Rep
                        cout_index = 1;
                        eval(['Sub_EG=EG' num2str(tmp_start) ';' ]);
                        eval(['Frindex' num2str(tmp_start) '=' num2str(cout_index) '+ Frindex' num2str(tmp_start) ';'  ]);
                        eval(['Count' num2str(tmp_start) '=' 'Count' num2str(tmp_start) '-' num2str(cout_index) ';'  ]);
                        eval(['tmp_count=Count' num2str(tmp_start) ';' ]);
                        
                        if tmp_count>=0
                            accum_frameNum=accum_frameNum+1;
                            eval(['subFr_ind=Frindex' num2str(tmp_start) ';' ]);
                            Clus_group(:,:, accum_frameNum) =  Sub_EG(:,:,subFr_ind);
                        end
                        
                    end
                    
                end
                
            end
            eval(['Fr_' num2str(FrameCount) '_ClusGroup=Clus_group' ';'])
            %     eval(['save ' 'Fr_' num2str(FrameCount) '_ClusGroup' ' Fr_' num2str(FrameCount) '_ClusGroup' ';']);
            clear Clus_group;
        end
        %%%%%%%%%%%%%%%%%%%%   Cluster end   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (FrameCount == NameObj(1).Frames)
            if (isOctave == 1)
                Octave_StopScan(2);
            else
                StopScan(2);
            end
            ScanStatus(1) = 2;
        end
        if (isOctave == 1)
            Octave_CheckScanStatus(ScanStatus);
        else
            CheckScanStatus(ScanStatus);
        end
        if (ScanStatus(1) == 0)
            if (isOctave == 1)
                Octave_StartScan();
            else
                StartScan();
            end
            ScanStatus(1) = 1;
        end
        
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%        Process at Once       %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
else
    if (isOctave == 1)
        Octave_ReceiveData_One_Shot(TotalData, DataSize_InByte, NameObj(1).Frames, EnStopWhenMemFull_Matlab);
    else
        ReceiveData_One_Shot(TotalData, DataSize_InByte, NameObj(1).Frames, EnStopWhenMemFull_Matlab);
    end
    
    fr_index=0;
    Data_sub_starInd = 1;
    DownSampRatio = NameObj(1).DownSampleRatio; % downsampling ratio in baseband beamformed data
    for frr = 1 : NameObj(1).Frames
        fr_index = fr_index+1;
        for ij=2:EG_num+1
            
            %%%%%%%%%% Check Data type (RF-data, Beamformed I-Q Data)
            if (NameObj(ij).BeamFormingMethod == 0)   %-> RF
                
                %%%% RF-Data
                Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio);  % Samples
                if Overall_para.En2D == 0 % 1D array
                    slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                else %2D array
                    slinesb = EG_para(ij-1).Focus_Num*pulse_firing; % Scan Lines
                end
                Data_sub_size = (Sub_EG_SampleNum*Na* slinesb*Sub_EG_BeamRep*2); % Total data size of each sub-EG
                Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                RFfloat = typecast(Data_sub, 'int16'); % Byte array to int16
                RFData =double(reshape(RFfloat, Sub_EG_SampleNum, Na,slinesb,Sub_EG_BeamRep));
                eval(['EG' num2str(ij-1) '=' 'RFData' ';']);
                Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                %%%% RF-Data  storage
                %             eval(['save ' 'Fr' num2str(fr_index) '_' 'EG' num2str(ij-1) ' RFData;']);
                clear RFData;
                
                %%%%%%    Implement area    %%%%%%


                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            elseif (NameObj(ij).BeamFormingMethod == 1) %-> Line-By-Line Beamformed IQ Data
                
                %%%% Beamformed I-Q Data (line by line scna)
                Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1) ; % Scan Lines
                
                Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                IQfloat = typecast(Data_sub, 'single');
                IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb));
                I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                bbIQData_Line = I1+1i.*Q1;
                Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                %%%% Beamformed I-Q Data (line by line scna) storage
                eval(['EG' num2str(ij-1) '=' 'bbIQData_Line' ';']);
                %             eval(['save ' 'Fr' num2str(fr_index) '_' 'EG' num2str(ij-1) ' bbIQData_Line;']);
                
                %%%%%%   Implement area    %%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            else %-> Plane-Wave or Photoacoustic Beamformed IQ Data
                
                %%%% Beamformed I-Q Data (PlaneWave)
                Sub_EG_BeamRep = EG_BeamRep(ij-1); %  Repeated firing
                Sub_EG_SampleNum = (NameObj(ij).SL_modified/DownSampleRatio); % Samples
                slinesb = (NameObj(ij).FOV_E_SC - NameObj(ij).FOV_S_SC + 1)*pulse_firing; % Scan Lines
                
                if slinesb==1 % plane-wave imaging
                    slinesb = Ne;
                else
                    slinesb = slinesb;
                end
                
                Data_sub_size = ( Sub_EG_SampleNum* slinesb*Sub_EG_BeamRep*4*2); % Total data size of each sub-EG
                Data_sub = TotalData(Data_sub_starInd:Data_sub_starInd+Data_sub_size-1);
                IQfloat = typecast(Data_sub, 'single');
                IQData =double (reshape(IQfloat,(NameObj(ij).SL_modified/DownSampleRatio)*2,slinesb,Sub_EG_BeamRep));
                I1 = (IQData(1:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                Q1 = (IQData(2:2:(NameObj(2).SL_modified/DownSampleRatio)*2,:,:));
                bbIQData_Plane = I1+1i.*Q1;
                Data_sub_starInd = Data_sub_starInd + Data_sub_size;
                %%%% Beamformed I-Q Data (PlaneWave) storage
                eval(['EG' num2str(ij-1) '=' 'bbIQData_Plane' ';']);
                %             eval(['save ' 'Fr' num2str(fr_index) '_' 'EG' num2str(ij-1) ' bbIQData_Plane;']);
                
                %%%%%%    Implement area    %%%%%%
                
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
            end
        end
        
        if Overall_para.EnSlotSwitch==1 && NameObj(ij).BeamFormingMethod == 0 % Combine Slot switch RF data
            slotData_comb = cat(2,EG1,EG2);
            %                 eval(['save ' 'slotData_comb_Frame' num2str(FrameCount) ' slotData_comb;']);
            
            %%%%%%    Implement area    %%%%%%
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
        
        %%%%%%%%%%%%%%%%%%%%   Cluster start   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if sum(EG_ClusterSet)>=1 && NameObj(ij).BeamFormingMethod>0
            
            clus_start_EG = min(find(EG_ClusterSet==1))-1;
            clus_end_EG = max(find(EG_ClusterSet==1));
            Total_beam = sum((EG_FOV_E_SC(1,clus_start_EG:clus_end_EG)-EG_FOV_S_SC(1,clus_start_EG:clus_end_EG)+1).*EG_BeamRep(1,clus_start_EG:clus_end_EG));
            EG_Cluster_Start2 =  EG_Cluster_Start(1,clus_start_EG:clus_end_EG);
            EG_Cluster_Interval2 =  EG_Cluster_Interval(1,clus_start_EG:clus_end_EG);
            EG_BeamRep2 = EG_BeamRep(1,clus_start_EG:clus_end_EG);
            
            for ij=clus_start_EG:clus_end_EG
                eval(['Count' num2str(ij) '=EG_BeamRep(1,' num2str(ij) ');' ';']);
                eval(['Frindex' num2str(ij) '=0;' ]);
            end
            
            accum_frameNum = 0;
            sub_BeamRep_count = zeros(1,clus_end_EG-clus_start_EG+1);
            Iden_index2 = EG_BeamRep2;
            for iEG = 1: EG_BeamRep(clus_start_EG)/EG_Cluster_Interval(clus_start_EG) % check start cycling EG
                Clus_interval_1 = EG_Cluster_Interval2(1);
                Iden_index = Clus_interval_1*(iEG-1);
                tmp_ind = find(Iden_index>=EG_Cluster_Start2); % check cycling EG number
                
                for Clus_ind = 1:length(tmp_ind)
                    
                    tmp_start = Clus_ind+clus_start_EG-1;
                    Clus_Rep = EG_Cluster_Interval2(Clus_ind);
                    cout_index = 0;
                    for sub_EG_ind=1:Clus_Rep
                        cout_index = 1;
                        eval(['Sub_EG=EG' num2str(tmp_start) ';' ]);
                        eval(['Frindex' num2str(tmp_start) '=' num2str(cout_index) '+ Frindex' num2str(tmp_start) ';'  ]);
                        eval(['Count' num2str(tmp_start) '=' 'Count' num2str(tmp_start) '-' num2str(cout_index) ';'  ]);
                        eval(['tmp_count=Count' num2str(tmp_start) ';' ]);
                        
                        if tmp_count>=0
                            accum_frameNum=accum_frameNum+1;
                            eval(['subFr_ind=Frindex' num2str(tmp_start) ';' ]);
                            Clus_group(:,:, accum_frameNum) =  Sub_EG(:,:,subFr_ind);
                        end
                    end
                end
                
            end
            eval(['Fr_' num2str(frr) '_ClusGroup=Clus_group' ';'])
            %         eval(['save ' 'Fr_' num2str(frr) '_ClusGroup' ' Fr_' num2str(frr) '_ClusGroup' ';']);
            clear Clus_group;
        end
        %%%%%%%%%%%%%%%%%%%%   Cluster end   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%    Implement area    %%%%%%
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
% clear TotalData;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%% end of PSRT to MATLAB
% clear