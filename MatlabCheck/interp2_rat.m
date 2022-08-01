function OUT = interp2_rat(X, RAT1, RAT2)
    
    [up1, down1] = numden(sym(rats(RAT1)));
    [up2, down2] = numden(sym(rats(RAT2))); 
    while up1 > 128
        up1 = up1/2;
        down1 = down1/2;
    end
    while up2 > 128
        up2 = up2/2;
        down2 = down2/2;
    end
    up1 = int32(up1);
    up2 = int32(up2);
    down1 = int32(down1);
    down2 = int32(down2);
    
    [size1, size2] = size(X);
    
    cent1 = round((size1+1)/2);
    cent2 = round((size2+1)/2);
    
    SIZE1 = int32(up1*size1);
    SIZE2 = int32(up2*size2);
    
%     M = zeros(size1,1);
%     M(cent1) = 1;
%     dw = angle(fft(M));
%     ds = angle(ifft(M, SIZE1));
%     X_up = zeros(SIZE1, size2);
%     for idx = 1:size2
%         X_fft = fft(X(:, idx));
%         mag = abs(X_fft);
%         phi = angle(X_fft);
%         y = ifft(fftshift(mag.*exp(1i*(phi-dw))), SIZE1);
%         X_up(:, idx) = fftshift(abs(y).*exp(1i*(angle(y)-ds)));
%     end
    X_up = imresize(X, [SIZE1, SIZE2]);
    
    CENT1 = round((SIZE1+1)/2);
    CENT2 = round((SIZE2+1)/2);
    if down1 > 1
        X_up = [flipud(X_up(CENT1:-down1:1, :)); X_up(CENT1+down1:down1:end, :)];
    end
    if down2 > 1
        X_up = [fliplr(X_up(:, CENT2:-down2:1)) X_up(:, CENT2+down2:down2:end)];
    end
%     [size1, size2] = size(X_up);
%     if ~mod(size1,2)
%         X_up = [X_up; zeros(1,size2)];
%     end
%     if ~mod(size2,2)
%         X_up = [X_up zeros(size1, 1)];
%     end
    if isreal(X)
        OUT = real(X_up);
    else
        OUT = X_up;
    end
end