function Envelope_dB = envelope_detection(img, DR)
    if isreal(img)
        envelope = abs(hilbert(img));
    else
        envelope = abs(img);
    end
    Envelope_dB = 20*log10(envelope/max(envelope(:))+eps) + DR;