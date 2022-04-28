function Envelope_dB = envelope_detection(img, DR)
    envelope = abs(hilbert(img));
    Envelope_dB = 20*log10(envelope/max(envelope(:))+eps) + DR;