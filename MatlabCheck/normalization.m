function y = normalization(x)
    modulus = abs(x);
    y = x./max(max(modulus));