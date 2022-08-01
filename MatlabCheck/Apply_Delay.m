function rf_data_delayed = Apply_Delay(rf_data, delay_curve)
  [Nsample, Nelement] = size(rf_data);
  rf_data_delayed = zeros(Nsample, Nelement);
    for idx = 1:Nelement
       A_line = rf_data(:, idx);
       K = delay_curve(idx);
       if K > 0
           delayed_signal = [zeros(K, 1); A_line(1:end-K)]; % the first k terms are zeros and the others (N-K) are delay by K terms
       else
           delayed_signal = [A_line(abs(K)+1:end, 1); zeros(abs(K), 1)]; % the last k terms are zeros and the others (N-K) are advanced by K terms
       end
       rf_data_delayed(:, idx) = delayed_signal;
    end

end


