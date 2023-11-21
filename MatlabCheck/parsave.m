function parsave(fname, psf_rf, psf_bb, speckle_rf, speckle_bb, dx, dz, depth, f0, k, bw, delay_curve)
    % save parameters in the parfor loop
    save(fname, 'psf_rf', 'psf_bb', 'speckle_rf', 'speckle_bb', 'dx', 'dz', 'depth', 'f0', 'k', 'bw', 'delay_curve')
end
