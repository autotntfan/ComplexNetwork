function parsave_diffc(fname, psf_rf, psf_bb, speckle_rf, speckle_bb, dx, dz, depth, f0, k, bw, soundv, beamformv)
    % save parameters in the parfor loop
    save(fname, 'psf_rf', 'psf_bb', 'speckle_rf', 'speckle_bb', 'dx', 'dz', 'depth', 'f0', 'k', 'bw', 'soundv', 'beamformv')
end