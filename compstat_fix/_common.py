import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import dz_unmixing_experiment as core
from scipy.special import erfinv
def setup(K, rho, N, R, seed=1, sigma=25.0, bw=12.0, start=200.0):
    dmu = sigma * 2 * np.sqrt(2) * erfinv(rho); mus = start + np.arange(K) * dmu
    src = [core.Source(f"S{i}", float(mus[i]), sigma, "x") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID)
    w = np.linspace(1, 0.6, K); w /= w.sum(); rng = np.random.default_rng(seed)
    X = np.array([core.kde_density(core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0, sources=src, rng=rng),
                                   t_grid=core.T_GRID, bandwidth=bw) for _ in range(R)])
    return X, H, w
