# -*- coding: utf-8 -*-
"""
probe3_metric_projection.py -- FIRST-PRINCIPLES test of the clip-hardness kappa=1.805.

HYPOTHESIS: the constrained KL-MLE = the unconstrained MLE PROJECTED onto the feasible simplex
in the FISHER (G) metric, not the Euclidean metric. The agents' kappa=1 model used a one-shot
EUCLIDEAN projection (undershoots: 0.85 vs measured 0.69 @K7). If the G-metric projection
reproduces meas/sand 1.00->0.69 WITHOUT any kappa knob, then kappa is EXPLAINED as the
Euclidean-vs-Fisher metric anisotropy along the worst mode, and the clipping model is parameter-free.

Test: draw a_hat ~ N(w*, Cov_unc), Cov_unc=G^{-1} Sigma G^{-1} (sandwich), scaled so sigma_unc(K=2)
matches the real-core measured sigma_err(K=2). Project each draw onto {w>=0, 1^T w=1} in
  (E) Euclidean metric  min ||w-a||^2
  (G) Fisher metric     min (w-a)^T G (w-a)
Compare worst-tangent-direction std ratio sigma_proj/sigma_unc to measured meas/sand.

rho=0.40, endpoint w, n=350. English prints. Run:
  & "C:/Users/Administrator/miniconda3/python.exe" probe3_metric_projection.py
"""
import math
import numpy as np
from scipy.special import erfinv
from scipy.optimize import minimize
import sigma_err_scan as S

SIGMA, START, BW, DT = 25.0, 200.0, 12.0, 1.0
TGRID = np.arange(50.0, 2600.0 + 1e-9, DT)
TRAPZ = getattr(np, "trapezoid", None) or np.trapz
RHO, N = 0.40, 350
MEAS_MS = {2: 1.00, 3: 1.01, 4: 1.05, 5: 0.89, 6: 0.88, 7: 0.69}   # probe_turnover meas/sand
NSAMP = 1500
rng = np.random.default_rng(20260616)


def dmu(rho): return 2.0 * math.sqrt(2.0) * erfinv(rho) * SIGMA
def gauss(t, mu, s): return np.exp(-0.5 * ((t - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))


def G_Sigma(K, rho):
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus]); h /= TRAPZ(h, TGRID, axis=1)[:, None]
    w = np.linspace(1.0, 0.6, K); w /= w.sum()
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    G = np.array([[TRAPZ(h[i] * h[j] / q, TGRID) for j in range(K)] for i in range(K)])
    KS = gauss(TGRID[:, None] - TGRID[None, :], 0.0, BW) * DT
    phi = np.array([KS @ (h[i] / q) for i in range(K)])
    Eq = lambda f: TRAPZ(f * q, TGRID)
    m = np.array([Eq(phi[i]) for i in range(K)])
    Sig = np.array([[Eq(phi[i] * phi[j]) - m[i] * m[j] for j in range(K)] for i in range(K)])
    return G, Sig, w


def cov_unc(G, Sig):                       # tangent unconstrained Cov (n=1 scale)
    P = np.eye(len(G)) - np.ones_like(G) / len(G)
    Gpi = np.linalg.pinv(P @ G @ P, rcond=1e-10)
    return P @ (Gpi @ (P @ Sig @ P) @ Gpi) @ P


def A_widened(K, rho):                      # data-Hessian metric A_ij=∫ h_i h_j q^bw / q^2
    mus = np.array([START + i * dmu(rho) for i in range(K)])
    h = np.stack([gauss(TGRID, m, SIGMA) for m in mus]); h /= TRAPZ(h, TGRID, axis=1)[:, None]
    seff = math.sqrt(SIGMA ** 2 + BW ** 2)
    hb = np.stack([gauss(TGRID, m, seff) for m in mus]); hb /= TRAPZ(hb, TGRID, axis=1)[:, None]
    w = np.linspace(1.0, 0.6, K); w /= w.sum()
    q = np.maximum((w[:, None] * h).sum(0), 1e-300)
    qb = (w[:, None] * hb).sum(0)           # widened data mean E[X]
    return np.array([[TRAPZ(h[i] * h[j] * qb / q ** 2, TGRID) for j in range(K)] for i in range(K)])


def meas_se_k2():                          # real-core sigma_err(K=2,rho,n) for absolute calibration
    core = S._load_real_core(); K = 2
    mus, _ = S.peak_ages(K, RHO, SIGMA)
    src = [core.Source(f"S{i+1}", float(mus[i]), float(SIGMA), f"s{i}") for i in range(K)]
    H = core.build_source_endmembers(src, core.T_GRID); w = S.fixed_interior_w(K, "endpoint")
    rng2 = np.random.default_rng(7)
    X = np.stack([core.kde_density(core.generate_sample(w, n_grains_range=(N, N), noise_pct=0.0,
                 sources=src, rng=rng2), t_grid=core.T_GRID, bandwidth=BW) for _ in range(2000)])
    wh = core.supervised_unmix(X, H, n_iter=2000, tol=1e-13)
    d = wh - wh.mean(0); return math.sqrt(np.linalg.eigvalsh(d.T @ d / len(wh))[-1])


def project(a, w0, Gm):                    # min (w-a)^T Gm (w-a) s.t. w>=0, sum w=1
    K = len(a)
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bnds = [(0, None)] * K
    obj = lambda w: float((w - a) @ Gm @ (w - a))
    jac = lambda w: 2.0 * (Gm @ (w - a))
    r = minimize(obj, np.clip(a, 1e-6, None) / np.clip(a, 1e-6, None).sum(),
                 method="SLSQP", jac=jac, bounds=bnds, constraints=cons,
                 options={"ftol": 1e-12, "maxiter": 200})
    return r.x


def worst_std(W, v):                       # std along worst mode v
    return float(np.std((W - W.mean(0)) @ v))


se2_meas = meas_se_k2()
print(f"[calib] real-core sigma_err(K=2, rho={RHO}, n={N}) = {se2_meas:.5f}")
# calibrate n_eff from K=2
G2, Sig2, _ = G_Sigma(2, RHO)
C2 = cov_unc(G2, Sig2)
se2_n1 = math.sqrt(np.linalg.eigvalsh(C2)[-1])
n_eff = (se2_n1 / se2_meas) ** 2
print(f"[calib] sandwich se(n=1,K=2)={se2_n1:.4f} -> n_eff={n_eff:.0f} (cf measured n={N})\n")

print("%2s | %8s | %7s %7s %7s %7s | %s"
      % ("K", "sig_unc", "ratio_E", "ratio_G", "ratio_A", "MEAS", "best vs meas"))
for K in range(2, 8):
    G, Sig, w = G_Sigma(K, RHO)
    Aw = A_widened(K, RHO)
    C = cov_unc(G, Sig) / n_eff
    lam, V = np.linalg.eigh(C)
    v = V[:, -1]; sig_unc = math.sqrt(max(lam[-1], 0))
    smp = rng.multivariate_normal(w, C, size=NSAMP)
    rE = worst_std(np.array([project(a, w, np.eye(K)) for a in smp]), v) / sig_unc
    rG = worst_std(np.array([project(a, w, G) for a in smp]), v) / sig_unc
    rA = worst_std(np.array([project(a, w, Aw) for a in smp]), v) / sig_unc
    print("%2d | %8.4f | %7.3f %7.3f %7.3f %7.3f | E %+4.0f%% G %+4.0f%% A %+4.0f%%"
          % (K, sig_unc, rE, rG, rA, MEAS_MS[K],
             100 * (rE - MEAS_MS[K]) / MEAS_MS[K], 100 * (rG - MEAS_MS[K]) / MEAS_MS[K],
             100 * (rA - MEAS_MS[K]) / MEAS_MS[K]))
