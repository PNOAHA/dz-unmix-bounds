"""
D3 part-1 self-check (ASCII-only): the KDE adds EXACTLY bw^2 to the empirical
density's variance, sample by sample, deterministically.

Claim: Let ages a_1..a_N be any fixed sample. Let X = (1/N) sum_i N(a_i, bw^2)
(area-normalized Gaussian KDE). Then
   Var_X(t) := E_X[t^2] - (E_X[t])^2 = (sample variance of a_i) + bw^2,
where E_X is the integral against the density X(t). The +bw^2 is N-independent
and does NOT depend on the realization beyond the sample variance term.

We verify numerically for several random samples and bandwidths that
   Var_X - samplevar  ==  bw^2   to float precision (analytic identity).

Also verify the mean is unchanged: E_X[t] = sample mean (KDE is unbiased in mean).
"""
import numpy as np

rng = np.random.default_rng(0)

def kde_moments(ages, bw, grid):
    # build area-normalized KDE on a fine grid, return mean and variance by quadrature
    a = ages[:,None]
    dens = np.exp(-0.5*((grid[None,:]-a)/bw)**2).sum(axis=0)
    dens /= (bw*np.sqrt(2*np.pi))*len(ages)
    dt = grid[1]-grid[0]
    Z = dens.sum()*dt
    dens = dens/Z
    m1 = (grid*dens).sum()*dt
    m2 = (grid**2*dens).sum()*dt
    return m1, m2 - m1**2

print("Test: Var_X - samplevar == bw^2  (should be ~0 to quadrature precision)")
for trial in range(5):
    N = rng.integers(20, 200)
    ages = rng.normal(500, 40, size=N)
    bw = rng.uniform(5, 30)
    lo, hi = ages.min()-12*bw, ages.max()+12*bw
    grid = np.linspace(lo, hi, 400000)
    m1, varX = kde_moments(ages, bw, grid)
    svar = ages.var()            # population (1/N) sample variance
    smean = ages.mean()
    print(f"  N={N:4d} bw={bw:6.3f}: Var_X={varX:12.6f} samplevar={svar:12.6f} "
          f"diff-bw^2={varX-svar-bw**2: .3e}  mean_err={m1-smean: .3e}")

# Now the population version used in the theorem: X = p = sum_k w*_k N(mu_k, sigma_eff^2)
# Its variance = Var_w*(mu) + sigma_eff^2 = Var_w*(mu) + sigma^2 + bw^2 (law of total variance).
print()
print("Population p variance check: Var_p = Var_w*(mu) + sigma^2 + bw^2")
sigma=25.0; bw=12.0
mus=np.array([200.,206.28,212.57,218.85]); w=np.array([0.229,0.187,0.312,0.271]); w=w/w.sum()
sigeff=np.sqrt(sigma**2+bw**2)
grid=np.linspace(0,500,600000)
dens=sum(wi*np.exp(-0.5*((grid-mi)/sigeff)**2)/(sigeff*np.sqrt(2*np.pi)) for wi,mi in zip(w,mus))
dt=grid[1]-grid[0]; dens/= dens.sum()*dt
m1=(grid*dens).sum()*dt; varp=(grid**2*dens).sum()*dt - m1**2
mubar=(w*mus).sum(); varmu=(w*(mus-mubar)**2).sum()
print(f"  Var_p={varp:.5f}  predicted={varmu+sigma**2+bw**2:.5f}  diff={varp-(varmu+sigma**2+bw**2):.3e}")

# Max achievable variance of the RAW-sigma model q_w on support {mu_1..mu_K}:
# q_w = sum w_k N(mu_k, sigma^2). Var_q = Var_w(mu) + sigma^2 <= max_w Var_w(mu) + sigma^2.
# max over simplex of Var_w(mu) is achieved by 2-point mass at extremes {mu_1, mu_K}: = (a)(1-a)(mu_K-mu_1)^2,
# maximized at a=1/2 -> (mu_K-mu_1)^2/4. So max Var_q = (mu_K-mu_1)^2/4 + sigma^2.
# The data p needs Var_p = Var_w*(mu)+sigma^2+bw^2. The DEFICIT condition (cannot be matched by ANY q_w
# that also matches the mean? no -- the right statement is the moment/pinning criterion). We just confirm
# the variance the model can produce while matching the mean mubar* is bounded, forcing 2-vertex.
print()
print("Max raw-sigma model variance (2-pt at extremes, a=1/2):",
      (mus[-1]-mus[0])**2/4 + sigma**2)
print("Data variance to be matched:", varp)
print("So if data var > any single-interior-config var while mean-matching, mass pushed to extremes.")
