# v0.3 review follow-up (2026-07-26), two jobs:
# (A) Extend the exact eigenvalue-ladder constants to K=8,9 via the charpoly
#     route (charpoly coeff of lam^(K-k) = (-1)^k e_k), aligning the constants
#     range with the valuation range K=2..8. Verify closed form
#     C_K = 2^(K-1)((K-1)!)^3/(2K-2)!.
# (B) Check the tangent-space (sum-to-one, Helmert Z) minimum eigenvalue against
#     the review finding: lambda_min(Z^T R Z)/lambda_min(R) -> 1 (even K, exact
#     by centrosymmetry parity) but -> K/(K-1) (odd K). Also check
#     lambda_max(Z^T R Z) == second-largest of R.
import time
import sympy as sp
from mpmath import mp, mpf, matrix, eigsy, sqrt

q, u, lam = sp.symbols('q u lam')

print("=== (A) exact ladder constants via charpoly, K=8,9 ===")
for K in (8, 9):
    t0 = time.time()
    M = sp.Matrix(K, K, lambda i, j: (1 - u) ** ((i - j) ** 2))
    p = sp.Poly(M.charpoly(lam).as_expr(), lam)
    coeffs = p.all_coeffs()  # lam^K ... lam^0 ; coeff of lam^(K-k) = (-1)^k e_k
    lows = []
    for k in range(0, K + 1):
        c = sp.expand((-1) ** k * coeffs[k])
        cp = sp.Poly(c, u)
        deg, coef = min(((m[0], cc) for m, cc in zip(cp.monoms(), cp.coeffs())),
                        key=lambda tt: tt[0])
        lows.append((deg, coef))
    ladder = []
    for k in range(1, K + 1):
        ladder.append((k, lows[k][0] - lows[k - 1][0],
                       sp.nsimplify(lows[k][1] / lows[k - 1][1])))
    CK = ladder[-1][2]
    pred = sp.Rational(2 ** (K - 1) * sp.factorial(K - 1) ** 3, sp.factorial(2 * K - 2))
    print(f"K={K} ({time.time()-t0:.1f}s): exponents={[e for _, e, _ in ladder]}")
    print(f"   ladder coeffs = {[str(c) for _, _, c in ladder]}")
    print(f"   C_K={CK}  closed-form {pred}  match={sp.simplify(CK-pred)==0}")
    print(f"   second-smallest coeff = {ladder[-2][2]}")

print()
print("=== (B) tangent-space gamma_K = lim lambda_min(Z^T R Z)/lambda_min(R) ===")
mp.dps = 150


def helmert(K):
    Z = matrix(K, K - 1)
    for j in range(1, K):
        nrm = sqrt(mpf(j) * (j + 1))
        for i in range(j):
            Z[i, j - 1] = 1 / nrm
        Z[j, j - 1] = -mpf(j) / nrm
    return Z


for K in range(2, 9):
    Z = helmert(K)
    for uval_str in ("1e-4", "1e-6"):
        uv = mpf(uval_str)
        qv = 1 - uv
        R = matrix(K, K)
        for i in range(K):
            for j in range(K):
                R[i, j] = qv ** ((i - j) ** 2)
        evR = sorted(eigsy(R, eigvals_only=True))
        RZ = (Z.T) * R * Z
        # symmetrize against tiny asymmetry from rounding
        S = (RZ + RZ.T) / 2
        evZ = sorted(eigsy(S, eigvals_only=True))
        ratio_min = evZ[0] / evR[0]
        ratio_top = evZ[-1] / evR[-2]   # tangent top vs second-largest of R
        gamma_pred = mpf(1) if K % 2 == 0 else mpf(K) / (K - 1)
        print(f"K={K} u={uval_str}: lam_min ratio={mp.nstr(ratio_min, 8)} "
              f"(pred gamma_K={mp.nstr(gamma_pred, 6)}), "
              f"tangent-top/second-largest={mp.nstr(ratio_top, 10)}")
