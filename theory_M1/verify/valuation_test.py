"""Adversarial test: can sympy do the exact valuation / Newton polygon
computation the candidate claims only Mathematica can do?

Matrix R[K,e]_ij = (1-e)^{(i-j)^2}  (exact polynomial entries over ZZ[e]).
Compute charpoly exactly, read e-valuation of each lambda-coefficient,
derive Newton polygon slopes = exact e-orders of all K eigenvalues.
"""
import time
import sympy as sp

e, lam = sp.symbols('e lam')

def valuations(K):
    M = sp.Matrix(K, K, lambda i, j: (1 - e)**((i - j)**2))
    t0 = time.time()
    p = M.charpoly(lam)          # exact, over ZZ[e] internally
    coeffs = sp.Poly(p.as_expr(), lam).all_coeffs()  # deg K ... 0
    vals = []
    for c in coeffs:
        cp = sp.Poly(sp.expand(c), e)
        if cp.is_zero:
            vals.append(None)
        else:
            # lowest nonzero power of e
            mono = min(m[0] for m in cp.monoms() if True)
            # monoms are sorted desc; find min degree with nonzero coeff
            degs = [m[0] for m in cp.monoms()]
            vals.append(min(degs))
    dt = time.time() - t0
    return vals, dt

print(f"sympy {sp.__version__}")
total0 = time.time()
for K in range(2, 9):
    vals, dt = valuations(K)
    # vals[i] is valuation of coeff of lam^(K-i); Newton polygon slopes:
    # expected valuation of coeff of lam^(K-j) is j(j-1)/2
    expected = [j*(j-1)//2 for j in range(0, K+1)]
    ok = vals == expected
    # slopes of lower hull between consecutive vertices
    slopes = [vals[j+1] - vals[j] for j in range(K)]
    print(f"K={K}: valuations={vals} expected={expected} "
          f"match={ok} slopes={slopes} time={dt:.2f}s")
print(f"TOTAL {time.time()-total0:.2f}s")
