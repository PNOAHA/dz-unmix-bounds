# Independent cross-check (second verification path, per user's two-route rule):
# 1) mpmath high-precision NUMERIC eigenvalues of R(K, q=1-u) at small u,
#    compared against the SYMBOLIC ladder predictions lambda_(k) ~ c_k * u^(k-1)
#    (k-th largest). Different arithmetic + different algorithm than the
#    sympy principal-minor route in test_sympy_vandermonde.py.
#    v2 (2026-07-26, after 3-lens review): dps is now ADAPTIVE in (K, u).
#    The original flat dps=50 hit the precision floor at K=7, u=1e-8
#    (lambda_min ~ 5e-47 vs ~1e-50 absolute noise => spurious 1.16e-5 relative
#    deviation that did NOT shrink with u). With adaptive dps the K=7
#    worst deviations are a clean 1.35e-5 -> 1.35e-7 (~13.5u), strictly O(u).
# 2) symbolic verification of the corrected generalized-Vandermonde det formula
#    det R = q^(2*sum i^2) * t^(K(K+1)/2) * prod_{i<j}(t^j - t^i),  t = q^-2
#    (1-based indices), for K = 3, 4, 5.
import itertools
import sympy as sp
from mpmath import mp, mpf, matrix, eig

q, u = sp.symbols('q u')


def R_sym(K):
    return sp.Matrix(K, K, lambda i, j: q ** ((i - j) ** 2))


def ladder_coeffs(K):
    """Full ladder via e_k/e_{k-1} lowest-order terms: returns list of
    (k, exponent, exact coeff) for k-th LARGEST eigenvalue."""
    M = R_sym(K)
    ek = {0: sp.Integer(1)}
    for k in range(1, K + 1):
        s = sp.Integer(0)
        for S in itertools.combinations(range(K), k):
            s += M[list(S), list(S)].det(method='berkowitz')
        ek[k] = sp.expand(s)
    out = []
    prev_exp, prev_c = 0, sp.Integer(1)
    for k in range(1, K + 1):
        p = sp.Poly(sp.expand(ek[k].subs(q, 1 - u)), u)
        exp_k, c_k = min(((m[0], c) for m, c in p.terms()), key=lambda t: t[0])
        out.append((k, exp_k - prev_exp, sp.nsimplify(c_k / prev_c)))
        prev_exp, prev_c = exp_k, c_k
    return out


print("=== path 2: mpmath adaptive-precision numeric eigenvalues vs symbolic ladder ===")
for K in (5, 7):
    lad = ladder_coeffs(K)
    for uval_str in ("1e-6", "1e-8"):
        expo = int(uval_str.split("e-")[1])
        mp.dps = 40 + (K - 1) * expo + 20   # keep lambda_min ~ C_K*u^(K-1) far above noise
        uval = mpf(uval_str)
        qval = 1 - uval
        A = matrix(K, K)
        for i in range(K):
            for j in range(K):
                A[i, j] = qval ** ((i - j) ** 2)
        ev, _ = eig(A)
        ev = sorted([e.real for e in ev], reverse=True)  # descending
        worst = mpf(0)
        for (k, ex, c) in lad:
            pred = mpf(str(sp.Float(c, 40))) * uval ** ex
            rel = abs(ev[k - 1] - pred) / pred
            worst = max(worst, rel)
        print(f"K={K} u={uval_str} (dps={mp.dps}): worst relative deviation over all {K} rungs = {mp.nstr(worst, 3)}")
        # measured (v2): K=5: 6.3e-6 -> 6.3e-8; K=7: 1.35e-5 -> 1.35e-7 (worst ~13.5u, strictly O(u))
        assert worst < 100 * uval, f"deviation not O(u): K={K}, u={uval_str}, worst={worst}"

print()
print("=== det formula check: R = D W D, generalized Vandermonde (1-based) ===")
t = sp.symbols('t')
for K in (3, 4, 5):
    detR = sp.factor(R_sym(K).det(method='berkowitz'))
    formula = (q ** (2 * sum(i * i for i in range(1, K + 1)))
               * t ** sp.Rational(K * (K + 1), 2)
               * sp.prod(t ** jj - t ** ii
                         for ii in range(1, K + 1) for jj in range(ii + 1, K + 1)))
    diff = sp.simplify(detR - formula.subs(t, q ** -2))
    print(f"K={K}: det R == q^(2*Sum i^2) * t^(K(K+1)/2) * prod(t^j - t^i) [t=q^-2]: {diff == 0}")

print()
print("=== ladder tables (k-th largest, exponent, exact coeff) ===")
for K in (5, 7):
    print(f"K={K}: {[(k, e, str(c)) for k, e, c in ladder_coeffs(K)]}")
