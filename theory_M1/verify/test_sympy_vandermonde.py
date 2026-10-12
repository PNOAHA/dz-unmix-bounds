# Adversarial test: can sympy do the candidate's "Mathematica-only" computation?
# Matrix R_ij = q^{(i-j)^2}, K=2..7:
#   - symbolic det (candidate claims sympy "hangs" for K>=6)
#   - e_k = sum of k x k principal minors, exact polynomial in q
#   - substitute q = 1-u, extract lowest-order term -> exponent + exact rational c_k
#   - ladder lambda_k ~ (e_k/e_{k-1}) leading coefficient
# Compare C_K (smallest eigenvalue constant) with draft values 1 / 1.33 / 2.4 / 5.49
import time, itertools
import sympy as sp

q, u = sp.symbols('q u')

def R(K):
    return sp.Matrix(K, K, lambda i, j: q**((i - j)**2))

def lowest_term(poly_in_q):
    # substitute q = 1-u, expand, return (min exponent, exact coeff)
    p = sp.Poly(sp.expand(poly_in_q.subs(q, 1 - u)), u)
    terms = p.terms()  # list of ((exp,), coeff) sorted desc
    exp, coeff = min(((e[0], c) for e, c in terms), key=lambda t: t[0])
    return exp, coeff

t0 = time.time()
report = []
for K in range(2, 8):
    tK = time.time()
    M = R(K)
    detM = M.det(method='berkowitz')
    t_det = time.time() - tK
    # e_k for all k
    ek = {0: sp.Integer(1)}
    for k in range(1, K + 1):
        s = sp.Integer(0)
        for S in itertools.combinations(range(K), k):
            s += M[list(S), list(S)].det(method='berkowitz')
        ek[k] = sp.expand(s)
    assert sp.expand(ek[K] - detM) == 0
    # ladder: lambda_k ~ e_k/e_{k-1} leading order at q=1-u
    ladder = []
    for k in range(1, K + 1):
        e_exp, e_c = lowest_term(ek[k])
        em_exp, em_c = lowest_term(ek[k - 1]) if k > 1 else (0, sp.Integer(1))
        lam_exp = e_exp - em_exp
        lam_c = sp.nsimplify(e_c / em_c)
        ladder.append((k, lam_exp, sp.Rational(lam_c) if lam_c.is_rational else lam_c))
    t_all = time.time() - tK
    Cmin = ladder[-1]
    report.append((K, t_det, t_all, Cmin, ladder))
    print(f"K={K}: det time={t_det:.3f}s, det+all-e_k time={t_all:.3f}s")
    print(f"   lambda_min: exponent (1-q)^{Cmin[1]}  (claim: K-1={K-1}),  C_K = {Cmin[2]} = {float(Cmin[2]):.4f}")
    lam2 = ladder[-2] if K >= 2 else None
    print(f"   second-smallest: exponent (1-q)^{lam2[1]} (claim: K-2={K-2}),  c = {lam2[2]} = {float(lam2[2]):.4f}")

print(f"\nTOTAL time K=2..7 (det + all principal minors + series): {time.time()-t0:.2f}s")

# closed-form guess check: C_K = 2^(K-1) * ((K-1)!)^3 / (2K-2)!
print("\nclosed-form check  C_K = 2^(K-1)((K-1)!)^3/(2K-2)! :")
for K, _, _, Cmin, _ in report:
    pred = sp.Rational(2**(K-1) * sp.factorial(K-1)**3, sp.factorial(2*K-2))
    print(f"  K={K}: computed {Cmin[2]}  vs predicted {pred}  match={sp.simplify(Cmin[2]-pred)==0}")
