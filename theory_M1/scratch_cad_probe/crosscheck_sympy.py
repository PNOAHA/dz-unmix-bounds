# Independent sympy cross-check of the Mathematica CAD probe outputs (two-route rule).
# Checks: (1) Probe B dominance certificates k=3,5,7 (exact factored polynomials);
#         (2) Probe C exact gain formula I13/I1 - 1;
#         (3) small-rho constant 1/sqrt(2*pi) and I_mean leading 2*pi*rho^2.
import sympy as sp

d, a, rho = sp.symbols('d a rho', positive=True)


def Mk(k, m):
    # k-th moment of N(m,1)
    t = sp.symbols('t')
    return sp.expand(sp.integrate(t**k * sp.exp(-(t - m)**2 / 2) / sp.sqrt(2 * sp.pi),
                                  (t, -sp.oo, sp.oo)))


mu = {k: sp.expand((1 - a) / 2 * Mk(k, -d) + (1 + a) / 2 * Mk(k, d)) for k in range(1, 15)}
var1 = sp.simplify(mu[2] - mu[1]**2)
print("Var1 =", var1)

print("\n(1) Probe B certificates:")
expected = {
    3: -2 * d**2 * (3 + 15 * d**2 + 4 * d**4),
    5: -8 * d**2 * (90 + 525 * d**2 + 340 * d**4 + 60 * d**6 + 3 * d**8),
    7: -6 * d**2 * (20685 + 152145 * d**2 + 151410 * d**4 + 49210 * d**6
                    + 6629 * d**8 + 385 * d**10 + 8 * d**12),
}
for k in (3, 5, 7):
    diff = sp.expand(Mk(k, d)**2 * var1 - d**2 * (mu[2 * k] - mu[k]**2))
    match = sp.expand(diff - expected[k]) == 0
    a_free = a not in diff.free_symbols
    print(f"  k={k}: matches Mathematica factorization: {match}; a-independent: {a_free}")

print("\n(2) Probe C gain formula:")
J = sp.Matrix([-2 * Mk(1, d), -2 * Mk(3, d)])
Sig = sp.Matrix([[mu[2] - mu[1]**2, mu[4] - mu[1] * mu[3]],
                 [mu[4] - mu[1] * mu[3], mu[6] - mu[3]**2]])
I13 = sp.simplify((J.T * Sig.inv() * J)[0, 0])
I1 = 4 * d**2 / var1
gain = sp.simplify(I13 / I1 - 1)
gain_mma = -2 * d**4 / (-3 + 3 * (-4 + a**2) * d**2 + 9 * (-1 + a**2) * d**4
                        + 2 * (-1 + a**2) * d**6)
print("  gain == Mathematica expression:", sp.simplify(gain - gain_mma) == 0)
print("  gain series (a=0):", sp.series(gain.subs(a, 0), d, 0, 9))
print("  gain(d=1,a=0) =", sp.nsimplify(gain.subs({d: 1, a: 0})),
      "= ", float(gain.subs({d: 1, a: 0})))

print("\n(3) small-rho constants:")
dr = sp.sqrt(2) * sp.erfinv(rho)
taustar_over_sqrts = sp.sqrt(1 + dr**2) / (2 * dr)   # a=0
c = sp.series(rho * taustar_over_sqrts, rho, 0, 3)
print("  rho*tau*/sqrt(s) =", c, "   [expect 1/sqrt(2*pi) + sqrt(pi/2)/6 rho^2]")
Imean = sp.series((4 * dr**2 / (1 + dr**2)), rho, 0, 5)
print("  I_mean(rho) =", Imean, "   [expect 2*pi*rho^2 - 2*pi^2/3 rho^4]")
