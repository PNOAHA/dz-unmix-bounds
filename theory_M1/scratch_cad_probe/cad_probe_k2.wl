(* cad_probe_k2.wl -- M1 ideal-lower-bound CAD probe (2026-07-26, approved half-day experiment)
   Scope (per adversarial-review downgrade): IDEAL lower-bound layer only, K=2 equal-sigma
   Gaussian two-source, supervised weight. NOT the real-method K x rho phase diagram
   (GPU 2026-06-12 already showed ideal theory does not describe the real method).
   Kill gates: (i) CAD returns readable region within 1800 s per call;
   (ii) qualitative consistency with ideal K=2 law: boundary monotone in separation,
   small-rho constant consistent with C_naive = 1/Sqrt[Pi] (sigma_err convention = Sqrt[2] x std).
   Runs single-kernel on D107 .200; wall time should be minutes. *)

(* Sources at -d and +d (both sigma=1), weight w on the -d source; a = 1-2w in (-1,1). *)
Mk[k_, m_] := Moment[NormalDistribution[m, 1], k];
mu[k_] := Expand[(1 - a)/2 Mk[k, -d] + (1 + a)/2 Mk[k, d]];

var1 = Simplify[mu[2] - mu[1]^2];
Print["mu1=", mu[1], "   mu2=", mu[2], "   mu3=", mu[3]];
Print["Var1=", var1];

(* ---------- Probe A: mean-statistic failure region, exact boundary ---------- *)
(* failure: signal (dmu1/dw * tau)^2 = 4 d^2 tau^2  <  noise floor s * Var1, s = z^2/N *)
tA = AbsoluteTiming[
  cadA = TimeConstrained[
    CylindricalDecomposition[
      {4 tau^2 d^2 < s (1 + d^2 - a^2 d^2), d > 0, s > 0, tau > 0, -1 < a < 1},
      {a, s, d, tau}], 1800, $TimedOut]];
Print["ProbeA CAD time = ", tA[[1]], " s"];
Print["ProbeA failure region (CAD): ", cadA];

taustar = Sqrt[s (1 + (1 - a^2) d^2)]/(2 d);
Print["exact boundary tau* = ", taustar];
Print["monotone-in-d certificate (should be True): ",
  Simplify[D[(1 + (1 - a^2) d^2)/(4 d^2), d] < 0, Assumptions -> d > 0 && -1 < a < 1]];

(* to rho coordinates: rho = Erf[delta/(2 Sqrt[2])], delta = 2d  =>  d = Sqrt[2] InverseErf[rho] *)
taurho = taustar /. d -> Sqrt[2] InverseErf[rho];
Print["tau*(rho) = ", Simplify[taurho, Assumptions -> 0 < rho < 1 && s > 0 && -1 < a < 1]];
Print["small-rho: rho * tau* * Sqrt[N]/z  (a=0) = ",
  Series[rho taurho/Sqrt[s] /. a -> 0, {rho, 0, 2}]];
Print["reference constants: 1/Sqrt[2 Pi] = ", N[1/Sqrt[2 Pi]],
  " ; sigma_err convention x Sqrt[2] -> 1/Sqrt[Pi] = ", N[1/Sqrt[Pi]], " (= C_naive)"];

(* Fisher consistency: mean-statistic information vs reduced ideal I_red = 2 Pi rho^2 (K=2) *)
I1 = 4 d^2/var1;
Print["I_mean(rho) series (a=0): ",
  Series[I1 /. a -> 0 /. d -> Sqrt[2] InverseErf[rho], {rho, 0, 4}], "   [I_red = 2 Pi rho^2]"];

(* numeric spot for gate (ii): boundary value at rho=0.1, a=0, z=1 *)
Print["gate spot: rho=0.1, a=0: rho*tau**Sqrt[N]/z = ",
  N[rho taurho/Sqrt[s] /. {a -> 0, rho -> 1/10}]];

(* ---------- Probe B: can a single higher odd moment ever beat the mean? ---------- *)
(* comparison M_k^2/Var_k > M_1^2/Var_1 ; a^2 terms cancel identically (checked below) *)
Do[
 diff = Simplify[Mk[k, d]^2 var1 - d^2 (mu[2 k] - mu[k]^2)];
 Print["k=", k, ": Mk^2 Var1 - M1^2 Vark = ", Factor[diff]];
 tB = AbsoluteTiming[
   cadB = TimeConstrained[
     CylindricalDecomposition[
       {Mk[k, d]^2 var1 > d^2 (mu[2 k] - mu[k]^2), d > 0, -1 < a < 1}, {a, d}],
     1800, $TimedOut]];
 Print["k=", k, ": CAD time = ", tB[[1]],
   " s ; region where moment-", k, " beats mean: ", cadB],
 {k, {3, 5, 7}}];

(* ---------- Probe C: exact Fisher gain of (m1,m3) jointly over m1 alone ---------- *)
J = {-2 Mk[1, d], -2 Mk[3, d]};
Sig = {{mu[2] - mu[1]^2, mu[4] - mu[1] mu[3]}, {mu[4] - mu[1] mu[3], mu[6] - mu[3]^2}};
I13 = Simplify[J . Inverse[Sig] . J, Assumptions -> d > 0 && -1 < a < 1];
gain = Simplify[I13/I1 - 1, Assumptions -> d > 0 && -1 < a < 1];
Print["exact gain (I13/I1 - 1) = ", gain];
Print["gain>0 region (Reduce): ",
  TimeConstrained[Reduce[gain > 0 && d > 0 && -1 < a < 1, {d, a}, Reals], 1800, $TimedOut]];
Print["gain series (a=0, small d): ", Series[gain /. a -> 0, {d, 0, 8}]];
Print["gain numeric (d=1, a=0): ", N[gain /. {d -> 1, a -> 0}]];
Print["gain numeric (d=2, a=1/2): ", N[gain /. {d -> 2, a -> 1/2}]];
Print["gain numeric (d=1/4, a=0): ", N[gain /. {d -> 1/4, a -> 0}]];
Print["ALL DONE"];
