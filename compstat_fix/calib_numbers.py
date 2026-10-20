"""All calibration numbers quoted in the manuscript / Online Resource 1, from the archived
(2000-iteration MU) and the certified-solver calibration JSON (identical KDE rows)."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import sigma_err_scan as S
def wstar(K):
    w = np.linspace(1, 0.6, K); return w / w.sum()
def summary(path):
    d = json.load(open(path)); R = d['meta']['R']; rows = []
    for c in d['cells']:
        K = c['K']; w = wstar(K); m = np.array(c['what_mean']); eig = np.array(c['sigma_err_tangent_eigs'])
        tr = float(max(eig.sum(), 0)); bn = float(np.linalg.norm(m - w)); rmse = math.sqrt(tr + bn ** 2)
        rows.append(dict(K=K, rho=c['rho'], N=c['N'], reg=c['regime'], s=c['sigma_err'], eig=list(eig), m=list(m), bias=bn,
                         se=math.sqrt(tr / R), rmse=rmse, frac=bn ** 2 / (tr + bn ** 2), top1=c['top1']))
    cal = [r for r in rows if r['K'] == 2 and r['reg'] == 'LAN']
    A = lambda k: np.array([r[k] for r in cal])
    x = np.log(1 / (np.sqrt(A('N')) * A('rho')))
    Csd = A('s') * np.sqrt(A('N')) * A('rho'); Crm = A('rmse') * np.sqrt(A('N')) * A('rho')
    out = dict(n_cal=len(cal), C=[Csd.mean(), Csd.std(), Csd.std() / Csd.mean()], C_rmse=[Crm.mean(), Crm.std()],
               slope_sd=np.polyfit(x, np.log(A('s')), 1)[0], slope_rmse=np.polyfit(x, np.log(A('rmse')), 1)[0],
               bias=[A('bias').min(), A('bias').max(), np.median(A('bias'))],
               bias_over_se=[(A('bias') / A('se')).min(), (A('bias') / A('se')).max()],
               bias_frac_mse=[A('frac').min(), A('frac').max()],
               rmse_over_s=[(A('rmse') / A('s')).min(), (A('rmse') / A('s')).max(), np.median(A('rmse') / A('s'))],
               w1=[min(r['m'][0] for r in cal), max(r['m'][0] for r in cal)])
    # fit_lan_slope as in the harness for K=2 fits, plus the figure's panel-B slope
    out['table_K2'] = [(r['rho'], r['N'], r['s'], r['bias'], r['se'], r['rmse'], r['rmse'] / r['s'], r['m'][0]) for r in sorted(cal, key=lambda r: (r['rho'], r['N']))]
    k3 = [r for r in rows if r['K'] == 3]
    b3 = np.array([r['bias'] for r in k3]); s3 = np.array([r['s'] for r in k3])
    out['K3_all'] = dict(bias=[b3.min(), b3.max()], med_bias_over_s=float(np.median(b3 / s3)))
    pin = [r for r in k3 if r['rho'] <= 0.19 + 1e-9 and r['N'] in (160, 350)]
    out['K3_pinned_bias_160_350'] = [min(r['bias'] for r in pin), max(r['bias'] for r in pin)]
    out['K3_pinned_w2'] = [min(r['m'][1] for r in pin), max(r['m'][1] for r in pin)]
    g = {(r['rho'], r['N']): r for r in k3}
    out['K3_sigma'] = {f"{rho}_{N}": g[(rho, N)]['s'] for rho in (0.19, 0.35, 0.47) for N in (160, 240, 350)}
    out['K3_rmse'] = {f"{rho}_{N}": g[(rho, N)]['rmse'] for rho in (0.19, 0.47) for N in (160, 240, 350)}
    out['K3_table'] = [(r['N'], r['rho'], r['s'], r['bias'], r['rmse'], r['m']) for r in sorted(k3, key=lambda r: (r['N'], r['rho'])) if r['N'] in (160, 350)]
    out['D3_eigs'] = dict(a=g[(0.19, 160)]['eig'][:2], b=g[(0.26, 350)]['eig'][:2])
    # rmse monotone in rho for K=3?  sigma local max?
    out['K3_rmse_monotone'] = {N: bool(np.all(np.diff([g[(rho, N)]['rmse'] for rho in sorted({r['rho'] for r in k3})]) < 0)) for N in (160, 240, 350)}
    out['K3_sigma_argmax_rho_ge_0.19'] = {N: max([rho for rho in sorted({r['rho'] for r in k3}) if rho >= 0.19], key=lambda rr: g[(rr, N)]['s']) for N in (160, 240, 350)}
    return out
res = {t: summary(os.path.join(ROOT, 'out_phase', f)) for t, f in (('archived_mu2000', 'sigma_err_slope_numpy.json'), ('certified', 'sigma_err_slope_numpy_certified.json'))}
json.dump(res, open(os.path.join(HERE, 'calib_numbers.json'), 'w'), indent=1, default=float)
for k in res['certified']:
    if k in ('table_K2', 'K3_table'): continue
    print(k, '\n   old', res['archived_mu2000'][k], '\n   new', res['certified'][k])
