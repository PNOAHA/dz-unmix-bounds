"""Section 5 quoted numbers from applicability_{srcK,noise,sep}.csv, archived (results/) vs
certified (compstat_fix/results_certified/). Same aggregation as make_paper2_figures.py."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_paper2_figures as M

def numbers(resdir):
    M.RESDIR = resdir
    dK, dN, dS = (M.load_axis(a) for a in ("srcK", "noise", "sep"))
    o = {}
    o['K_x'] = dK['x'].tolist()
    for k in ('sup_top1_mean', 'sup_top1_dom_mean', 'uns_top1_mean', 'uns_top1_dom_mean'):
        o['K_' + k] = dK[k].tolist()
    o['K_adv_all'] = (dK['sup_top1_mean'] - dK['uns_top1_mean']).tolist()
    o['K_all_pass'] = bool(np.all(dK['sup_top1_dom_mean'] >= M.THRESH))
    o['noise_sup_dom_0_020'] = [dN['sup_top1_dom_mean'][0], dN['sup_top1_dom_mean'][-1]]
    o['noise_sup_all_0_020'] = [dN['sup_top1_mean'][0], dN['sup_top1_mean'][-1]]
    o['noise_uns_all_0_020'] = [dN['uns_top1_mean'][0], dN['uns_top1_mean'][-1]]
    o['noise_x_last'] = dN['x'][-1]
    o['noise_adv_all_020'] = dN['sup_top1_mean'][-1] - dN['uns_top1_mean'][-1]
    o['noise_adv_dom_020'] = dN['sup_top1_dom_mean'][-1] - dN['uns_top1_dom_mean'][-1]
    x = dS['x']; sd = dS['sup_top1_dom_mean']
    o['sep_x0'] = x[0]; o['sep_sup_dom_x0'] = sd[0]; o['sep_uns_all_x0'] = dS['uns_top1_mean'][0]
    o['sep_plateau_mean_ge3'] = float(np.mean(sd[x >= 3])); o['sep_plateau_min_ge3'] = float(np.min(sd[x >= 3]))
    o['sep_crossing'] = M.crossing(x, sd, M.THRESH)
    o['sep_first_ge085'] = float(x[np.argmax(sd >= M.THRESH)])
    o['sep_last_lt085'] = float(x[np.where(sd < M.THRESH)[0].max()]) if np.any(sd < M.THRESH) else None
    o['sep_all_above_after_crossing'] = bool(np.all(sd[x >= o['sep_first_ge085']] >= M.THRESH))
    adv = dS['sup_top1_mean'] - dS['uns_top1_mean']; advd = dS['sup_top1_dom_mean'] - dS['uns_top1_dom_mean']
    i = int(np.argmax(adv)); o['sep_adv_peak'] = [float(x[i]), float(adv[i])]
    o['sep_neg_adv_all'] = [[float(x[j]), float(adv[j]), float(advd[j])] for j in np.where(adv < 0)[0]]
    o['sep_neg_adv_dom'] = [[float(x[j]), float(adv[j]), float(advd[j])] for j in np.where(advd < 0)[0]]
    o['uns_sep_min'] = float(np.min(dS['uns_top1_mean']))
    return o

if __name__ == '__main__':
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {'archived': numbers(os.path.join(base, 'results')),
           'certified': numbers(os.path.join(base, 'compstat_fix', 'results_certified'))}
    json.dump(out, open(os.path.join(base, 'compstat_fix', 'section5_numbers.json'), 'w'), indent=1, default=float)
    for k in out['archived']:
        print(k, '\n  old', out['archived'][k], '\n  new', out['certified'][k])
