"""Bias / RMSE decomposition of the Fig. 3 sigma_err harness (v1.2.0).

Reads out_phase/sigma_err_slope_numpy.json and writes analysis/bias_rmse_allcells.csv:
per cell, sampling spread sigma_err=sqrt(tr Cov), L2 bias ||mean(w_hat)-w*||, RMSE=sqrt(tr Cov+bias^2).
Usage: python analysis/bias_rmse.py [input.json [output.csv]]
compstat-fix: optional arguments select the certified-solver input
(out_phase/sigma_err_slope_numpy_certified.json -> analysis/bias_rmse_allcells_certified.csv).
"""
import json, math, numpy as np, csv, os, sys
_HERE = os.path.dirname(os.path.abspath(__file__))
_IN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_HERE, '..', 'out_phase', 'sigma_err_slope_numpy.json')
_OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_HERE, 'bias_rmse_allcells.csv')
d=json.load(open(_IN))
R=d['meta']['R']
def wstar(K):
    w=np.linspace(1.0,0.6,K); return w/w.sum()   # endpoint kind (meta w_kind=endpoint)
rows=[]
for c in d['cells']:
    K=c['K']; w=wstar(K); m=np.array(c['what_mean'])
    assert abs(c['wstar_true']-w.max())<1e-12
    eig=np.array(c['sigma_err_tangent_eigs']); tr=float(max(eig.sum(),0))
    b=m-w; bn=float(np.linalg.norm(b))
    se_b=math.sqrt(tr/R)            # MC SE scale of ||mean - E w||
    rmse=math.sqrt(tr+bn**2)
    rows.append(dict(K=K,rho=c['rho'],N=c['N'],regime=c['regime'],sigma_err=c['sigma_err'],
        trcov=tr,sqrt_trcov=math.sqrt(tr),bias_l2=bn,bias_mc_se=se_b,rmse=rmse,
        bias_frac_mse=bn**2/(tr+bn**2), what_mean=';'.join(f'{x:.4f}' for x in m),
        wstar=';'.join(f'{x:.4f}' for x in w), C_sd=c['sigma_err']*math.sqrt(c['N'])*c['rho'],
        C_rmse=rmse*math.sqrt(c['N'])*c['rho'], top1=c['top1']))
with open(_OUT,'w',newline='') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
cal=[r for r in rows if r['K']==2 and r['regime']=='LAN']
print('ncal',len(cal))
a=lambda k:np.array([r[k] for r in cal])
print('C_sd mean %.4f sd %.4f'%(a('C_sd').mean(),a('C_sd').std()))
print('C_rmse mean %.4f sd %.4f'%(a('C_rmse').mean(),a('C_rmse').std()))
print('bias range %.4f-%.4f median %.4f'%(a('bias_l2').min(),a('bias_l2').max(),np.median(a('bias_l2'))))
print('bias/sigma_err range %.3f-%.3f median %.3f'%((a('bias_l2')/a('sigma_err')).min(),(a('bias_l2')/a('sigma_err')).max(),np.median(a('bias_l2')/a('sigma_err'))))
print('rmse/sigma_err range %.3f-%.3f median %.3f'%((a('rmse')/a('sigma_err')).min(),(a('rmse')/a('sigma_err')).max(),np.median(a('rmse')/a('sigma_err'))))
print('bias_frac_mse range %.3f-%.3f'%(a('bias_frac_mse').min(),a('bias_frac_mse').max()))
print('bias/MCSE range %.2f-%.2f'%((a('bias_l2')/a('bias_mc_se')).min(),(a('bias_l2')/a('bias_mc_se')).max()))
# slope of rmse vs 1/(sqrt N rho)
x=np.log(1/(np.sqrt(a('N'))*a('rho')))
print('slope sd %.3f rmse %.3f'%(np.polyfit(x,np.log(a('sigma_err')),1)[0],np.polyfit(x,np.log(a('rmse')),1)[0]))
for r in sorted(cal,key=lambda r:(r['rho'],r['N'])):
    print('%.2f %4d sd=%.4f tr^.5=%.4f bias=%.4f(se %.4f) rmse=%.4f mean=%s'%(r['rho'],r['N'],r['sigma_err'],r['sqrt_trcov'],r['bias_l2'],r['bias_mc_se'],r['rmse'],r['what_mean']))
print('--- K=3 cells summary')
for reg in ('LAN','ill'):
    k3=[r for r in rows if r['K']==3 and r['regime']==reg]
    b=np.array([r['bias_l2'] for r in k3]); s=np.array([r['sigma_err'] for r in k3])
    print(reg,len(k3),'bias %.3f-%.3f'%(b.min(),b.max()),'bias/sd med %.2f'%np.median(b/s))
for r in [r for r in rows if r['K']==3 and r['regime']=='LAN']:
    print('%.2f %4d sd=%.4f bias=%.4f rmse=%.4f mean=%s w*=%s'%(r['rho'],r['N'],r['sigma_err'],r['bias_l2'],r['rmse'],r['what_mean'],r['wstar']))
k2ill=[r for r in rows if r['K']==2 and r['regime']=='ill']
b=np.array([r['bias_l2'] for r in k2ill]); print('K2 ill bias %.3f-%.3f'%(b.min(),b.max()))
