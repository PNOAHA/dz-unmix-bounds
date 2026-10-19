"""Audit small saved outputs against SI claims, without treating snapshots as fresh MC."""
from pathlib import Path
import json, math, numpy as np
p=Path(__file__).resolve().parent/'reference_results'
j=json.loads((p/'n2_population_kkt.json').read_text());cs=[c for v in j['grid'].values() for c in v]
face=[c for c in cs if c['survivors']==[0,c['K']-1]]
any_pinned=[c for c in cs if c['pinned_set']]
partial=[c for c in any_pinned if c['survivors']!=[0,c['K']-1]]
kkt=dict(saved_grid_cells=len(cs),any_pinned_cells=len(any_pinned),endpoint_face_cells=len(face),partial_face_cells=len(partial),min_slack_any_pinned=min(c['eta'][i] for c in any_pinned for i in c['pinned_set']),min_slack_endpoint_face=min(c['eta'][i] for c in face for i in range(1,c['K']-1)),SI_claim_cells=82,SI_claim_min_slack=1.11e-6,status='82-cell count/minimum recovered for any-pinned selection; endpoint-only claim mismatches 17 cells',partial_face_examples=[{k:c[k] for k in ['K','bw','kind','rho','survivors','pinned_set','min_eta_pinned']} for c in partial])
t=json.loads((p/'top1_confirm.json').read_text());errs=[abs(c['top1']-c['top1_pred']) for c in t]
w=max(t,key=lambda c:abs(c['top1']-c['top1_pred']))
top=dict(cells=len(t),within_absolute_002=sum(e<=.02 for e in errs),within_absolute_003=sum(e<=.03 for e in errs),max_error=max(errs),worst=w,status='36 cells recovered; inspect tolerance rather than asserting all 35 are within 2 percent')
j=json.loads((p/'c1_prefactor_vs_K.json').read_text());fits=j['fits'];const={}
for K,f in fits.items():
    cell=next(c for c in j['cells'] if str(c['K'])==K and c['rho']==0.06)
    adj=f['cK_from_Nscan_rho006']
    const[K]=dict(C_Nscan=adj*cell['rho_1K']/0.06,adjacent_C_times_Km1=adj*(int(K)-1),N_scan_slope=f['N_scan_slope'])
n3=json.loads((p/'n3_k4_modes.json').read_text());mode={k:dict(ratios_to_t=v['ratios_to_t'],interior=v['interior']) for k,v in n3['per_rho'].items()}
print(json.dumps(dict(KKT=kkt,top1=top,K3_to_6_fits=const,K4_modes=mode),indent=2))
