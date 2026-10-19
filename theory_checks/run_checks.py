"""Run historical SI probes in an isolated temporary copy.
--smoke reduces only expensive Monte-Carlo jobs; all deterministic probes run unchanged.
The JSON records every AST change. Reduced runs verify execution, not SI numbers.
"""
from pathlib import Path
import argparse, ast, concurrent.futures, json, os, shutil, subprocess, sys, tempfile, time

HEAVY = {'n1_k3_pinning.py','n2_population_kkt.py','n3_k4_modes.py','n4_stability_decay.py',
'c1_prefactor.py','c3_top1_split.py','top1_confirm.py','probe_turnover.py',
'probe2_clip_eigcross.py','probe2_clip_truncgauss.py','probe2_clip_model.py',
'probe2_clip_final.py','probe2_clip_mechanism.py','probe3_metric_projection.py',
'probe2_clip_effectiveK.py','probe2_clip_effK_reconcile.py'}
class Reduce(ast.NodeTransformer):
    def __init__(self): self.changes=[]
    def note(self,node,old,new):
        self.changes.append({'line':node.lineno,'before':old,'after':new})
    def visit_Assign(self,node):
        node=self.generic_visit(node)
        caps={'N':2000,'R':24,'N_ITER':4000,'NITER':12000,'n_iter':4000,'NSAMP':40}
        if len(node.targets)==1 and isinstance(node.targets[0],(ast.Tuple,ast.List)) and isinstance(node.value,(ast.Tuple,ast.List)):
            for target,value in zip(node.targets[0].elts,node.value.elts):
                if isinstance(target,ast.Name) and target.id in caps and isinstance(value,ast.Constant) and isinstance(value.value,(int,float)) and value.value>caps[target.id]:
                    self.note(node,str(value.value),str(caps[target.id]));value.value=caps[target.id]
        if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in caps and isinstance(node.value,ast.Constant) and isinstance(node.value.value,(int,float)) and node.value.value>caps[name]:
                self.note(node,ast.unparse(node.value),str(caps[name]));node.value=ast.Constant(caps[name])
            if name=='grid' and isinstance(node.value,ast.List) and all(isinstance(x,ast.Tuple) for x in node.value.elts):
                self.note(node,ast.unparse(node.value),'[(500, 24), (2000, 24)]');node.value=ast.parse('[(500,24),(2000,24)]',mode='eval').body
            if name=='Ns' and isinstance(node.value,ast.List):
                self.note(node,ast.unparse(node.value),'[500, 1000, 2000]');node.value=ast.parse('[500,1000,2000]',mode='eval').body
        return node
    def visit_FunctionDef(self,node):
        node=self.generic_visit(node)
        caps={'N':2000,'R':24,'n_iter':4000}
        params=node.args.args[-len(node.args.defaults):] if node.args.defaults else []
        for arg,val in zip(params,node.args.defaults):
            if arg.arg in caps and isinstance(val,ast.Constant) and isinstance(val.value,(int,float)) and val.value>caps[arg.arg]:
                self.note(node,str(val.value),str(caps[arg.arg]));val.value=caps[arg.arg]
        return node
    def visit_For(self,node):
        node=self.generic_visit(node)
        if isinstance(node.target,ast.Name) and node.target.id=='N' and isinstance(node.iter,ast.List):
            self.note(node,ast.unparse(node.iter),'[300, 600]');node.iter=ast.parse('[300,600]',mode='eval').body
        return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        for kw in node.keywords:
            if kw.arg=='processes':
                self.note(node,ast.unparse(kw.value),'2');kw.value=ast.Constant(2)
            if kw.arg=='n_iter' and isinstance(kw.value,ast.Constant) and kw.value.value>4000:
                self.note(node,ast.unparse(kw.value),'4000');kw.value=ast.Constant(4000)
        if isinstance(node.func,ast.Name) and node.func.id=='range' and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value in [1500,2000]:
            self.note(node,ast.unparse(node),'range(40)');node.args=[ast.Constant(40)]
        return node

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--only',nargs='+')
    ap.add_argument('--output',default='validation.json')
    ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--timeout',type=int,default=540)
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1]
    names=a.only or sorted(p.name for p in (repo/'theory_checks').glob('*.py') if p.name not in {'run_checks.py','verification_summary.py'})
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg',PYTHONIOENCODING='utf-8')
    def run(name):
        if Path(name).name!=name or not name.endswith('.py'):raise ValueError(name)
        with tempfile.TemporaryDirectory(prefix='ln-v121-check-') as td:
            dst=Path(td)/'repo';shutil.copytree(repo,dst,ignore=shutil.ignore_patterns('.git','__pycache__'))
            f=dst/'theory_checks'/name;changes=[]
            if a.smoke and name in HEAVY:
                tr=Reduce();tree=tr.visit(ast.parse(f.read_text(encoding='utf-8')));ast.fix_missing_locations(tree)
                f.write_text(ast.unparse(tree)+'\n',encoding='utf-8');changes=tr.changes
            e=env.copy();e.update(PYTHONPATH=str(dst),DZ_CORE_DIR=str(dst))
            start=time.monotonic()
            try:
                cp=subprocess.run([sys.executable,str(f)],cwd=f.parent,env=e,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=a.timeout)
                result={'script':name,'mode':'reduced execution smoke' if changes else 'full','changes':changes,'exit_code':cp.returncode,'seconds':round(time.monotonic()-start,3),'output':cp.stdout.replace(str(dst),'<TEMP_REPO>')}
            except subprocess.TimeoutExpired as ex:
                out=ex.stdout or b''
                if isinstance(out,bytes):out=out.decode('utf-8',errors='replace')
                result={'script':name,'mode':'reduced execution smoke' if changes else 'full','changes':changes,'exit_code':None,'timeout_seconds':a.timeout,'output':out.replace(str(dst),'<TEMP_REPO>')}
            print(name, result['mode'], result['exit_code'],flush=True)
            return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as pool:results=list(pool.map(run,names))
    Path(a.output).write_text(json.dumps({'smoke':a.smoke,'results':results},indent=2)+'\n',encoding='utf-8')
    return int(any(x['exit_code']!=0 for x in results))
if __name__=='__main__':raise SystemExit(main())
