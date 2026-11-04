import os, sys, subprocess, json
T=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')  # repository root
sys.path.insert(0, T+'/tools')
from kfold import FOLD_SETS
for f in FOLD_SETS['thin']:
    for drop in ('', '15', '10,15'):
        for simdir in ('', 'data/train_sim_t'):
            env=dict(os.environ, **f['env']); 
            if simdir: env['REALISTIX_SIM_DIR']=simdir
            code=f"""
import sys; sys.path.insert(0,'{T}/src')
from realistix.splits import partition
r=partition('real'); s=partition('sim'); drop={{int(v) for v in '{drop}'.split(',') if v}}
tr=[t for t in r['train'] if t.aoa not in drop]
print(len(tr), sorted({{t.aoa for t in tr}}), len(r['val_aoa']), sorted({{t.aoa for t in r['val_aoa']}}), len(s['train']), sum(t.aoa==20 for t in s['train']))"""
            out=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True)
            print(f"{f['name']:5s} drop={drop or '-':6s} simdir={'transported' if simdir else 'release':11s} -> real train n, angles, eval n, eval angles, sim train n, sim@20: {out.stdout.strip() or out.stderr[-300:]}")
