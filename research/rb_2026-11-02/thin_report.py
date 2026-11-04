"""Read the real-coverage dose (NOTES §3c): accuracy-only fold score per arm, level and seed; paired contrasts by seed."""
import os, sys, numpy as np
T=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')  # repository root
sys.path.insert(0, T+'/tools'); sys.path.insert(0, T+'/src')
import kfold
from kfold import FOLD_SETS, RUNS, score_of, W_ACC
F={f['name']:f for f in FOLD_SETS['thin']}
ARMS={'real':'a20','rawp':'a20','rawx':'a20','rawk':'a20k','trk':'a20k','clkx':'a20','clkk':'a20k','clkp':'a20'}
S={}
for d in ('05','10','15'):
    for arm,fold in ARMS.items():
        for s in (0,1,2):
            S[(d,arm,s)]=score_of(os.path.join(RUNS,f'kf_th{d}{arm}_{fold}_s{s}'), F[fold], W_ACC)
print(f"{'d':>3} " + "".join(f"{a:>22}" for a in ARMS) + "   (accuracy-only score on 20 deg; per seed)")
for d in ('05','10','15'):
    cells=[]
    for a in ARMS:
        v=[S[(d,a,s)] for s in (0,1,2)]
        cells.append(" ".join('  -   ' if x is None else f"{x:6.2f}" for x in v))
    print(f"{d:>3} " + "".join(f"{c:>22}" for c in cells))
print("\ncontrasts, paired by seed (mean +/- SE over seeds that both sides finished; n positive / n)")
for name,(a,b) in {'regularisation rawp-real':('rawp','real'),'sim at removed angles rawx-rawp':('rawx','rawp'),'transfer at target rawk-rawx':('rawk','rawx'),'re-clocked, none@20 clkx-rawx':('clkx','rawx'),'re-clocked at removed angles clkx-clkp':('clkx','clkp'),'re-clocked regularisation clkp-real':('clkp','real'),'paired: re-clocked - raw clkp-rawp':('clkp','rawp'),'re-clocked, incl. 20 clkk-rawk':('clkk','rawk'),'transfer, re-clocked clkk-clkx':('clkk','clkx'),'regulariser, re-clocked clkx-real':('clkx','real'),'realism   trk-rawk':('trk','rawk'),
                   'total     trk-real':('trk','real')}.items():
    row=[]
    for d in ('05','10','15'):
        x=[S[(d,a,s)]-S[(d,b,s)] for s in (0,1,2) if S[(d,a,s)] is not None and S[(d,b,s)] is not None]
        if not x: row.append(f"{'-':>22}"); continue
        x=np.array(x); se=x.std(ddof=1)/np.sqrt(len(x)) if len(x)>1 else float('nan')
        row.append(f"{x.mean():+7.3f} +/- {se:5.3f} ({(x>0).sum()}/{len(x)})")
    print(f"  {name:34s} " + "  ".join(f"d={d}: {r}" for d,r in zip(('5','10','15'),row)))
