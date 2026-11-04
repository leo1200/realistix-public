import json, numpy as np
from scipy.stats import spearmanr
rows=json.load(open('clock_cells.json'))
re=np.array([r['re'] for r in rows]); mis=np.abs(np.log([r['ratio'] for r in rows]))
print("cell  Re     clock ratio  |log|")
for r,m in zip(rows,mis): print(f"     {r['re']:>6}   {r['ratio']:.2f}      {m:.2f}")
C={'sim where real remains (real - rawp)':('real','rawp'),'all raw sim, none@20 (real - rawx)':('real','rawx'),
   'sim at 20 deg (rawx - rawk)':('rawx','rawk'),'transported vs raw (rawk - trk)':('rawk','trk')}
print("\nrelative rel_l2 gain per cell (positive = the second arm is better); Spearman vs clock mismatch |log ratio| and vs Re")
for name,(a,b) in C.items():
    allg,allm,allr=[],[],[]
    line=f"{name:40s}"
    for d in ('05','10','15'):
        g=np.array([(r['err'][f'th{d}{a}']-r['err'][f'th{d}{b}'])/r['err'][f'th{d}real'] for r in rows])
        rm=spearmanr(mis,g)[0]; rr=spearmanr(re,g)[0]
        line+=f" | d{d}: mean {g.mean()*100:+5.1f}%  rho(mis) {rm:+.2f}  rho(Re) {rr:+.2f}"
        allg+=list(g); allm+=list(mis); allr+=list(re)
    rho,p=spearmanr(allm,allg); print(line + f" || pooled rho(mis) {rho:+.2f} (p {p:.2f}), rho(Re) {spearmanr(allr,allg)[0]:+.2f}")
lo=mis<np.median(mis)
for name,(a,b) in C.items():
    g=np.array([[(r['err'][f'th{d}{a}']-r['err'][f'th{d}{b}'])/r['err'][f'th{d}real'] for d in ('05','10','15')] for r in rows])
    print(f"  {name:40s} clock close (|log|<{np.median(mis):.2f}, n={lo.sum()}): {g[lo].mean()*100:+5.1f}%   clock off (n={(~lo).sum()}): {g[~lo].mean()*100:+5.1f}%")
