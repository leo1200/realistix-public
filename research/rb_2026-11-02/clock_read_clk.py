import json, numpy as np
from scipy.stats import spearmanr
R=json.load(open('clock_cells_clk.json'))
re=np.array([r['re'] for r in R]); mis=np.abs(np.log([r['ratio'] for r in R])); k=np.abs(np.log(re/4900))
g=lambda r,d,a,b:(r['err'][f'th{d}{a}']-r['err'][f'th{d}{b}'])/r['err'][f'th{d}real']*100
print("    Re  raw clock | d15: re-clocked vs raw (none@20)  incl.20 | target: raw  re-clocked | d05: re-clocked vs raw  target re-clocked")
for r in R:
    print(f"{r['re']:>6}   {r['ratio']:5.2f}  |            {g(r,'15','rawx','clkx'):+5.1f}%          {g(r,'15','rawk','clkk'):+5.1f}% |    {g(r,'15','rawx','rawk'):+5.1f}%   {g(r,'15','clkx','clkk'):+5.1f}%  |       {g(r,'05','rawx','clkx'):+5.1f}%            {g(r,'05','clkx','clkk'):+5.1f}%")
for d in ('05','15'):
    for name,(a,b) in {'re-clocked vs raw, none@20':('rawx','clkx'),'target, re-clocked':('clkx','clkk'),'target, raw':('rawx','rawk')}.items():
        x=np.array([g(r,d,a,b) for r in R])
        print(f"d{d} {name:28s} mean {x.mean():+5.2f}%  cells better {int((x>0).sum())}/16  rho vs raw clock mismatch {spearmanr(mis,x)[0]:+.2f}  vs Re {spearmanr(re,x)[0]:+.2f}")
