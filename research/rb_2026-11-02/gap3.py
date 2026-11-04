"""What is the sim-real gap made of? (1) sampling-noise floors (split halves), (2) how much a systematic, calibratable
correction removes (leave-one-cell-out mean gap: global, same AoA, same Re), (3) where in (Re, AoA) the gap sits."""
import os, glob, h5py, numpy as np
R='data/RealPDE-Competition-Data/training_data'
def load(p):
    with h5py.File(p,'r') as h: u=h['u'][:, ::2, ::2].astype(np.float64); v=h['v'][:, ::2, ::2].astype(np.float64)
    ok=((np.abs(u)>=1e-8)|(np.abs(v)>=1e-8)).all(0)
    m=lambda a,b: dict(m=np.stack([u[a:b].mean(0),v[a:b].mean(0)]), s=np.stack([(u[a:b]**2).mean(0),(v[a:b]**2).mean(0)]))
    T=len(u); return dict(full=m(0,T), h1=m(0,T//2), h2=m(T//2,T), ok=ok, T=T)
cells=[os.path.basename(p)[:-3] for p in sorted(glob.glob(f'{R}/train_real/*.h5')) if not p.endswith('7575_0.h5')]
real={c:load(f'{R}/train_real/{c}.h5') for c in cells}
sim={c:load(f'{R}/train_sim/{c}.h5') for c in cells}
def norm(t,M):
    s=np.sqrt(t['s'][:,M].sum(0).mean()); m=t['m'][:,M]/s
    return np.stack([m, np.sqrt(np.clip(t['s'][:,M]/s**2-m**2,0,None))])   # (2 stats, 2 ch, n)
rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
print("(1) split-half difference of each trajectory's own statistics (mean flow / fluct RMS); a full-length estimate's noise is about half this")
for name,d in (('real',real),('sim',sim)):
    e=np.median([[rel(norm(t['h1'],t['ok'])[k],norm(t['h2'],t['ok'])[k]) for k in (0,1)] for t in d.values()],0)
    print(f"  {name:4s}: {e[0]:.3f} / {e[1]:.3f}   frames per trajectory {np.median([t['T'] for t in d.values()]):.0f}")
H,W=real[cells[0]]['ok'].shape
G={}; NS={}; NR={}
for c in cells:
    M=real[c]['ok']&sim[c]['ok']; g=np.full((2,2,H,W),np.nan)
    r=norm(real[c]['full'],M); s=norm(sim[c]['full'],M)
    g[:,:,M]=r-s; G[c]=g; NS[c]=(s,M); NR[c]=r
groups={'raw':lambda c,o: False,'global':lambda c,o: True,
        'same AoA':lambda c,o: o.split('_')[1]==c.split('_')[1],'same Re':lambda c,o: o.split('_')[0]==c.split('_')[0]}
print("\n(2) sim at target + leave-one-cell-out mean gap (a calibration learned on other cells): median rel err, mean flow / fluct RMS")
res={}
for gname,f in groups.items():
    e=[]
    for c in cells:
        s,M=NS[c]; others=[G[o] for o in cells if o!=c and f(c,o)]
        corr=np.nan_to_num(np.nanmean(np.stack(others),0)[:,:,M]) if others else 0
        est=s+corr; e.append([rel(est[k],NR[c][k]) for k in (0,1)])
    res[gname]=np.array(e); md=np.median(e,0); print(f"  {gname:9s}: {md[0]:.3f} / {md[1]:.3f}")
print("  (bars to beat: real Re-neighbour interpolation 0.027 / 0.136; real AoA-neighbour interpolation 0.100 / 0.313)")
print("\n(3) raw sim-at-target mean-flow error by AoA and by Re band; and after same-AoA calibration")
for a in (0,5,10,15,20):
    i=[k for k,c in enumerate(cells) if c.endswith(f'_{a}')]
    print(f"  AoA {a:>2}: raw {np.median(res['raw'][i,0]):.3f}  calibrated {np.median(res['same AoA'][i,0]):.3f}   (n={len(i)})")
re=np.array([int(c.split('_')[0]) for c in cells])
for lo,hi in ((0,9000),(9000,18000),(18000,30000)):
    i=np.where((re>=lo)&(re<hi))[0]
    print(f"  Re {lo:>5}-{hi:<5}: raw {np.median(res['raw'][i,0]):.3f}  calibrated {np.median(res['same AoA'][i,0]):.3f}   (n={len(i)})")
