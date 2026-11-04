"""Is the release sim a better proxy for real at a cell than real neighbours are -- and does its *trend* help?

Per trajectory: per-cell E[u],E[v],E[u^2],E[v^2] and a validity mask on the 32x64 grid. Each field is normalised by its
own RMS speed over the mask shared by every trajectory in a comparison, so dimensional real and non-dimensional sim meet.
Statistics compared: time-mean map (2ch) and fluctuation-RMS map (2ch), relative L2 against the real target.
  plain    real neighbour(s)                         -> real_c      (interpolation / nearest-neighbour)
  sim      sim_c                                     -> real_c      (sim at target as teacher)
  delta    real neighbour(s) + (sim_c - sim neighbour(s)) -> real_c (sim supplies only the regime trend)
"""
import os, glob, h5py, numpy as np
R='data/RealPDE-Competition-Data/training_data'
def mom(p):
    with h5py.File(p,'r') as h: u=h['u'][:, ::2, ::2].astype(np.float64); v=h['v'][:, ::2, ::2].astype(np.float64)
    ok=((np.abs(u)>=1e-8)|(np.abs(v)>=1e-8)).all(0)
    return dict(m=np.stack([u.mean(0),v.mean(0)]), s=np.stack([(u*u).mean(0),(v*v).mean(0)]), ok=ok)
real={os.path.basename(p)[:-3]:mom(p) for p in sorted(glob.glob(f'{R}/train_real/*.h5')) if not p.endswith('7575_0.h5')}
sim={os.path.basename(p)[:-3]:mom(p) for p in sorted(glob.glob(f'{R}/train_sim/*.h5'))}
def norm(t,M):
    s=np.sqrt(t['s'][:,M].sum(0).mean()); m=t['m'][:,M]/s
    f=np.sqrt(np.clip(t['s'][:,M]/s**2-m**2,0,None)); return m,f
def stats(terms,M):  # terms: list of (coef, traj)
    out=[0,0]
    for c,t in terms:
        m,f=norm(t,M); out[0]=out[0]+c*m; out[1]=out[1]+c*f
    return out
rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
RES=sorted({int(k.split('_')[0]) for k in sim}); AOAS=[0,5,10,15,20]
rows=[]
for c in real:
    re_,a=map(int,c.split('_')); i=RES.index(re_); j=AOAS.index(a)
    for axis,(lo,hi) in (('Re',(f'{RES[i-1]}_{a}' if i>0 else None, f'{RES[i+1]}_{a}' if i+1<len(RES) else None)),
                         ('AoA',(f'{re_}_{AOAS[j-1]}' if j>0 else None, f'{re_}_{AOAS[j+1]}' if j+1<len(AOAS) else None))):
        nb=[n for n in (lo,hi) if n and n in real]
        if not nb: continue
        kind='bracket' if len(nb)==2 else 'one-sided'
        M=real[c]['ok']&sim[c]['ok']
        for n in nb: M=M&real[n]['ok']&sim[n]['ok']
        w=1/len(nb)
        tgt=stats([(1,real[c])],M)
        pl=stats([(w,real[n]) for n in nb],M)
        sm=stats([(1,sim[c])],M)
        de=stats([(w,real[n]) for n in nb]+[(1,sim[c])]+[(-w,sim[n]) for n in nb],M)
        rows.append((axis,kind,*[rel(x[k],tgt[k]) for x in (pl,sm,de) for k in (0,1)]))
rows=np.array(rows,dtype=object)
print(f"{'axis':>4} {'kind':>9} {'n':>3} | mean-flow rel err: plain / sim / delta | fluct-RMS rel err: plain / sim / delta | delta beats plain (mean, fluct)")
for axis in ('Re','AoA'):
    for kind in ('bracket','one-sided'):
        sel=[r for r in rows if r[0]==axis and r[1]==kind]
        if not sel: continue
        A=np.array([r[2:] for r in sel],float); md=np.median(A,0)
        print(f"{axis:>4} {kind:>9} {len(sel):>3} |          {md[0]:.3f} / {md[2]:.3f} / {md[4]:.3f}      |          {md[1]:.3f} / {md[3]:.3f} / {md[5]:.3f}       |   {np.mean(A[:,4]<A[:,0]):.0%}, {np.mean(A[:,5]<A[:,1]):.0%}")
print("(medians over target cells; lower = closer to the real target)")
