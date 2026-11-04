import os, glob, h5py, numpy as np
from scipy.ndimage import gaussian_filter
exec(open('gap.py').read().split("real={")[0])   # reuse imports, R, mom
def mom_f(p, sig):
    with h5py.File(p,'r') as h: u=h['u'][:, ::2, ::2].astype(np.float64); v=h['v'][:, ::2, ::2].astype(np.float64)
    ok=((np.abs(u)>=1e-8)|(np.abs(v)>=1e-8)).all(0)
    if sig:
        w=gaussian_filter(ok.astype(float),sig)+1e-12
        u=gaussian_filter(u*ok,(0,sig,sig))/w*ok; v=gaussian_filter(v*ok,(0,sig,sig))/w*ok
    return dict(m=np.stack([u.mean(0),v.mean(0)]), s=np.stack([(u*u).mean(0),(v*v).mean(0)]), ok=ok)
real={os.path.basename(p)[:-3]:mom(p) for p in sorted(glob.glob(f'{R}/train_real/*.h5')) if not p.endswith('7575_0.h5')}
cells=sorted(real)
def norm(t,M):
    s=np.sqrt(t['s'][:,M].sum(0).mean()); m=t['m'][:,M]/s
    return m, np.sqrt(np.clip(t['s'][:,M]/s**2-m**2,0,None))
rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
print("sim at target vs real, after a Gaussian spatial filter on the sim (32x64 cells): median rel err, mean-flow / fluct-RMS")
sims={}
for sig in (0,1,2):
    sims[sig]={c:mom_f(f'{R}/train_sim/{c}.h5',sig) for c in cells}
    e=[]
    for c in cells:
        M=real[c]['ok']&sims[sig][c]['ok']; a=norm(sims[sig][c],M); b=norm(real[c],M)
        e.append((rel(a[0],b[0]),rel(a[1],b[1]), np.linalg.norm(a[1])/np.linalg.norm(b[1])))
    e=np.median(np.array(e),0); print(f"  sigma={sig}: {e[0]:.3f} / {e[1]:.3f}   (sim fluct energy / real = {e[2]:.2f}x)")
# scaled trend: plain + rho*(sim_c - mean sim_nb), Re and AoA brackets
sim=sims[0]; RES=sorted({int(k.split('_')[0]) for k in sim} | {27975,15225}); AOAS=[0,5,10,15,20]
allsim={os.path.basename(p)[:-3]:mom(p) for p in glob.glob(f'{R}/train_sim/*.h5')}
print("regime trend from sim, scaled by rho (bracketed cells): median rel err, mean-flow / fluct-RMS")
for axis in ('Re','AoA'):
    res={}
    for c in cells:
        re_,a=map(int,c.split('_')); i=RES.index(re_); j=AOAS.index(a)
        nb=[f'{RES[i-1]}_{a}',f'{RES[i+1]}_{a}'] if axis=='Re' and 0<i<len(RES)-1 else ([f'{re_}_{AOAS[j-1]}',f'{re_}_{AOAS[j+1]}'] if axis=='AoA' and 0<j<4 else [])
        if len(nb)<2 or not all(n in real for n in nb): continue
        M=real[c]['ok']&allsim[c]['ok']
        for n in nb: M=M&real[n]['ok']&allsim[n]['ok']
        T=norm(real[c],M); P=[norm(real[n],M) for n in nb]; S=norm(allsim[c],M); SN=[norm(allsim[n],M) for n in nb]
        for rho in (0,0.1,0.25,0.5,1.0):
            est=[0.5*(P[0][k]+P[1][k])+rho*(S[k]-0.5*(SN[0][k]+SN[1][k])) for k in (0,1)]
            res.setdefault(rho,[]).append((rel(est[0],T[0]),rel(est[1],T[1])))
    print(f"  {axis}: "+"  ".join(f"rho={r}: {np.median(np.array(v),0)[0]:.3f}/{np.median(np.array(v),0)[1]:.3f}" for r,v in res.items()))
