"""(a) novel-angle calibration: correct sim at c with the mean gap of cells at the ADJACENT angles only (nothing at c's
angle is used), vs real AoA-neighbour interpolation. (b) is the sim's split-half noise a start-up transient?"""
exec(open('gap3.py').read().split('print("(1)')[0])
import numpy as np
H,W=real[cells[0]]['ok'].shape
G={}; NS={}; NR={}
for c in cells:
    M=real[c]['ok']&sim[c]['ok']; g=np.full((2,2,H,W),np.nan); r=norm(real[c]['full'],M); s=norm(sim[c]['full'],M)
    g[:,:,M]=r-s; G[c]=g; NS[c]=(s,M); NR[c]=r
REs=sorted({int(c.split('_')[0]) for c in cells})
def run(sel):
    e=[]
    for c in cells:
        others=[G[o] for o in cells if sel(c,o)]
        if not others: continue
        s,M=NS[c]; corr=np.nan_to_num(np.nanmean(np.stack(others),0)[:,:,M]); est=s+corr
        e.append([rel(est[k],NR[c][k]) for k in (0,1)])
    return np.median(e,0), len(e)
aoa=lambda c:int(c.split('_')[1]); rei=lambda c:REs.index(int(c.split('_')[0]))
print("(a) sim + gap learned at the adjacent angle(s) only: median rel err mean flow / fluct RMS")
for name,sel in (('adjacent angles, all Re',lambda c,o: abs(aoa(o)-aoa(c))==5),
                 ('adjacent angles, Re within 2 steps',lambda c,o: abs(aoa(o)-aoa(c))==5 and abs(rei(o)-rei(c))<=2)):
    md,n=run(sel); print(f"  {name:36s}: {md[0]:.3f} / {md[1]:.3f}  (n={n})")
print("  bars: real AoA neighbours bracketed 0.100 / 0.313, one-sided 0.182 / 0.419")
import h5py
def q(p):
    with h5py.File(p,'r') as h: u=h['u'][:, ::2, ::2].astype(np.float64); v=h['v'][:, ::2, ::2].astype(np.float64)
    ok=((np.abs(u)>=1e-8)|(np.abs(v)>=1e-8)).all(0); T=len(u); k=T//4
    st=lambda a,b: norm(dict(m=np.stack([u[a:b].mean(0),v[a:b].mean(0)]),s=np.stack([(u[a:b]**2).mean(0),(v[a:b]**2).mean(0)])),ok)[0]
    Q=[st(i*k,(i+1)*k) for i in range(4)]
    return rel(Q[0],Q[1]), rel(Q[2],Q[3])
for name,d in (('real','train_real'),('sim','train_sim')):
    r=np.median([q(f'{R}/{d}/{c}.h5') for c in cells[::3]],0)
    print(f"(b) {name}: quarter-to-quarter mean-flow difference, Q1 vs Q2 {r[0]:.3f}, Q3 vs Q4 {r[1]:.3f}")
