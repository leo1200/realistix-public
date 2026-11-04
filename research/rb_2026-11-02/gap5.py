"""Can an effective (Re, AoA) remap -- the two operating knobs a calibrated simulator would expose -- close the sim-real
gap? The 100 release sims, bilinearly interpolated in (log Re, AoA), stand in for a simulator evaluated at any
operating point. For each real cell: error of sim at (k*Re, AoA+d) for a grid of (k, d).
  per-cell best   : oracle remap per cell (upper bound on what two knobs can do)
  per-angle best  : one (k, d) per measured angle, fitted on that angle's cells (in-sample)
  leave-angle-out : (k, d) for a held-out angle interpolated from the fits at the other angles (the novel-angle test)"""
import os, glob, h5py, numpy as np
R='data/RealPDE-Competition-Data/training_data'
CACHE='research/rb_2026-11-02/moments.npz'
if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True); mom=z['mom'].item()
else:
    mom={}
    for kind in ('real','sim'):
        for p in sorted(glob.glob(f'{R}/train_{kind}/*.h5')):
            c=os.path.basename(p)[:-3]
            if kind=='real' and c=='7575_0': continue
            with h5py.File(p,'r') as h: u=h['u'][:, ::2, ::2].astype(np.float64); v=h['v'][:, ::2, ::2].astype(np.float64)
            ok=((np.abs(u)>=1e-8)|(np.abs(v)>=1e-8)).all(0)
            mom[(kind,c)]=dict(m=np.stack([u.mean(0),v.mean(0)]), s=np.stack([(u*u).mean(0),(v*v).mean(0)]), ok=ok)
    np.savez(CACHE, mom=np.array(mom,dtype=object)); print('cached', CACHE, flush=True)
real={c:t for (k,c),t in mom.items() if k=='real'}; sim={c:t for (k,c),t in mom.items() if k=='sim'}
REs=sorted({int(c.split('_')[0]) for c in sim}); A=[0,5,10,15,20]; lRE=np.log(REs)
def norm(t,M):
    s=np.sqrt(t['s'][:,M].sum(0).mean()); m=t['m'][:,M]/s
    return np.stack([m, np.sqrt(np.clip(t['s'][:,M]/s**2-m**2,0,None))])
rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
def corners(re_e, a_e):
    if not (REs[0]<=re_e<=REs[-1] and 0<=a_e<=20): return None
    i=min(np.searchsorted(lRE,np.log(re_e),side='right')-1,len(REs)-2); wr=(np.log(re_e)-lRE[i])/(lRE[i+1]-lRE[i])
    j=min(int(a_e//5),3); wa=(a_e-5*j)/5
    out=[]
    for ii,w1 in ((i,1-wr),(i+1,wr)):
        for jj,w2 in ((j,1-wa),(j+1,wa)):
            if w1*w2>1e-9: out.append((w1*w2, sim[f'{REs[ii]}_{A[jj]}']))
    return out
KS=np.round(np.arange(0.6,1.61,0.1),2); DS=np.arange(-5,5.01,1.0)
cells=sorted(real); err={}
for c in cells:
    re_,a=map(int,c.split('_')); E=np.full((len(KS),len(DS),2),np.nan)
    for ki,k in enumerate(KS):
        for di,d in enumerate(DS):
            cs=corners(re_*k,a+d)
            if cs is None: continue
            M=real[c]['ok'].copy()
            for _,t in cs: M&=t['ok']
            est=sum(w*norm(t,M) for w,t in cs); tgt=norm(real[c],M)
            E[ki,di]=[rel(est[0],tgt[0]),rel(est[1],tgt[1])]
    err[c]=E
k0,d0=list(KS).index(1.0),list(DS).index(0.0)
med=lambda v: np.nanmedian(np.array(v),0)
print("median rel err, mean flow / fluct RMS")
print("  sim at nominal (Re, AoA)         : %.3f / %.3f"%tuple(med([err[c][k0,d0] for c in cells])))
print("  per-cell best remap (oracle)     : %.3f / %.3f"%tuple(med([err[c][np.unravel_index(np.nanargmin(err[c][...,0]),err[c].shape[:2])] for c in cells])))
fit={}
for a in A:
    cs=[c for c in cells if c.endswith(f'_{a}')]
    tot=np.nanmean(np.stack([err[c][...,0] for c in cs]),0); ki,di=np.unravel_index(np.nanargmin(tot),tot.shape)
    fit[a]=(KS[ki],DS[di]); e=med([err[c][ki,di] for c in cs])
    print(f"  per-angle fit  AoA {a:>2}: k={KS[ki]:.1f} d={DS[di]:+.0f}  -> {e[0]:.3f} / {e[1]:.3f}  (nominal {med([err[c][k0,d0] for c in cs])[0]:.3f})")
loo=[]
for a in (5,10,15):   # interior angles: interpolate the neighbours' fitted (k, d)
    k=round(np.mean([fit[a-5][0],fit[a+5][0]]),1); d=np.mean([fit[a-5][1],fit[a+5][1]])
    ki=int(np.argmin(abs(KS-k))); di=int(np.argmin(abs(DS-d)))
    for c in cells:
        if c.endswith(f'_{a}'): loo.append(err[c][ki,di])
print("  leave-angle-out (5/10/15 from neighbours' fits): %.3f / %.3f"%tuple(med(loo)))
print("  bars: real AoA interpolation 0.100 / 0.313; sim + adjacent-angle additive correction 0.180 / 0.613")
