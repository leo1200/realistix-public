"""Static crossover: extrapolating to a novel angle as real coverage recedes. Target angle a* in {20, 0}; real data
kept only at angles at least d away. Estimates of real at (Re, a*), median rel err over the Re that have a real a* cell:
  nearest real  : real at (Re, a* -/+ d), the closest kept angle (what interpolation degrades to)
  raw sim       : release sim at (Re, a*)
  transported   : data/train_sim_t at (Re, a*) -- the Aug-25 flow map never saw real at 20 deg (leak-free there; at 0 deg
                  it trained on the target, so in-sample and only an upper bound)
  sim + gap@d   : raw sim at a* + (real - sim) at the nearest kept angle, same Re (a calibration that can recede too)"""
import os, glob, h5py, numpy as np
R='data/RealPDE-Competition-Data/training_data'
z=np.load('research/rb_2026-11-02/moments.npz',allow_pickle=True); mom=z['mom'].item()
TC='research/rb_2026-11-02/moments_simt.npz'
if os.path.exists(TC): tm=np.load(TC,allow_pickle=True)['mom'].item()
else:
    tm={}
    for a in (0,20):
        for p in sorted(glob.glob(f'data/train_sim_t/*_{a}.h5')):
            with h5py.File(p,'r') as h: u=h['u'][:, ::2, ::2].astype(np.float64); v=h['v'][:, ::2, ::2].astype(np.float64)
            ok=((np.abs(u)>=1e-8)|(np.abs(v)>=1e-8)).all(0)
            tm[os.path.basename(p)[:-3]]=dict(m=np.stack([u.mean(0),v.mean(0)]), s=np.stack([(u*u).mean(0),(v*v).mean(0)]), ok=ok)
    np.savez(TC, mom=np.array(tm,dtype=object))
real={c:t for (k,c),t in mom.items() if k=='real'}; sim={c:t for (k,c),t in mom.items() if k=='sim'}
def norm(t,M):
    s=np.sqrt(t['s'][:,M].sum(0).mean()); m=t['m'][:,M]/s
    return np.stack([m, np.sqrt(np.clip(t['s'][:,M]/s**2-m**2,0,None))])
rel=lambda a,b: np.linalg.norm(a-b)/np.linalg.norm(b)
for tgt in (20,0):
    print(f"target angle {tgt} deg   (median rel err, mean flow / fluct RMS; n = Re with real at both angles)")
    for d in (5,10,15,20):
        src=tgt-d if tgt==20 else tgt+d
        rows=[]
        for c in real:
            re_,a=c.split('_')
            if int(a)!=tgt or f'{re_}_{src}' not in real or c not in tm: continue
            n=f'{re_}_{src}'
            M=real[c]['ok']&sim[c]['ok']&real[n]['ok']&sim[n]['ok']&tm[c]['ok']
            T=norm(real[c],M)
            est=dict(nearest=norm(real[n],M), raw=norm(sim[c],M), transported=norm(tm[c],M),
                     simgap=norm(sim[c],M)+norm(real[n],M)-norm(sim[n],M))
            rows.append({k:[rel(v[i],T[i]) for i in (0,1)] for k,v in est.items()})
        md={k:np.median([r[k] for r in rows],0) for k in rows[0]}
        print(f"  real kept >= {d:>2} deg away (n={len(rows):>2}): nearest real {md['nearest'][0]:.3f}/{md['nearest'][1]:.3f}"
              f" | raw sim {md['raw'][0]:.3f}/{md['raw'][1]:.3f} | transported {md['transported'][0]:.3f}/{md['transported'][1]:.3f}"
              f" | sim+gap@d {md['simgap'][0]:.3f}/{md['simgap'][1]:.3f}")
