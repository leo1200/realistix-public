"""Robust time-scale check: spectral centroid of v in the wake band 0.006-0.25 cycles/frame (nperseg 512), real vs
release sim vs transported, and how the sim/real ratio scales with Re (convective vs physical time units predict
ratio ~ 1/Re: a sim at U=1 for every Re against a tunnel whose speed sets Re)."""
import os, glob, h5py, numpy as np
from scipy.signal import welch
P='data/RealPDE-Competition-Data/training_data'; TS='data/train_sim_t'
YS,XS=(24,32,40),(80,100,120)
def spec(p):
    with h5py.File(p,'r') as h:
        v=np.stack([h['v'][:,y,x] for y in YS for x in XS],1).astype(np.float64)
        u=np.stack([h['u'][:,y,x] for y in YS for x in XS],1).astype(np.float64)
    keep=np.abs(v).max(0)>1e-8; v=v[:,keep]-v[:,keep].mean(0)
    f,S=welch(v,nperseg=min(512,len(v)),axis=0); S=S.mean(1); k=(f>0.006)&(f<0.25)
    return (S[k]*f[k]).sum()/S[k].sum(), f[k][np.argmax(S[k])], np.abs(u[:,keep]).mean()
cells=[os.path.basename(p)[:-3] for p in sorted(glob.glob(f'{P}/train_real/*.h5')) if not p.endswith('7575_0.h5')]
rows=[]
for c in cells:
    r=spec(f'{P}/train_real/{c}.h5'); s=spec(f'{P}/train_sim/{c}.h5'); t=spec(f'{TS}/{c}.h5')
    rows.append((int(c.split('_')[0]),int(c.split('_')[1]),r[0],s[0],t[0],r[1],s[1],r[2],s[2]))
A=np.array(rows,float); re,aoa,cr,cs,ct=A[:,0],A[:,1],A[:,2],A[:,3],A[:,4]
print(f"spectral centroid, cycles/frame: real median {np.median(cr):.4f}, sim {np.median(cs):.4f}, transported {np.median(ct):.4f}")
print(f"sim/real centroid ratio: median {np.median(cs/cr):.2f}, IQR {np.percentile(cs/cr,25):.2f}-{np.percentile(cs/cr,75):.2f}; transported/real {np.median(ct/cr):.2f}")
b=np.polyfit(np.log(re),np.log(cs/cr),1)[0]; br=np.polyfit(np.log(re),np.log(cr),1)[0]; bs=np.polyfit(np.log(re),np.log(cs),1)[0]
print(f"log-log slope vs Re: real centroid {br:+.2f}, sim centroid {bs:+.2f}, sim/real ratio {b:+.2f}   (convective-vs-physical time predicts real ~ +1, sim ~ 0, ratio ~ -1)")
print(f"real mean |u| at probes vs Re slope {np.polyfit(np.log(re),np.log(A[:,7]),1)[0]:+.2f}, sim {np.polyfit(np.log(re),np.log(A[:,8]),1)[0]:+.2f}")
for lo,hi in ((0,7000),(7000,14000),(14000,21000),(21000,30000)):
    m=(re>=lo)&(re<hi); print(f"  Re {lo:>5}-{hi:<5} n={m.sum():>2}: sim/real centroid {np.median(cs[m]/cr[m]):.2f}  transported/real {np.median(ct[m]/cr[m]):.2f}")
