"""Does the re-clocked set (data/train_sim_clk) run at the real clock? Peak and centroid of v at nine wake probes,
cycles per frame, release sim and re-clocked sim against real, by Re band and at 20 deg (outside the fit)."""
import os, glob, h5py, numpy as np
from scipy.signal import welch
P='data/RealPDE-Competition-Data/training_data'; CK='data/train_sim_clk'
def spec(p):
    with h5py.File(p,'r') as h: v=np.stack([h['v'][:,y,x] for y in (24,32,40) for x in (80,100,120)],1).astype(np.float64)
    v=v[:,np.abs(v).max(0)>1e-8]; v-=v.mean(0); f,S=welch(v,nperseg=min(512,len(v)),axis=0); S=S.mean(1); k=(f>0.006)&(f<0.25)
    return f[k][np.argmax(S[k])], (S[k]*f[k]).sum()/S[k].sum()
cells=[os.path.basename(p)[:-3] for p in sorted(glob.glob(f'{P}/train_real/*.h5')) if not p.endswith('7575_0.h5')]
A=[]
for c in cells:
    r=spec(f'{P}/train_real/{c}.h5'); s=spec(f'{P}/train_sim/{c}.h5'); k=spec(f'{CK}/{c}.h5')
    A.append((int(c.split('_')[0]),int(c.split('_')[1]),*r,*s,*k))
A=np.array(A,float); re,aoa=A[:,0],A[:,1]
for name,i in (('peak',0),('centroid',1)):
    rs=A[:,4+i]/A[:,2+i]; rk=A[:,6+i]/A[:,2+i]
    print(f"{name:8s} ratio to real: release median {np.median(rs):.2f} (Re slope {np.polyfit(np.log(re),np.log(rs),1)[0]:+.2f}) | re-clocked median {np.median(rk):.2f} (Re slope {np.polyfit(np.log(re),np.log(rk),1)[0]:+.2f})")
    for lo,hi in ((0,7000),(7000,14000),(14000,21000),(21000,30000)):
        m=(re>=lo)&(re<hi); print(f"   Re {lo:>5}-{hi:<5} n={m.sum():>2}: release {np.median(rs[m]):.2f}  re-clocked {np.median(rk[m]):.2f}")
    m=aoa==20; print(f"   AoA 20 (not in the fit) n={m.sum()}: release {np.median(rs[m]):.2f}  re-clocked {np.median(rk[m]):.2f}")
