import os, glob, h5py, numpy as np
from scipy.signal import welch
P='data/RealPDE-Competition-Data/training_data'; TS='data/train_sim_t'
YS,XS=(24,32,40),(80,100,120)
def spec(p):
    with h5py.File(p,'r') as h: v=np.stack([h['v'][:,y,x] for y in YS for x in XS],1).astype(np.float64)
    v=v[:,np.abs(v).max(0)>1e-8]; v-=v.mean(0)
    f,S=welch(v,nperseg=min(512,len(v)),axis=0); S=S.mean(1); k=(f>0.006)&(f<0.25)
    lag1=np.mean([np.corrcoef(c[:-1],c[1:])[0,1] for c in v.T])
    return f[k][np.argmax(S[k])], S[f>=0.25].sum()/S[f>0.006].sum(), lag1
cells=[os.path.basename(p)[:-3] for p in sorted(glob.glob(f'{P}/train_real/*.h5')) if not p.endswith('7575_0.h5')][::2]
R=np.array([[*spec(f'{P}/train_real/{c}.h5'),*spec(f'{P}/train_sim/{c}.h5'),*spec(f'{TS}/{c}.h5')] for c in cells])
print(f"({len(cells)} cells) peak / real peak: sim {np.median(R[:,3]/R[:,0]):.2f}, transported {np.median(R[:,6]/R[:,0]):.2f}")
print(f"high-frequency energy fraction (>= 0.25 cycles/frame): real {np.median(R[:,1]):.3f}, sim {np.median(R[:,4]):.3f}, transported {np.median(R[:,7]):.3f}")
print(f"lag-1 frame autocorrelation of v: real {np.median(R[:,2]):.3f}, sim {np.median(R[:,5]):.3f}, transported {np.median(R[:,8]):.3f}")
