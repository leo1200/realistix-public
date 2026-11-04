"""Time scales as the forecaster sees them: dominant frequency of the cross-flow velocity in the wake, in cycles per
frame (dt 0.02 in every set), real vs release sim vs transported sim at the same cell. Welch PSD of v at 9 wake probes
on the raw 64x128 grid, averaged; peak above 4 cycles per record."""
import os, glob, h5py, numpy as np
from scipy.signal import welch
P='data/RealPDE-Competition-Data/training_data'; TS='data/train_sim_t'
YS,XS=(24,32,40),(80,100,120)
def fpeak(p):
    with h5py.File(p,'r') as h:
        v=np.stack([h['v'][:,y,x] for y in YS for x in XS],1).astype(np.float64)
    v=v[:, np.abs(v).max(0)>1e-8]
    if v.shape[1]==0: return np.nan, np.nan
    v-=v.mean(0); f,S=welch(v,nperseg=256,axis=0); S=S.mean(1); k=f>4/256
    i=np.argmax(S[k]); return f[k][i], S[k][i]/np.median(S[k])
cells=[os.path.basename(p)[:-3] for p in sorted(glob.glob(f'{P}/train_real/*.h5')) if not p.endswith('7575_0.h5')]
rows=[]
for c in cells:
    fr,qr=fpeak(f'{P}/train_real/{c}.h5'); fs,qs=fpeak(f'{P}/train_sim/{c}.h5'); ft,qt=fpeak(f'{TS}/{c}.h5')
    rows.append((c,fr,fs,ft,qr,qs))
R=np.array([r[1:] for r in rows]); ok=np.isfinite(R[:,0])&np.isfinite(R[:,1])&(R[:,3]>3)&(R[:,4]>3)   # clear peaks only
print(f"cells with a clear peak in both: {ok.sum()} of {len(rows)}")
print(f"real peak, cycles/frame: median {np.median(R[ok,0]):.4f} (range {R[ok,0].min():.4f}-{R[ok,0].max():.4f})")
print(f"sim / real peak frequency : median {np.median(R[ok,1]/R[ok,0]):.2f}, IQR {np.percentile(R[ok,1]/R[ok,0],25):.2f}-{np.percentile(R[ok,1]/R[ok,0],75):.2f}")
print(f"transported / real        : median {np.median(R[ok,2]/R[ok,0]):.2f}, IQR {np.percentile(R[ok,2]/R[ok,0],25):.2f}-{np.percentile(R[ok,2]/R[ok,0],75):.2f}")
re=np.array([int(r[0].split('_')[0]) for r in rows]); aoa=np.array([int(r[0].split('_')[1]) for r in rows])
print(f"corr with Re  : real {np.corrcoef(re[ok],R[ok,0])[0,1]:+.2f}, sim {np.corrcoef(re[ok],R[ok,1])[0,1]:+.2f}")
print(f"corr with AoA : real {np.corrcoef(aoa[ok],R[ok,0])[0,1]:+.2f}, sim {np.corrcoef(aoa[ok],R[ok,1])[0,1]:+.2f}")
print(f"corr real vs sim peak across cells: {np.corrcoef(R[ok,0],R[ok,1])[0,1]:+.2f}")
for a in (0,5,10,15,20):
    m=ok&(aoa==a)
    if m.sum(): print(f"  AoA {a:>2}: n={m.sum():>2}  real {np.median(R[m,0]):.4f}  sim {np.median(R[m,1]):.4f}  transported {np.nanmedian(R[m,2]):.4f}")
np.save('freq_rows.npy', np.array(rows,dtype=object))
