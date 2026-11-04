"""Fit the sim->real clock law on cells away from 20 deg. k = sim frames per real frame = f_real / f_sim (cycles per
frame). Physics for a tunnel whose speed sets Re (fixed chord, water) against a convective-unit CFD (U = c = 1):
k = U/c * dt_real / dt_sim = Re * nu / c^2, i.e. k = Re / Re0. Two independent estimates of Re0:
  (a) wake frequencies: dominant v peak at nine probes, real vs sim, slope fixed at 1 (and free, as a check);
  (b) freestream speed: real U from the top and bottom rows (no wake), c = nu * Re / U, Re0 = c^2 / nu."""
import os, glob, h5py, numpy as np
from scipy.signal import welch
P='data/RealPDE-Competition-Data/training_data'
def peak(p):
    with h5py.File(p,'r') as h: v=np.stack([h['v'][:,y,x] for y in (24,32,40) for x in (80,100,120)],1).astype(np.float64)
    v=v[:,np.abs(v).max(0)>1e-8]; v-=v.mean(0); f,S=welch(v,nperseg=min(512,len(v)),axis=0); S=S.mean(1); k=(f>0.006)&(f<0.25)
    return f[k][np.argmax(S[k])]
def ufree(p):
    with h5py.File(p,'r') as h: u=np.concatenate([h['u'][:, 1:4, 20:120], h['u'][:, -4:-1, 20:120]],1).astype(np.float64)
    u=u[np.abs(u)>1e-8]; return np.median(u)
cells=[os.path.basename(p)[:-3] for p in sorted(glob.glob(f'{P}/train_real/*.h5')) if not p.endswith('7575_0.h5') and not p.endswith('_20.h5')]
re=np.array([int(c.split('_')[0]) for c in cells],float)
k=np.array([peak(f'{P}/train_real/{c}.h5')/peak(f'{P}/train_sim/{c}.h5') for c in cells])
U=np.array([ufree(f'{P}/train_real/{c}.h5') for c in cells])
slope,icpt=np.polyfit(np.log(re),np.log(k),1)
Re0_a=np.exp(np.median(np.log(re)-np.log(k)))
c=np.median(1e-6*re/U); Re0_b=c**2/1e-6
print(f"{len(cells)} cells (20 deg excluded)")
print(f"(a) frequencies: free fit k ~ Re^{slope:.2f}; with slope 1, Re0 = {Re0_a:.0f}  (IQR of per-cell Re/k {np.percentile(re/k,25):.0f}-{np.percentile(re/k,75):.0f})")
print(f"(b) freestream: U/Re median {np.median(U/re):.3e} m/s, chord = nu*Re/U = {c*100:.1f} cm at nu 1.0e-6, Re0 = {Re0_b:.0f}")
print(f"    U vs Re log slope {np.polyfit(np.log(re),np.log(U),1)[0]:.2f} (1 = tunnel speed sets Re)")
np.save('clock_fit.npy', dict(Re0_a=Re0_a, Re0_b=Re0_b, slope=slope))
