"""Does simulation help more where its clock happens to match? Per real cell at 20 deg: (1) the sim/real clock ratio
(spectral centroid of v at nine wake probes, as freq2.py), (2) rel_l2 of each coverage-dose arm's 3-seed ensemble on
that cell (stride-40 windows, CPU). Sim value per cell = real-only minus sim-arm error (positive = sim helps)."""
import os, sys, json, h5py, numpy as np, torch
from scipy.signal import welch
T=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')  # repository root
sys.path.insert(0,T+'/tools'); sys.path.insert(0,T+'/src'); sys.path.insert(0,T+'/starting_kit')
os.environ.update(REALISTIX_HELDOUT_AOA='', REALISTIX_HELDOUT_RE='', REALISTIX_HELDOUT_CELLS='')
from torch.utils.data import DataLoader
from cell_report import load_ensemble, cell_scores
from realistix.data import WindowDataset
from realistix.splits import Traj
P='data/RealPDE-Competition-Data/training_data'
def centroid(p):
    with h5py.File(p,'r') as h: v=np.stack([h['v'][:,y,x] for y in (24,32,40) for x in (80,100,120)],1).astype(np.float64)
    v=v[:,np.abs(v).max(0)>1e-8]; v-=v.mean(0); f,S=welch(v,nperseg=min(512,len(v)),axis=0); S=S.mean(1); k=(f>0.006)&(f<0.25)
    return (S[k]*f[k]).sum()/S[k].sum()
cells=sorted([int(os.path.basename(p).split('_')[0]) for p in os.listdir(P+'/train_real') if p.endswith('_20.h5')])
ARMS=[(f'th{d}{a}', 'a20k' if a in ('rawk','clkk') else 'a20') for d in ('05','15') for a in ('real','rawx','rawk','clkx','clkk')]
ens={t:load_ensemble(t,f,[0,1,2],'cpu') for t,f in ARMS}
out=[]
for re_ in cells:
    c=f'{re_}_20'; ratio=centroid(f'{P}/train_sim/{c}.h5')/centroid(f'{P}/train_real/{c}.h5')
    dl=DataLoader(WindowDataset([Traj(re_,20)],'real',stride=40),batch_size=64)
    row=dict(re=re_, ratio=ratio, err={t:cell_scores(ens[t],dl,'cpu')['_rel_l2'] for t,_ in ARMS})
    out.append(row); print(re_, f"{ratio:.2f}", " ".join(f"{t}:{v:.4f}" for t,v in row['err'].items()), flush=True)
json.dump(out,open('clock_cells_clk.json','w'),indent=1)
