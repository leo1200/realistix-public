import torch, sys
sys.path.insert(0,'src')
from realistix.gpudata import GPUWindows
from realistix.splits import Traj, external, available
torch.manual_seed(0)
def stats(name, tr, split, center):
    g=GPUWindows(tr,split,'cpu',center=center); x,y,s,_,_=g.sample(256,torch.Generator().manual_seed(0))
    x=x.reshape(256,20,2,32,64); y=y.reshape(256,20,2,32,64)
    d=(y-x[:,-1:]).pow(2).mean().sqrt()   # target departure from the anchor frame, the residual the model predicts
    print(f"{name:22s} center={center!s:5}: |mean|={x.mean(dim=(1,3,4)).abs().mean():.3f} f2f std={x.std(dim=1).mean():.3f} residual rms={d:.3f} scale={s.mean():.4f}")
import random
rr=available('real'); ss=available('sim'); random.seed(0)
stats('ours real',random.sample(rr,12),'real',False)
stats('ours sim',random.sample(ss,12),'sim',False)
B='data/realpdebench/conv'
for k in ('real','sim'):
    tr=random.sample(external(f'{B}/{k}'),12)
    for c in (False,True): stats(f'RB {k}',tr,k,c)
