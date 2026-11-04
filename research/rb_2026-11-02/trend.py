"""Does the CFD get the CHANGE of the flow between regimes right? For neighbouring cells (same Re, angle a -> a+D; or same
angle, Re step n), compare the real change r(b)-r(a) with the sim change s(b)-s(a), each statistic normalised per
trajectory as in gap.py. Reports: correlation of the two change maps, and relative error |ds - dr| / |dr| (< 1 means the
sim's trend carries information beyond copying the neighbour; the additive-calibration estimate is exactly this)."""
import numpy as np
z=np.load('research/rb_2026-11-02/moments.npz',allow_pickle=True); mom=z['mom'].item()
real={c:t for (k,c),t in mom.items() if k=='real'}; sim={c:t for (k,c),t in mom.items() if k=='sim'}
def norm(t,M):
    s=np.sqrt(t['s'][:,M].sum(0).mean()); m=t['m'][:,M]/s
    return np.stack([m, np.sqrt(np.clip(t['s'][:,M]/s**2-m**2,0,None))])
REs=sorted({int(c.split('_')[0]) for c in sim})
def pairs(axis,D):
    for c in real:
        re_,a=map(int,c.split('_'))
        if axis=='AoA': o=f'{re_}_{a+D}'
        else:
            i=REs.index(re_); o=f'{REs[i+D]}_{a}' if i+D<len(REs) else None
        if o and o in real: yield c,o
print(f"{'change':>14} {'n':>3} | mean flow: corr, rel err | fluct RMS: corr, rel err   (median over pairs)")
for axis,Ds in (('AoA',(5,10,15)),('Re',(1,2,4))):
    for D in Ds:
        out=[]
        for a,b in pairs(axis,D):
            M=real[a]['ok']&real[b]['ok']&sim[a]['ok']&sim[b]['ok']
            dr=norm(real[b],M)-norm(real[a],M); ds=norm(sim[b],M)-norm(sim[a],M)
            row=[]
            for k in (0,1):
                x,y=dr[k].ravel(),ds[k].ravel()
                row+= [np.corrcoef(x,y)[0,1], np.linalg.norm(y-x)/np.linalg.norm(x)]
            out.append(row)
        md=np.median(out,0); lab=f"{axis} +{D}{'deg' if axis=='AoA' else ' steps'}"
        print(f"{lab:>14} {len(out):>3} |   {md[0]:+.2f}, {md[1]:.2f}          |   {md[2]:+.2f}, {md[3]:.2f}")
