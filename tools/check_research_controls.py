#!/usr/bin/env python3
"""Scientific parity and sampling checks for the September 9 experiment controls."""
import os
from pathlib import Path
import sys
import types
import unittest
import json
import subprocess
import tempfile
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'starting_kit')]
from realistix.gpudata import MixedGPUWindows
from realistix.losses import sps_oracle, sps_bulk, sps_prefactors
from realistix.evaluate import to_ntchw
from realistix.models import build_model
import scoring

def trajectories(tag):
    return [types.SimpleNamespace(re=i+1,aoa=5*i,path=lambda split,i=i:f'{split}{i}') for i in range(2)]

def fake_load(path,phase=0):
    # Different lengths expose accidental window-weighted 'trajectory' sampling.
    n=49 if path.endswith('0') else 89
    return np.ones((n,2,4,4),np.float32)*(1 if path.endswith('0') else 2)

class Controls(unittest.TestCase):
    def test_sps_epsilon_floors_use_physical_units(self):
        torch.manual_seed(19)
        y=torch.full((2,40,32,64),.1)
        p=(y+torch.randn_like(y)*1e-5).requires_grad_()
        scale=torch.tensor([.001,.5])
        pp=to_ntchw(p.detach()*scale[:,None,None,None]).numpy()
        yy=to_ntchw(y*scale[:,None,None,None]).numpy()
        official,_=scoring.aggregate_sps(pp,yy,2,np.minimum(pp[...,:2],yy[...,:2]),np.maximum(pp[...,:2],yy[...,:2]))
        loss=sps_oracle(p,y,scale)
        self.assertAlmostEqual(float(1-loss),official,places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_cooperative_pack_rejects_changed_companions(self):
        from build_ckpt import check_cooperative_companions, _sha256
        with tempfile.TemporaryDirectory() as d:
            paths = [str(Path(d, f'm{i}.pt')) for i in range(3)]
            for path in paths: Path(path).write_text(path)
            args = dict(cooperate_with=','.join(paths[:2]),
                        cooperate_sha256={p: _sha256(p) for p in paths[:2]})
            check_cooperative_companions(paths[2], args, paths)
            with self.assertRaises(SystemExit):
                check_cooperative_companions(paths[2], args, paths+[paths[0]])
            Path(paths[0]).write_text('replacement weights')
            with self.assertRaises(SystemExit):
                check_cooperative_companions(paths[2], args, paths)

    def test_head_prefactors_match_official_reward(self):
        from realistix.models.uncertainty import sps_reward
        torch.manual_seed(11)
        y=torch.rand(2,40,32,64)*.1
        p=y+.01; p[1,:,:,32:]=y[1,:,:,32:]+.03
        lo=torch.full_like(p,.02); hi=lo.clone()
        pp=to_ntchw(p).numpy();yy=to_ntchw(y).numpy()
        official,_=scoring.aggregate_sps(pp,yy,2,pp[...,:2]-.02,pp[...,:2]+.02)
        relaxed=sps_reward(lo,hi,y-p,(y!=0).float(),tau_frac=1e-5,
                           sample_weights=sps_prefactors(p,y))
        self.assertAlmostEqual(float(relaxed),official,places=6)

    def test_cooperative_prediction_is_actual_mean(self):
        from train_mixed import CooperativeForecast
        class Linear(torch.nn.Module):
            def __init__(self, value):
                super().__init__(); self.weight=torch.nn.Parameter(torch.tensor(value))
            def forward(self,x,*unused): return x*self.weight
        member=Linear(1.); companions=[Linear(2.).requires_grad_(False),Linear(3.).requires_grad_(False)]
        ensemble=CooperativeForecast(member,companions)
        x=torch.ones(2,40,4,4)
        out=ensemble(x)
        self.assertTrue(torch.equal(out,2*x))
        out.mean().backward()
        self.assertAlmostEqual(float(member.weight.grad),1/3,places=6)
        self.assertTrue(all(m.weight.grad is None for m in companions))

    def test_angle_and_cell_fold_environment(self):
        for name in ('a00_re1','mim1','far1'):
            code = ("import sys,os; sys.path[:0]=['tools','src']; "
                    "import foldenv; "
                    f"foldenv.apply_fold('{name}'); "
                    "from realistix import splits; "
                    "assert splits.partition('real')['train']; "
                    "assert all(os.environ[k]==v for k,v in foldenv.fold_settings('"+name+"').items())")
            subprocess.run([sys.executable,'-c',code],cwd=ROOT,check=True)

    def test_resume_uses_last_override_and_holdout(self):
        import kfold
        from train_mixed import build_parser
        import shlex
        args='--ema .998 --w-mvpe .3'
        with tempfile.TemporaryDirectory() as d:
            rec=vars(build_parser().parse_args(shlex.split(kfold.BASE+' '+args)+['--out',d]))
            env=dict(REALISTIX_HELDOUT_AOA='',REALISTIX_HELDOUT_RE='',REALISTIX_HELDOUT_CELLS='13950_0')
            rec.update(env)
            Path(d,'args.json').write_text(json.dumps(rec))
            Path(d,'history.json').write_text(json.dumps([dict(step=8000)]))
            Path(d,'best.pt').touch()
            self.assertTrue(kfold._finished(d,args,env))
            rec['film']=False; rec['no_film']=True
            Path(d,'args.json').write_text(json.dumps(rec))
            self.assertTrue(kfold._finished(d,args+' --no-film',env))
            rec['film']=True; rec['no_film']=False
            Path(d,'args.json').write_text(json.dumps(rec))
            with self.assertRaises(SystemExit): kfold._finished(d,args,dict(env,REALISTIX_HELDOUT_CELLS='6300_0'))
            Path(d,'best.pt').unlink()
            self.assertFalse(kfold._finished(d,args,env))

    def test_oracle_matches_official_scorer(self):
        torch.manual_seed(4)
        y=torch.randn(3,40,32,64)*.05
        y[:,:,0:4,:]=0
        for exact in (False,True):
            p=(y.clone() if exact else torch.randn_like(y)*.04).requires_grad_()
            scale=torch.tensor([.3,.7,1.2])
            pp=to_ntchw(p.detach()*scale[:,None,None,None]).numpy()
            yy=to_ntchw(y*scale[:,None,None,None]).numpy()
            official,_=scoring.aggregate_sps(pp,yy,2,np.minimum(pp[...,:2],yy[...,:2]),np.maximum(pp[...,:2],yy[...,:2]))
            loss=sps_oracle(p,y,scale)
            self.assertAlmostEqual(float(1-loss),official,places=6)
            loss.backward()
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertLessEqual(float(1-loss),float(1-sps_bulk(p,y,scale))+1e-6)

    def test_importance_weights_match_replication(self):
        torch.manual_seed(7)
        p=torch.randn(2,40,32,64)*.04
        y=torch.randn_like(p)*.05
        y[0,:,:8,:]=0
        scale=torch.tensor([.3,.8])
        idx=torch.tensor([0,1,1,1])
        for loss in (sps_bulk,sps_oracle):
            weighted=loss(p,y,scale,sample_weights=torch.tensor([1.,3.]))
            replicated=loss(p[idx],y[idx],scale[idx])
            self.assertAlmostEqual(float(weighted),float(replicated),places=6)

    @patch('realistix.gpudata.load_traj',side_effect=fake_load)
    def test_mixture_counts_and_extremes(self,_):
        for fraction in (0,.25,.75,1):
            ds=MixedGPUWindows(trajectories(0),trajectories(1),'cpu',sim_batch_frac=fraction)
            x,y,sc,tag,reg=ds.sample(32)
            self.assertEqual(int(tag.sum()),round(32*fraction))
            self.assertEqual(tuple(x.shape),(32,40,4,4))
            self.assertEqual(len(ds.real.fields),2)
            self.assertEqual(len(ds.sim.fields),2)
            self.assertTrue(torch.isfinite(x).all())
        with self.assertRaises(ValueError):
            MixedGPUWindows(trajectories(0),[],'cpu',sim_batch_frac=.5)

    @patch('realistix.gpudata.load_traj',side_effect=fake_load)
    def test_sampling_unit_and_angles(self,_):
        ratios={}
        for unit in ('window','trajectory'):
            ds=MixedGPUWindows(trajectories(0),[],'cpu',sampling=unit)
            generator=torch.Generator().manual_seed(6)
            *_,reg=ds.sample(10000,generator)
            ratios[unit]=float((reg[:,0]==1).float().mean())
            *_,reg=ds.sample(10,angles=[5])
            self.assertTrue((reg[:,1]==5).all())
        self.assertAlmostEqual(ratios['window'],1/6,delta=.02)
        self.assertAlmostEqual(ratios['trajectory'],.5,delta=.02)

    def test_architecture_roundtrip_and_default(self):
        args=dict(width=16,film=True,cond='norm',mults='1,2',blocks=1)
        torch.manual_seed(1); legacy=build_model(args)
        torch.manual_seed(1); default=build_model(dict(args,norm_groups=8))
        self.assertTrue(all(torch.equal(v,default.state_dict()[k]) for k,v in legacy.state_dict().items()))
        for groups in (4,16):
            model=build_model(dict(args,norm_groups=groups))
            self.assertTrue(all(m.num_groups==groups for m in model.modules() if isinstance(m,torch.nn.GroupNorm)))
            # Group count is not in the state dict: saved args must reconstruct it.
            clone=build_model(dict(args,norm_groups=groups)); clone.load_state_dict(model.state_dict())
            x=torch.randn(2,40,8,8)
            with torch.no_grad(): self.assertTrue(torch.equal(model(x),clone(x)))
        with self.assertRaises(ValueError): build_model(dict(args,norm_groups=3))

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
