#!/usr/bin/env python3
"""Reject direct/ancestor calibration leakage and verify deterministic selection."""
from pathlib import Path
import sys
import tempfile
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from realistix.calibration import calibration_mask, validate_oof_members


class Calibration(unittest.TestCase):
    def test_explicit_member_precision_overrides_autocast(self):
        from train_final_head import Ensemble
        class Conv(torch.nn.Conv2d):
            n_modalities=0
        ens=Ensemble.__new__(Ensemble);torch.nn.Module.__init__(ens)
        ens.models=torch.nn.ModuleList([Conv(2,2,1)])
        x=torch.randn(1,2,4,4)
        ens.member_precision='auto'
        with torch.autocast('cpu',dtype=torch.bfloat16):auto=ens.members(x)
        ens.member_precision='float32'
        with torch.autocast('cpu',dtype=torch.bfloat16):full=ens.members(x)
        self.assertEqual(auto.dtype,torch.bfloat16)
        self.assertEqual(full.dtype,torch.float32)
        self.assertTrue(torch.equal(full,ens.members(x)))

    def test_nested_exclusions_and_ancestors(self):
        with tempfile.TemporaryDirectory() as directory:
            paths=[Path(directory)/f'm{i}.pt' for i in range(2)]
            args=dict(real_trajs='train',REALISTIX_HELDOUT_CELLS='1_0,2_0',
                      REALISTIX_HELDOUT_AOA='',REALISTIX_HELDOUT_RE='')
            def save(a):
                for p in paths:torch.save(dict(args=a,model={'w':torch.zeros(1)}),p)
            save(args)
            self.assertEqual(len(validate_oof_members(paths,['2_0'],['2_0','3_0'],['1_0','2_0','3_0'])),2)
            for bad in [dict(args,REALISTIX_HELDOUT_CELLS='1_0'),
                        dict(args,REALISTIX_HELDOUT_CELLS='2_0'),
                        dict(args,init_from='seen_outer.pt'),dict(args,real_trajs='all')]:
                save(bad)
                with self.assertRaises(ValueError):
                    validate_oof_members(paths,['2_0'],['2_0','3_0'],['1_0','2_0','3_0'])

    def test_selection_and_rng_preservation(self):
        regimes=torch.tensor([[1.,0.],[2.,5.],[2.,0.],[3.,0.]])
        cells=torch.tensor([[2.,0.],[2.,5.]])
        torch.manual_seed(13); state=torch.random.get_rng_state()
        with torch.random.fork_rng(devices=[]):torch.nn.Linear(7,3)
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
        self.assertEqual(calibration_mask(regimes,cells).tolist(),[False,True,True,False])


if __name__=='__main__':unittest.main()
