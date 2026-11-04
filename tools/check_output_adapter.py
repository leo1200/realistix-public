#!/usr/bin/env python3
"""Check the mean-preserving constraint and its plain-UNet deployment fold."""
from pathlib import Path
import sys
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from realistix.models import build_model
from realistix.output_adapter import CenteredOutput,install,folded_state,verify_mean_constraint,verify_constraint

class AdapterTests(unittest.TestCase):
    def test_mean_projection_preserves_fluctuations(self):
        torch.manual_seed(17)
        args=dict(width=8)
        plain=build_model(args).eval()
        with torch.no_grad():plain.out.weight.normal_(0,.02)
        base={k:v.clone() for k,v in plain.state_dict().items()}
        model=install(plain,args,'mean')
        with torch.no_grad():model.out.delta_weight.normal_(0,.02);model.out.delta_bias.normal_(0,.02)
        state=folded_state(model);verify_constraint(base,state,'mean')
        original=build_model(args).eval();original.load_state_dict(base)
        packed=build_model(args).eval();packed.load_state_dict(state)
        x=torch.randn(2,40,32,64)
        p=model(x).reshape(2,20,2,32,64);b=original(x).reshape_as(p)
        # Subtracting the O(1) persistence anchor in float32 can differ by several
        # ulps after adding a constant field. This tests numerical preservation.
        torch.testing.assert_close(p-p.mean(1,keepdim=True),b-b.mean(1,keepdim=True),atol=1e-6,rtol=2e-5)
        torch.testing.assert_close(p.reshape(2,40,32,64),packed(x),atol=3e-6,rtol=2e-5)
        (p-p.mean(1,keepdim=True)).square().mean().backward()
        self.assertLess(model.out.delta_weight.grad.abs().max().item(),1e-8)
        state['out.weight'][0].add_(.01)
        with self.assertRaises(ValueError):verify_constraint(base,state,'mean')

    def test_mean_and_mean_gradient(self):
        torch.manual_seed(9)
        layer=CenteredOutput(torch.nn.Conv2d(8,40,3,padding=1))
        with torch.no_grad():layer.delta_weight.normal_(0,.1);layer.delta_bias.normal_(0,.1)
        x=torch.randn(2,8,8,8)
        p=layer(x);base=layer.base(x)
        torch.testing.assert_close(p.reshape(2,20,2,8,8).mean(1),
                                   base.reshape(2,20,2,8,8).mean(1),atol=2e-7,rtol=2e-6)
        target=torch.randn(2,2,8,8)
        (p.reshape(2,20,2,8,8).mean(1)-target).square().mean().backward()
        self.assertLess(layer.delta_weight.grad.abs().max().item(),1e-8)
        self.assertLess(layer.delta_bias.grad.abs().max().item(),1e-8)

    def test_fold_and_frozen_backbone(self):
        torch.manual_seed(11)
        args=dict(width=8,film=True,cond='norm')
        model=install(build_model(args),args).eval()
        self.assertEqual([n for n,p in model.named_parameters() if p.requires_grad],
                         ['out.delta_weight','out.delta_bias'])
        with torch.no_grad():model.out.delta_weight.normal_(0,.02);model.out.delta_bias.normal_(0,.02)
        packed=build_model(args).eval();packed.load_state_dict(folded_state(model),strict=True)
        x=torch.randn(2,40,32,64)
        with torch.no_grad():torch.testing.assert_close(model(x),packed(x),atol=3e-6,rtol=2e-5)

    def test_checkpoint_constraint_guard(self):
        torch.manual_seed(15)
        args=dict(width=8)
        plain=build_model(args)
        base={k:v.clone() for k,v in plain.state_dict().items()}
        model=install(plain,args)
        with torch.no_grad():model.out.delta_weight.normal_(0,.02)
        state=folded_state(model);verify_mean_constraint(base,state)
        state['stem.weight'].add_(.01)
        with self.assertRaises(ValueError):verify_mean_constraint(base,state)
        state=folded_state(model);state['out.bias'].add_(.01)
        with self.assertRaises(ValueError):verify_mean_constraint(base,state)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
