#!/usr/bin/env python3
"""Verify forward preservation, temporal derivative routing and loss endpoints."""
from pathlib import Path
import sys
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ckpt import fold_decomposed_output, fold_modality
from realistix.models import build_model
from realistix.losses import (CompositeLoss, cooperative_mean_gradient,
                             cooperative_tke_loss, tke_rel_l2)


class Routes(unittest.TestCase):
    def test_decomposed_output_packs_as_ordinary_convolution(self):
        torch.manual_seed(7)
        args = dict(arch='unet', width=8, film=True, cond='norm', decompose=True)
        source = build_model(args).eval()
        with torch.no_grad():
            for name, q in source.named_parameters():
                if name.startswith(('out.', 'out_mean.')) or name.endswith(('.f1.weight', '.f2.weight')):
                    q.normal_(0, .02)
        deployed = build_model(dict(args, film=False, cond='none', decompose=False)).eval()
        sd, count = fold_modality(fold_decomposed_output(source.state_dict(), args))
        self.assertGreater(count, 0)
        deployed.load_state_dict(sd, strict=True)
        x = torch.randn(2,40,32,64)
        with torch.no_grad():
            torch.testing.assert_close(source(x), deployed(x), rtol=2e-5, atol=3e-6)
        with self.assertRaises(SystemExit):
            fold_decomposed_output(source.state_dict(), dict(args, anchor='blend'))
        with self.assertRaises(SystemExit):
            fold_decomposed_output(source.state_dict(), dict(args, out_cumsum=True))

    def test_preserves_values_and_routes_mean(self):
        torch.manual_seed(2)
        p = torch.randn(2, 40, 8, 8, requires_grad=True)
        ensemble = (p + torch.randn_like(p) + torch.randn_like(p)) / 3
        self.assertIs(cooperative_mean_gradient(ensemble, p, 3), ensemble)
        out = cooperative_mean_gradient(ensemble, p, 3, 3)
        self.assertTrue(torch.equal(out, ensemble))
        direction = torch.randn_like(p)
        got = torch.autograd.grad((out * direction).sum(), p)[0]
        mean = direction.reshape(2,20,2,8,8).mean(1,keepdim=True).expand(2,20,2,8,8).reshape_as(p)
        torch.testing.assert_close(got, direction / 3 + 2 * mean / 3)

    def test_tke_derivative_unchanged(self):
        torch.manual_seed(3)
        p = torch.randn(2, 40, 8, 8, requires_grad=True)
        target = torch.randn_like(p)
        ensemble = (p + torch.randn_like(p) + torch.randn_like(p)) / 3
        old = torch.autograd.grad(tke_rel_l2(ensemble, target), p, retain_graph=True)[0]
        routed = cooperative_mean_gradient(ensemble, p, 3, 3)
        new = torch.autograd.grad(tke_rel_l2(routed, target), p)[0]
        torch.testing.assert_close(new, old, atol=1e-8, rtol=1e-5)

    def test_loss_endpoints(self):
        torch.manual_seed(4)
        p = torch.randn(2,40,32,64,requires_grad=True)
        target = torch.randn_like(p); ensemble=(p+torch.randn_like(p))/2
        field = CompositeLoss(mse=1,rel=8,tke=0,spec=.1,mvpe=.3,tmean=.5)
        full = CompositeLoss(mse=1,rel=8,tke=1,spec=.1,mvpe=.3,tmean=.5)
        expected=full(p,target)[0]
        actual=field(p,target)[0]+cooperative_tke_loss(ensemble,p,target,1,3)
        torch.testing.assert_close(actual,expected)
        torch.testing.assert_close(cooperative_tke_loss(ensemble,p,target,0,3),
                                   3*tke_rel_l2(ensemble,target))
        actual.backward();self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
