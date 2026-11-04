"""Trainable mean/fluctuation output corrections that fold into one convolution."""
import torch
from torch import nn
from torch.nn import functional as F


def centered(tensor, frames=20):
    values=tensor.reshape(frames, tensor.shape[0]//frames, *tensor.shape[1:])
    return (values-values.mean(0,keepdim=True)).reshape_as(tensor)


class ProjectedOutput(nn.Module):
    def __init__(self, base, mode='fluctuation'):
        super().__init__()
        if base.out_channels != 40 or base.groups != 1:
            raise ValueError('requires the ordinary 20-frame/two-channel output')
        self.base=base
        if mode not in ('mean','fluctuation'):raise ValueError('unknown output projection')
        self.mode=mode
        for p in self.base.parameters():p.requires_grad_(False)
        self.delta_weight=nn.Parameter(torch.zeros_like(base.weight[:2] if mode=='mean' else base.weight))
        self.delta_bias=nn.Parameter(torch.zeros_like(base.bias[:2] if mode=='mean' else base.bias))

    def forward(self, features):
        raw=F.conv2d(features,self.delta_weight,self.delta_bias,
                     self.base.stride,self.base.padding,self.base.dilation)
        if self.mode=='mean':
            correction=raw.repeat(1,20,1,1)
        else:
            shape=raw.shape
            values=raw.reshape(shape[0],20,2,*shape[-2:])
            correction=(values-values.mean(1,keepdim=True)).reshape_as(raw)
        return self.base(features)+correction


CenteredOutput=ProjectedOutput


def install(model, args, mode='fluctuation'):
    if args.get('arch','unet') != 'unet' or args.get('anchor','last') != 'last':
        raise ValueError('output adaptation requires the ordinary residual UNet')
    if any(args.get(k) for k in ('decompose','out_gain','out_cumsum','warp','mimo')):
        raise ValueError('output adaptation requires the ordinary output path')
    for p in model.parameters():p.requires_grad_(False)
    model.out=ProjectedOutput(model.out,mode)
    return model


def folded_state(model):
    if not isinstance(model.out,ProjectedOutput):raise ValueError('adapter missing')
    sd={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('out.')}
    for suffix in ('weight','bias'):
        delta=getattr(model.out,'delta_'+suffix)
        correction=(delta.repeat(20,*([1]*(delta.ndim-1))) if model.out.mode=='mean' else centered(delta))
        sd['out.'+suffix]=(getattr(model.out.base,suffix)+correction).detach().cpu()
    return sd


def verify_constraint(base, adapted, mode='fluctuation'):
    """Check a materialized checkpoint's claimed frozen-backbone constraint."""
    if set(base)!=set(adapted):raise ValueError('output adapter changed state keys')
    for key,value in base.items():
        if key in ('out.weight','out.bias'):
            shape=(20,2,*value.shape[1:])
            if mode=='fluctuation':
                before,after=value.float().reshape(shape).mean(0),adapted[key].float().reshape(shape).mean(0)
            elif mode=='mean':
                before,after=centered(value.float()),centered(adapted[key].float())
            else:raise ValueError('unknown output projection')
            if not torch.allclose(before,after,atol=1e-7,rtol=0):
                raise ValueError(f'output adapter violated {mode} constraint at {key}')
        elif not torch.equal(value,adapted[key]):
            raise ValueError(f'output adapter changed frozen parameter {key}')


def verify_mean_constraint(base, adapted):
    return verify_constraint(base,adapted,'fluctuation')
