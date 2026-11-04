"""Model zoo, plus the one place that maps a checkpoint's saved args to a module.

Every training script writes ``{"model": state_dict, "args": vars(args)}``, and
three consumers have to rebuild the module from that: the interval-head trainer,
the post-hoc scanner and the checkpoint packer. Before there was more than one
architecture each of them open-coded ``UNet(width=a["width"], ...)``, which stops
working the moment a checkpoint is not a U-Net. Build it here instead.
"""

from __future__ import annotations

from .unet import UNet, count_params

__all__ = ["UNet", "build_model", "count_params"]


def build_model(args: dict):
    """Rebuild the model a run trained, from that run's saved ``args``."""
    n_mod = 2 if args.get("film") else 0
    frame_mode = args.get("foil_frame", "none")
    if args.get("arch", "unet") == "fno":
        from .fno import FNO
        if frame_mode != "none":
            raise ValueError("foil_frame requires arch=unet")
        return FNO(width=args["width"], layers=args.get("fno_layers", 4),
                   modes1=args.get("fno_modes1", 16),
                   modes2=args.get("fno_modes2", 17), n_modalities=n_mod)
    if frame_mode != "none" or args.get("arch", "unet") != "unet":
        # The foil-frame variant remains removed after the 2026-09-11
        # consolidation; its checkpoints reload from git history.
        raise ValueError(f"model family removed: arch={args.get('arch', 'unet')!r}, foil_frame={frame_mode!r}")
    model_cls, frame_kwargs = UNet, {}
    model = model_cls(**frame_kwargs, width=args["width"],
                cond=args.get("cond", "enc"), out_gain=bool(args.get("out_gain")),
                mults=tuple(float(v) for v in str(args.get("mults", "1,2,4")).split(",")),
                blocks_per_level=args.get("blocks", 2),
                n_modalities=n_mod,
                aux_pressure=bool(args.get("aux_p", 0)),
                aux_s2r=bool(args.get("aux_s2r", 0)),
                attn=bool(args.get("attn")),
                anchor=args.get("anchor", "last"),
                warp=bool(args.get("warp")),
                decompose=bool(args.get("decompose")),
                variant=args.get("variant", "unet"),
                tf_layers=int(args.get("tf_layers", 4)),
                xscan=args.get("xscan", "") or "",
                in_stats=bool(args.get("in_stats")),
                mimo=int(args.get("mimo") or 0),
                temporal_rank=int(args.get("temporal_rank") or 0),
                out_cumsum=bool(args.get("out_cumsum")),
                stem=args.get("stem", "dense"),
                geo_in=bool(args.get("geo_in")),
                **{k: float(args[k]) for k in ("drop_path", "chan_drop")
                   if args.get(k)})
    groups = args.get("norm_groups", 8)
    if groups != 8:
        from torch import nn
        for module in model.modules():
            if isinstance(module, nn.GroupNorm):
                if groups < 1 or module.num_channels % groups:
                    raise ValueError(f"{groups} groups do not divide {module.num_channels} channels")
                module.num_groups = groups
    return model
