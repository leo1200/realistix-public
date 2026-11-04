"""Provenance guards and prediction substitution for nested head calibration."""
import hashlib
from pathlib import Path
import torch


def validate_oof_members(paths, calibration_cells, head_train_cells, available_cells):
    """Reject direct or ancestor leakage into calibration/outer evaluation cells.

    Only ordinary members with explicit whole-trajectory split metadata are
    supported. Subsampling can remove additional training cells but cannot make
    an excluded trajectory safe if an initialization or teacher saw it.
    """
    calibration_cells = set(calibration_cells)
    if not calibration_cells or not calibration_cells <= set(head_train_cells):
        raise ValueError('OOF calibration cells must be a nonempty subset of head training cells')
    forbidden = calibration_cells | (set(available_cells) - set(head_train_cells))
    if len(paths) < 2 or len(set(map(str, paths))) != len(paths):
        raise ValueError('OOF spread needs at least two distinct members')
    hashes = {}
    for path in paths:
        ck = torch.load(str(path), map_location='cpu', mmap=True)
        a = ck['args']
        if a.get('real_trajs', 'train') != 'train':
            raise ValueError(f'{path}: OOF member must use the train split')
        if any(a.get(k) for k in ('init_from', 'distill_from', 'cooperate_with', 'distill_unsafe')):
            raise ValueError(f'{path}: OOF member ancestry is unsupported')
        keys = ('REALISTIX_HELDOUT_CELLS', 'REALISTIX_HELDOUT_RE', 'REALISTIX_HELDOUT_AOA')
        if any(k not in a or a[k] is None for k in keys):
            raise ValueError(f'{path}: explicit OOF split metadata required')
        cells = set(filter(None, a[keys[0]].split(',')))
        re_values = {int(v) for v in a[keys[1]].split(',') if v}
        aoa_values = {int(v) for v in a[keys[2]].split(',') if v}
        seen = {c for c in available_cells if c not in cells
                and int(c.split('_')[0]) not in re_values
                and int(c.split('_')[1]) not in aoa_values}
        leak = forbidden & seen
        if leak:
            raise ValueError(f'{path}: OOF/outer leakage: {sorted(leak)}')
        h = hashlib.sha256()
        with open(path, 'rb') as source:
            for chunk in iter(lambda: source.read(1 << 20), b''): h.update(chunk)
        hashes[str(Path(path).resolve())] = h.hexdigest()
    return hashes


def calibration_mask(regime, cells):
    """Whole-cell selection, independent of inputs, errors or random state."""
    return (regime[:, None, :] == cells[None, :, :]).all(-1).any(-1)
