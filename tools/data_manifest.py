#!/usr/bin/env python3
"""Pin the release the recipe trains on: ordered trajectory list, sizes, SHA256.

`train_final.sh` asserts "81 real, 100 sim", and a count is not an identity: a
different data directory with the same counts, a re-exported file, or a research
holdout variable that reorders the list (the sampler draws by index, so order is part
of the recipe) all pass a count. This records what `realistix.splits.available()` returns -- the
release as enumerated, in enumeration order -- with each file's size and digest, and
checks a data directory against that record. (The trainers' *training* order is the
concatenation of `partition()` buckets, which the holdout variables reorder; the build
script unsets those before any import, and `available()` is unaffected by them.)

    python tools/data_manifest.py --write tools/release_manifest.json   # once, on the release
    python tools/data_manifest.py --check tools/release_manifest.json   # in the build and in check_recipe

The check reads REALPDE_DATA (or the default data root) through `realistix.splits`
exactly as training does, with whatever environment the caller has -- so run it
with the research variables cleared, as train_final.sh does. Hashing the 2.6 GB
release takes a few tens of seconds; `--no-hash` compares names and sizes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from realistix import splits  # noqa: E402


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def describe(with_hash: bool = True) -> dict:
    out = {"excluded": {k: sorted(v) for k, v in splits.EXCLUDED.items()}}
    for split in ("real", "sim"):
        rows = []
        for t in splits.available(split):
            p = t.path(split)
            rows.append({"stem": t.stem, "bytes": os.path.getsize(p),
                         **({"sha256": sha256(p)} if with_hash else {})})
        out[split] = rows
    return out


def check(manifest: dict, with_hash: bool = True) -> list[str]:
    now = describe(with_hash)
    bad = []
    for split in ("real", "sim"):
        want, got = manifest[split], now[split]
        if [r["stem"] for r in want] != [r["stem"] for r in got]:
            ws, gs = {r["stem"] for r in want}, {r["stem"] for r in got}
            if ws == gs:
                bad.append(f"{split}: same {len(gs)} trajectories in a DIFFERENT ORDER "
                           f"(a research holdout variable is probably set)")
            else:
                bad.append(f"{split}: missing {sorted(ws - gs)[:5]}, extra {sorted(gs - ws)[:5]} "
                           f"(recorded {len(ws)}, found {len(gs)})")
            continue
        for w, g in zip(want, got):
            if w["bytes"] != g["bytes"]:
                bad.append(f"{split}/{w['stem']}: size {g['bytes']} != recorded {w['bytes']}")
            elif with_hash and "sha256" in w and w["sha256"] != g["sha256"]:
                bad.append(f"{split}/{w['stem']}: sha256 differs from the recorded release file")
    if manifest.get("excluded") != now["excluded"]:
        bad.append(f"excluded set {now['excluded']} != recorded {manifest.get('excluded')}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", metavar="JSON")
    g.add_argument("--check", metavar="JSON")
    ap.add_argument("--no-hash", action="store_true", help="names and sizes only")
    a = ap.parse_args()
    if a.write:
        m = describe(not a.no_hash)
        m["data_root"] = splits.DATA_ROOT
        json.dump(m, open(a.write, "w"), indent=1)
        print(f"wrote {a.write}: {len(m['real'])} real, {len(m['sim'])} sim from {splits.DATA_ROOT}")
        return
    bad = check(json.load(open(a.check)), not a.no_hash)
    for b in bad:
        print("  " + b)
    print(("data manifest OK -- the trainers see the recorded release, in the recorded order"
           if not bad else f"data manifest FAILED -- {len(bad)} problem(s)")
          + f" ({splits.DATA_ROOT})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
