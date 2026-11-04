"""RealPDE 2026, Track 1 (Sim2Real): training package of team iapetos1918, reproducing Codabench submission 947208.

    train(data_dir, output_dir, device)   trains the method from scratch on the release in data_dir
    load_predictor(output_dir, device)    returns an object whose predict(input_array) answers like the Codabench file

train() runs tools/train_final.sh -- the documented one-command rebuild of the model behind 947208 (four U-Net
forecasters, six interval-head teachers, one distilled student head, then packing and our own integrity checks; about
2 h 15 min on one A100-PCIE-40GB) -- with every path it writes under output_dir:

    output_dir/work/      a copy of this package's code; the script runs there, so training writes nothing under /workspace
    output_dir/runs/      every training run (members, teachers, student: checkpoints, args.json, history.json)
    output_dir/build/     the packed Codabench-style submission directory and its zip
    output_dir/eval/      the parity-check windows, cut from the release's own trajectories
    output_dir/train.log  the complete log;  output_dir/TRAIN_STATUS.json  a summary

and then copies the packed inference directory to the root of output_dir: submission.py (byte-identical to the file
of submission 947208), model.pth, and the two fused-kernel cubins fastgn_<sha12>_sm80/sm90.cubin compiled from the CUDA
source embedded in submission.py. load_predictor() imports that submission.py, which loads model.pth and captures its
CUDA graphs once, at import; predict() then calls the module's own predict.

Nothing here selects among runs or checkpoints: seeds and step counts are fixed in tools/train_final.sh.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
#: The inference source of Codabench submission 947208 (final_score 81.99600), shipped in this package.
CHAMPION = "dec12kfinal5b"
INFER_SHA256 = "1af16254833af11d3a7dfa02857801bf5b46aca380c3c7b726d735a8fcec1612"
NAME = SUB = "final"
#: What train() leaves at the root of output_dir (and moves aside if a previous run left it there).
_OURS = ("work", "runs", "build", "eval", "submission.py", "model.pth", "train.log", "TRAIN_STATUS.json")


def _log(msg, log=None):
    line = f"[solution {datetime.datetime.now(datetime.timezone.utc):%H:%M:%SZ}] {msg}\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    if log is not None:
        log.write(line)
        log.flush()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, cwd, env, log=None):
    """Run cmd, streaming its output to stdout (and to the open file `log`); return the exit code."""
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        sys.stdout.write(line)
        if log is not None:
            log.write(line)
            log.flush()
    sys.stdout.flush()
    return p.wait()


def _gpu_index(device):
    d = str(device or "cuda")
    return d.split(":", 1)[1] if d.startswith("cuda:") else None


def train(data_dir, output_dir, device="cuda"):
    out = os.path.abspath(output_dir)
    os.makedirs(out, exist_ok=True)
    status = {"package": "iapetos1918 Track 1, reproduces Codabench submission 947208", "inference": CHAMPION,
              "data_dir": os.path.abspath(data_dir), "output_dir": out,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        _train(os.path.abspath(data_dir), out, device, status)
    except BaseException as e:
        status.setdefault("error", repr(e)[:500])
        raise
    finally:
        with open(os.path.join(out, "TRAIN_STATUS.json"), "w") as f:
            json.dump(status, f, indent=1)


def _train(data, out, device, status):
    t0 = time.time()

    # A previous attempt's artefacts are moved aside, never mixed into this run (the script refuses to reuse a run
    # directory, and an old member must not enter a new ensemble).
    old = [n for n in _OURS if os.path.lexists(os.path.join(out, n))] + \
          [os.path.basename(p) for p in glob.glob(os.path.join(glob.escape(out), "fastgn_*.cubin"))]
    if old:
        prev = os.path.join(out, "_previous_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        os.makedirs(prev)
        for n in old:
            shutil.move(os.path.join(out, n), os.path.join(prev, n))
        _log(f"moved a previous run's {len(old)} entries to {prev}")

    # The script writes inside its own tree (submissions/<SUB>/), so it runs from a copy under output_dir.
    work = os.path.join(out, "work")
    skip = os.path.realpath(out)
    shutil.copytree(HERE, work, ignore=lambda d, names: [n for n in names if n in ("__pycache__", ".git")
                    or n.endswith(".pyc") or os.path.realpath(os.path.join(d, n)) == skip])
    src = os.path.join(work, "submissions", CHAMPION, "submission.py")
    if _sha256(src) != INFER_SHA256:
        raise RuntimeError(f"{src} is not the inference file of submission 947208")

    env = {k: v for k, v in os.environ.items() if not k.startswith(("REALISTIX_", "REALPDE_"))}
    env.update(REALPDE_DATA=data, R=os.path.join(out, "runs"), REALISTIX_BUILD=os.path.join(out, "build"),
               CHECK_PACK_EVAL=os.path.join(out, "eval", "val_aoa"), NAME=NAME, SUB=SUB, CHAMPION=CHAMPION,
               HEAD_MODE="student", PY=sys.executable, PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "4"), HDF5_USE_FILE_LOCKING="FALSE")
    if _gpu_index(device) is not None:
        env["GPU"] = _gpu_index(device)
    with open(os.path.join(out, "train.log"), "w") as log:
        # The recipe pins the release byte for byte (tools/release_manifest.json, checked inside train_final.sh and
        # again by check_recipe). If the provided data differ, say so loudly and train on them anyway: the working
        # copy's manifest is regenerated from data_dir, and TRAIN_STATUS.json records the difference.
        man = [sys.executable, "tools/data_manifest.py"]
        rc = _run(man + ["--check", "tools/release_manifest.json"], work, env, log)
        status["data_matches_release"] = rc == 0
        if rc != 0:
            _log("WARNING: the data in data_dir differ from the recorded release (listed above); "
                 "training on them with a manifest regenerated from data_dir", log)
            shutil.copy2(os.path.join(work, "tools", "release_manifest.json"),
                         os.path.join(work, "tools", "release_manifest.recorded.json"))
            if _run(man + ["--write", "tools/release_manifest.json"], work, env, log) != 0:
                raise RuntimeError("could not describe the data in data_dir (see train.log)")
        rc = _run(["bash", "tools/train_final.sh"], work, env, log)
    status["train_final_rc"] = rc

    # The packed inference directory: pack_submission stages it under build/<SUB>/ after the model is written and
    # before the parity and recipe checks, so a failed check still leaves a usable model.
    staged = next((d for d in (os.path.join(out, "build", SUB), os.path.join(work, "submissions", SUB))
                   if os.path.isfile(os.path.join(d, "model.pth"))), None)
    if staged is None:
        status["error"] = "no model.pth was produced"
        raise RuntimeError(f"training produced no model (train_final.sh exit {rc}; see {out}/train.log)")
    for f in ["submission.py", "model.pth"] + sorted(os.path.basename(p) for p in glob.glob(os.path.join(glob.escape(staged), "fastgn_*.cubin"))):
        shutil.copy2(os.path.join(staged, f), os.path.join(out, f))
    status["files"] = {os.path.basename(p): _sha256(p) for p in
                       [os.path.join(out, "submission.py"), os.path.join(out, "model.pth")] + sorted(glob.glob(os.path.join(glob.escape(out), "fastgn_*.cubin")))}
    status["inference_is_947208_file"] = status["files"]["submission.py"] == INFER_SHA256
    status["checks"] = "all passed" if rc == 0 else "model produced, but a post-pack check failed (see train.log)"
    status["wall_s"] = round(time.time() - t0, 1)
    _log(f"done in {status['wall_s'] / 3600:.2f} h: {status['checks']}; inference files at {out}")
    if not status["inference_is_947208_file"]:
        raise RuntimeError("the packed submission.py is not the file of submission 947208")


_INFER, _IMPORT_LOCK = {}, threading.Lock()
#: One lock for every Predictor: the inference module is one per process, so its CUDA-graph buffers are shared.
_PREDICT_LOCK = threading.Lock()
_CHUNK = 1024          # windows per module call; a multiple of 64, so the bucket sequence is that of one large call


def _import_inference(output_dir):
    """Import output_dir/submission.py once per process (setup and CUDA graph capture happen at its import)."""
    out = os.path.realpath(output_dir)
    path = os.path.join(out, "submission.py")
    for f in (path, os.path.join(out, "model.pth")):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"{f} is missing (train() writes it)")
    with _IMPORT_LOCK:
        mod = _INFER.get(path)
        if mod is not None:
            return mod
        name = "rx_submission_" + hashlib.sha256(path.encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        dwb, sys.dont_write_bytecode = sys.dont_write_bytecode, True
        try:
            spec.loader.exec_module(mod)
            mod._load()                                    # idempotent; raises if the checkpoint did not load
        except BaseException:
            sys.modules.pop(name, None)
            raise
        finally:
            sys.dont_write_bytecode = dwb
        sha = _sha256(path)
        sys.stderr.write(f"[solution] loaded {path} sha256={sha[:12]}{'' if sha == INFER_SHA256 else ' (NOT the 947208 file)'} "
                         f"device={mod._state.get('device')} graphs={sorted(mod._GRAPHS)} "
                         f"fused={mod._state.get('fused')}/{mod._FGN.get('origin')}\n")
        _INFER[path] = mod
        return mod


class Predictor:
    """predict(input_array): float32 (N, 20, 32, 64, 3) in physical units, any N -> dict of float32 arrays
    'prediction', 'lower', 'upper' of the same shape (pressure channel 0, lower <= prediction <= upper)."""

    def __init__(self, module):
        self._m = module
        self._lock = _PREDICT_LOCK                        # the module's graph buffers are shared between calls

    def predict(self, input_array, metadata=None):
        x = input_array
        if not isinstance(x, np.ndarray) and type(x).__module__.split(".")[0] == "torch":
            x = x.detach().cpu().numpy()
        x = np.ascontiguousarray(x, dtype=np.float32)
        with self._lock:
            n = x.shape[0] if x.ndim else 0
            if x.ndim != 5 or n <= _CHUNK:
                return self._m.predict(x)
            res = None
            for i in range(0, n, _CHUNK):                 # bounded host memory for very large calls
                r = self._m.predict(x[i:i + _CHUNK])
                if res is None:
                    res = {k: np.empty((n,) + v.shape[1:], np.float32) for k, v in r.items()}
                for k, v in r.items():
                    res[k][i:i + v.shape[0]] = v
            return res

    __call__ = predict


def load_predictor(output_dir, device="cuda"):
    gpu = _gpu_index(device)
    if gpu is not None:
        import torch
        torch.cuda.set_device(int(gpu))
    return Predictor(_import_inference(output_dir))
