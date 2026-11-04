# Third-party code

## Muon optimizer (adapted)
`src/realistix/muon.py` implements the Muon update (Newton-Schulz orthogonalised Nesterov momentum, the
`max(1, rows/cols)^0.5` scale and decoupled weight decay) following Keller Jordan's reference implementation,
https://github.com/KellerJordan/Muon, which is distributed under the MIT License:

```
MIT License

Copyright (c) 2024 Keller Jordan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Official scorer
`starting_kit/scoring.py` is the organisers' scoring program from the Track 1 starting kit (v9), shipped unmodified.
The training code imports it for the global normalisation constant `SIGMA_GLOBAL` and the metric functions that the
loss and the interval-head reward mirror.

## Libraries (from the base image or pinned in the Dockerfile)
PyTorch 2.2.2 (BSD-3-Clause), NumPy 1.26.4 (BSD-3-Clause), h5py 3.16.0 (BSD-3-Clause), CUDA runtime libraries and
NVRTC 12.1 (NVIDIA license, part of the base image).
