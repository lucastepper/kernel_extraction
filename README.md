# kernel_extraction

`kernel_extraction` estimates the memory kernel of a generalized Langevin equation
from time correlation functions. The memory kernel describes how past motion
contributes to friction. The package solves discretized Volterra equations on
a uniform time grid and returns either the memory kernel `Γ(t)` or its running
integral `G(t)`, depending on the method.

A derivation is in the [derivations folder](derivations/) of the repository.

## Installation

Install `kernel_extraction` from [PyPI](https://pypi.org/project/kernel-extraction/).

```bash
pip install kernel-extraction
```

To build from source, with a Rust toolchain and maturin installed, run this from the repository root.

```bash
maturin build --release
pip install target/wheels/correlation-*.whl
```

The build embeds `src/kernel_steps.ptx` and requires no CUDA toolkit.

## Extract a kernel from correlations

This example uses the sample correlations in the repository. `vv_corr` is the
velocity autocorrelation, and `xdu_corr` correlates position with the potential
gradient. Use one-dimensional NumPy arrays of dtype `float64` sampled at the
same lag spacing. Set `dt` to that spacing and `trunc` to the requested number
of output samples.

```python
import numpy as np
from kernel_extraction import ExtractionMethod, compute_kernel

# Run from the repository root.
vv_corr = np.load("test/data/vv_corr.npy").astype(np.float64)
xdu_corr = np.load("test/data/xdu_corr.npy").astype(np.float64)
dt = 0.01
mass = 2.494 / vv_corr[0]  # kBT / Cvv(0) for these sample data

kernel_int = compute_kernel(
    ExtractionMethod.G_METHOD_XDU,
    dt=dt,
    mass=mass,
    vv_corr=vv_corr,
    xdu_corr=xdu_corr,
    trunc=200,
    device="auto",
)
```

| Method | Required correlations | Returns |
| --- | --- | --- |
| `G_METHOD_XDU` | `vv_corr`, `xdu_corr` | Running integral |
| `G_METHOD_XDU_NOMASS` | `vv_corr`, `xdu_corr` | Running integral |
| `G_METHOD_XDU_HALFSTEP` | `vv_corr`, `xdu_corr` | Running integral on the half-step grid |
| `G_METHOD_VDU` | `vv_corr`, `vdu_corr` | Running integral |
| `GAMMA_METHOD` | `vv_corr`, `va_corr`, `aa_corr`, `adu_corr` | Memory kernel |

Pass the mass to every method except `G_METHOD_XDU_NOMASS`. To extract a
position-dependent kernel from conditional correlations, use
`compute_kernel_2d` with arrays shaped `(n_lags, n_bins)` and a mass per bin
or thermal energy `kbt`. If you are starting from trajectories,
`kernel_from_traj` computes the correlations and returns them with the kernel.
That workflow also requires the `correlation` and `md_utils` Python modules.

## Choosing a device

The same wheel supports both CPU and CUDA on hosts that match its Python ABI
and platform tag. Pass `device` to `compute_kernel`, `compute_kernel_2d`,
`kernel_from_traj`, or a low-level extraction function.

- `device="cpu"` uses the CPU and is the default in the Python wrappers.
- `device="cuda"` uses CUDA device 0 and raises `RuntimeError` if CUDA setup fails.
- `device="auto"` tries CUDA device 0 and uses the CPU if CUDA setup fails.

CUDA requires a visible NVIDIA GPU, a driver that supports the embedded
CUDA 12.6 PTX, and a CUDA 12 cuBLAS shared library on the system library
search path. The CPU backend requires no NVIDIA libraries.
