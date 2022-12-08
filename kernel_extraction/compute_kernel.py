import sys
import tqdm
from typing import Iterable, Optional
import numpy as np
from enum import Enum
from .kernel_extraction import (
    g_method_vdu,
    g_method_xdu,
    g_method_xdu_nomass,
    g_method_xdu_halfstep,
    gamma_method,
)


def check_mass(mass: Optional[float]) -> float:
    if mass is None:
        raise ValueError("Method requires mass but it is None")
    return mass


def get_corr(corrs: dict[str, np.ndarray], key: str, idx_bin: int) -> np.ndarray:
    if key not in corrs:
        raise ValueError(f"Method requires correlation: {key}, not in corrs")
    if idx_bin is None:
        return corrs[key]
    return corrs[key][:, idx_bin]


def get_nbins(corrs: dict[str, np.ndarray], warn_if_more: int = 1000) -> int:
    """Assume that the first dimension is the number of times and the second
    is the number of bins. Get any correlation and return the number of bins.
    Assumes 'corr' in any key for a correlation.
    """
    for key, item in corrs.items():
        if "corr" in key and item.ndim == 2:
            if item.shape[1] > warn_if_more:
                print(f"WARNING: Number of bins is {item.shape[1]}")
            return item.shape[1]


def get_trunc(trunc, corrs):
    """ Choose trunc to be the length given or the length of the shortest
    correlation. Does not check which correlations are actually used. """
    for key, item in corrs.items():
        if "corr" in key:
            trunc = min(trunc, len(item))
    return trunc

class ExtractionMethod(Enum):
    """
    Enum for the different kernel extraction methods that this module provides.
    Methods named after wether they extract kernel or running integral and the
    correlation functions they use
    Variants:
        g_method_xdu
        g_method_xdu_nomass
        g_method_xdu_halfstep
        g_method_vdu
    """

    G_METHOD_XDU = "g_method_xdu"
    G_METHOD_XDU_NOMASS = "g_method_xdu_nomass"
    G_METHOD_XDU_HALFSTEP = "g_method_xdu_halfstep"
    G_METHOD_VDU = "g_method_vdu"
    GAMMA_METHOD = "gamma_method"

    def __str__(self):
        return self.value


def compute_kernel(
    extraction_method: ExtractionMethod,
    dt: float,
    mass: float = None,
    trunc: int = sys.maxsize,
    idx_bin: int = None,
    device: str = "cpu",
    **kwargs,
) -> np.ndarray:
    corrs = {key: kwargs[key] for key in kwargs if "corr" in key}
    if extraction_method == ExtractionMethod.G_METHOD_XDU:
        return g_method_xdu(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "xdu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
        )
    if extraction_method == ExtractionMethod.G_METHOD_XDU_NOMASS:
        return g_method_xdu_nomass(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "xdu_corr", idx_bin),
            dt,
            trunc,
            device,
        )
    if extraction_method == ExtractionMethod.G_METHOD_XDU_HALFSTEP:
        return g_method_xdu_halfstep(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "xdu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
        )
    if extraction_method == ExtractionMethod.G_METHOD_VDU:
        return g_method_vdu(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "vdu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
        )
    if extraction_method == ExtractionMethod.GAMMA_METHOD:
        return gamma_method(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "va_corr", idx_bin),
            get_corr(corrs, "aa_corr", idx_bin),
            get_corr(corrs, "adu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
        )


def compute_kernel_2d(
    extraction_method: ExtractionMethod,
    dt: float,
    mass: Iterable = None,
    kbt: float = None,
    trunc: int = sys.maxsize,
    idx_bin: int = None,
    device: str = "cpu",
    verbose=True,
    **kwargs,
) -> np.ndarray:
    if mass is None:
        if kbt is None:
            raise ValueError("Method requires mass or kbt but both are None")
        mass = kbt / get_corr(kwargs, "vv_corr", None)[0]
    else:
        if not isinstance(mass, Iterable):
            raise ValueError("Mass must be iterable")
    corrs = {}
    for key, item in kwargs.items():
        if not "corr" in key:
            continue
        if "du" in key:
            # for 2d kernel, we divide potential and kernel by mass, so the volterra
            # equation can be derived. Rescale kernel after extraction
            item = item / mass.reshape(1, -1)
        corrs[key] = item
    trunc = get_trunc(trunc, corrs)
    nbins = get_nbins(corrs)
    kernel = np.zeros((trunc, nbins))
    bar = tqdm.tqdm(desc=f"{extraction_method}", disable=not verbose, total=len(mass), leave=False)
    for idx_bin, m in enumerate(mass):
        # Since we rescale kernel and potential by mass, we need to set mass to one
        # during extraction and then rescale after the extraction
        kernel[:, idx_bin] = m * compute_kernel(
            extraction_method, dt, mass=1.0, trunc=trunc, idx_bin=idx_bin, device=device, **corrs
        )
        bar.update(1)
    return kernel
