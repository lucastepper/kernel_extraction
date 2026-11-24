import sys
from enum import Enum
from typing import Iterable, Optional
import tqdm
import numpy as np
from .kernel_extraction import (
    g_method_vdu,
    g_method_xdu,
    g_method_xdu_nomass,
    g_method_xdu_halfstep,
    gamma_method,
)


def check_mass(mass: Optional[float]) -> float:
    """Check that mass is not None and return it."""
    if mass is None:
        raise ValueError("Method requires mass but it is None")
    return mass


def get_corr(corrs: dict[str, np.ndarray], key: str, idx_bin: int) -> np.ndarray:
    """Get correlation from dict. If idx_bin is not None, extract that bin"""
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
    return None


def get_trunc(trunc, corrs):
    """Choose trunc to be the length given or the length of the shortest
    correlation. Does not check which correlations are actually used."""
    for key, item in corrs.items():
        if "corr" in key:
            trunc = min(trunc, len(item))
            return trunc
    raise ValueError("No correlation found in corrs, when looking for trunc")


def get_dueff(trajs, kbt, dt, intervals=None, cond_mass=None, fe=None, vels=None):
    """ Compute the effective potenital given by:
        U_eff(x, v) = U_pmf(x) + m(x)/2 v^2 + kbt log(sqrt(m(x))).
    """
    import md_utils

    trajs = md_utils.to_list_ndarrays(trajs)
    if vels is not None:
        vels = md_utils.to_list_ndarrays(vels)
    if cond_mass is None:
        assert intervals is not None
        cond_mass = md_utils.get_conditional_mass(trajs, dt, kbt, intervals)
    values = 0.5 * (intervals[1:] + intervals[:-1])
    if fe is None:
        fe = md_utils.get_fe(trajs)
        fe = md_utils.clean_fe(fe, verbose=False)
    grad_fe = np.gradient(fe[1], fe[1])
    dueffs = []
    for x in trajs:
        if vels is None:
            v = np.gradient(x, dt)
        dueffs.append(
            np.interp(x, fe[0], grad_fe)
            + 0.5 * np.interp(x, values, cond_mass) * v ** 2
            + kbt * np.interp(x, values, np.log(np.sqrt(cond_mass)))
        )
    if len(dueffs) == 1:
        dueffs = dueffs[0]
    return dueffs


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
    verbose: bool = True,
    kernel: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Compute the kernel from correlations using Volterra equations.
    Arguments:
        extraction_method: Enum for the extraction method
        dt (float): Time step
        mass (float): Mass
        trunc (int): Truncation length of the kernel, default len(corrs)
        idx_bin (int): If 2d correlations and idx_bin is not None, extract kernel for that bin
            using corr[:, idx_bin], else use corr[:], default None
        device (str): 'cpu' (default), 'cuda', or 'auto' to use CUDA if available
        verbose (bool): Show progress bar, default True
        **kwargs: Correlation functions, must contain 'vv_corr', 'xdu_corr', 'aa_corr', etc.
            The correlation functions must be of shape (ntimes) or (ntimes, nbins), when idx_bin
        kernel (np.ndarray(trunc)): Kernel (or kernel_int) to extend, computes the rest of the
            elements between len(kernel) and trunc, default None (compute from scratch)
    Returns:
        kernel (np.ndarray(trunc)): The kernel
    """
    corrs = {key: item for key, item in kwargs.items() if "corr" in key}
    if extraction_method == ExtractionMethod.G_METHOD_XDU:
        return g_method_xdu(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "xdu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
            verbose,
            kernel,
        )
    if extraction_method == ExtractionMethod.G_METHOD_XDU_NOMASS:
        return g_method_xdu_nomass(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "xdu_corr", idx_bin),
            dt,
            trunc,
            device,
            verbose,
            kernel,
        )
    if extraction_method == ExtractionMethod.G_METHOD_XDU_HALFSTEP:
        return g_method_xdu_halfstep(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "xdu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
            verbose,
            kernel,
        )
    if extraction_method == ExtractionMethod.G_METHOD_VDU:
        return g_method_vdu(
            get_corr(corrs, "vv_corr", idx_bin),
            get_corr(corrs, "vdu_corr", idx_bin),
            dt,
            check_mass(mass),
            trunc,
            device,
            verbose,
            kernel,
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
            verbose,
            kernel,
        )
    raise ValueError(f"Unknown extraction method: {extraction_method}")


def compute_kernel_2d(
    extraction_method: ExtractionMethod,
    dt: float,
    mass: Optional[Iterable] = None,
    kbt: float = None,
    trunc: int = sys.maxsize,
    device: str = "cpu",
    verbose=True,
    verbose_single_bin=False,
    kernel: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """Compute the kernel for a 2d system from conditional correlations.
    Arguments:
        extraction_method: Enum for the extraction method
        dt (float): Time step
        mass (np.ndarray(nbins)): Mass as a function of the bin index
        kbt (float): Thermal energy, Needed to compute mass from vv_corr if mass is None
        trunc (int): Truncation length of the kernel, default len(corrs)
        device (str): 'cpu' (default), 'cuda', or 'auto' to use CUDA if available
        verbose (bool): Show progress bar, default True
        verbose_single_bin (bool): Show and additional progress bar for each bin, default False
        **kwargs: Correlation functions, must contain 'vv_corr', 'xdu_corr', 'aa_corr', etc.
            The correlation functions must be of shape (ntimes, nbins)
        kernel (np.ndarray(trunc, nbins)): Kernel (or kernel_int) to extend, computes the rest of
            the elements between len(kernel) and trunc, default None (compute from scratch)
    Returns:
        kernel (np.ndarray(trunc, nbins)): The kernel
    """
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
    # check if we need to extend the kernel
    if kernel is None:
        kernel = np.zeros((trunc, nbins))
        len_kernel_to_extend = 0
    elif kernel.shape[0] < trunc:
        return kernel
    else:
        len_kernel_to_extend = len(kernel)
        kernel = np.concatenate((kernel, np.zeros((trunc - len(kernel), nbins))), axis=0)
    # compute the kernel for each bin
    pbar = tqdm.tqdm(desc=f"{extraction_method}", disable=not verbose, total=len(mass), leave=False)
    for idxb, m in enumerate(mass):
        # Since we rescale kernel and potential by mass, we need to set mass to one
        # during extraction and then rescale after the extraction
        kernel[:, idxb] = m * compute_kernel(
            extraction_method,
            dt,
            mass=1.0,
            trunc=trunc,
            idx_bin=idxb,
            device=device,
            verbose=verbose_single_bin,
            **corrs,
            kernel=kernel[:len_kernel_to_extend, idxb],
        )
        pbar.update(1)
    return kernel
