import sys
from typing import Optional
import numpy as np
from enum import Enum
from .kernel_extraction import (
    g_method_vdu,
    g_method_xdu,
    g_method_xdu_nomass,
    g_method_xdu_halfstep,
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
