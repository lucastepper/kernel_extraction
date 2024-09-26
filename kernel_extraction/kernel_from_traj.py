import sys
from typing import Optional, Union
from dataclasses import dataclass
import numpy as np
from scipy.integrate import cumulative_trapezoid
from .compute_kernel import ExtractionMethod, compute_kernel

# from kernel_extraction import ExtractionMethod, compute_kernel

IMPORT_ERROR = False
try:
    import correlation
    import md_utils
except ImportError:
    IMPORT_ERROR = True


def check_imports():
    """Check if imports are available"""
    if IMPORT_ERROR:
        raise ImportError("Could not import correlation and md_utils. Make sure they are installed")


@dataclass
class Result:
    """Class to hold the result of the kernel extraction."""

    _kernel: Optional[np.ndarray]
    _kernel_int: Optional[np.ndarray]
    vv_corr: Optional[np.ndarray]
    xdu_corr: Optional[np.ndarray]
    vdu_corr: Optional[np.ndarray]
    va_corr: Optional[np.ndarray]
    aa_corr: Optional[np.ndarray]
    dt: float
    mean_vsqrd: Optional[float]
    mean_du: Optional[float]

    def set_kernel(self, kernel: Optional[np.ndarray], kernel_int: Optional[np.ndarray]):
        """Set the kernel"""
        self._kernel = kernel
        self._kernel_int = kernel_int

    @property
    def kernel(self) -> np.ndarray:
        """Return the kernel, either from self._kernel or np.gradeint(self._kernel_int)"""
        if self._kernel is not None:
            return self._kernel
        if self._kernel_int is not None:
            return np.gradient(self._kernel_int, self.dt)
        raise ValueError("No kernel found")

    @property
    def kernel_int(self) -> np.ndarray:
        """Return the kernel_int, either from self._kernelInt or
        cumulative_trapezoid(self._kernel_int)"""
        if self._kernel_int is not None:
            return self._kernel_int
        if self._kernel is not None:
            return cumulative_trapezoid(self._kernel, dx=self.dt, initial=0.0)
        raise ValueError("No kernel found")

    @classmethod
    def default(cls):
        """Make a default Result object, using None and 0"""
        return cls(
            _kernel=None,
            _kernel_int=None,
            vv_corr=None,
            xdu_corr=None,
            vdu_corr=None,
            va_corr=None,
            aa_corr=None,
            dt=0.0,
            mean_vsqrd=None,
            mean_du=None,
        )


def kernel_from_traj(
    trajs: Union[np.ndarray, list],
    extraction_method: ExtractionMethod,
    dt: float,
    kbt: float,
    bins: int,
    mass: Optional[float] = None,
    trunc: int = sys.maxsize,
    device: str = "cpu",
    verbose: bool = True,
) -> Result:
    """Compute the kernel from trajectories using Volterra equations,
    first computing the correlations.
    Arguments:
        extraction_method: Enum for the extraction method
        dt (float): Time step
        kbt (float): Thermal energy
        bins (int): Number of bins for the free energy
        mass (float): Mass, default None
            when mass is none, compute it from m = kbt / mean_vsqrd
        trunc (int): Truncation length of the kernel, default len(corrs)
        device (str): Device to use for computation, 'cpu' or 'cuda', default 'cpu'
        **kwargs: Correlation functions, must contain 'vv_corr', 'xdu_corr', 'aa_corr', etc.
            The correlation functions must be of shape (ntimes) or (ntimes, nbins), when idx_bin
    Returns:
        kernel (Result): Result dataclass containing the kernel and correlation functions
            and some other useful quantities
    """
    check_imports()
    if isinstance(trajs, np.ndarray):
        len_trajs = trajs.shape[-1]
    else:
        len_trajs = min(len(trj) for trj in trajs)
    trunc = max(trunc, len_trajs - 2)
    # find out which correlation functions to compute
    corr_objs = {"vv_corr": correlation.VVCorr(dt=dt, trunc=trunc)}
    fe = md_utils.get_fe(trajs, kbt=kbt, bins=bins)
    fe = md_utils.clean_fe(fe, verbose=False)
    # define gradient functions for later, when computing the same number of gradients for both
    # arguments of the correlation function, can use grad with dt step, else need 2d gradient
    get_vel_2dt = lambda x: np.gradient(x, dt)
    get_acc = lambda x: np.diff(np.diff(x) / dt) / dt
    get_acc_2dt = lambda x: np.gradient(np.gradient(x, dt), dt)
    if "g_method_xdu" in extraction_method.value:
        corr_objs["xdu_corr"] = correlation.XdUCorr(dt=dt, fe=fe, trunc=trunc)
    elif extraction_method == ExtractionMethod.G_METHOD_VDU:
        # to get the logic to set up potential Interpolation use XdUCorr and replace the conv_func1
        vdu_corr_obj = correlation.XdUCorr(dt=dt, fe=fe, trunc=trunc)
        vdu_corr_obj.conv_func_1 = get_vel_2dt
        corr_objs["vdu_corr"] = vdu_corr_obj
    elif extraction_method == ExtractionMethod.GAMMA_METHOD:
        corr_objs["va_corr"] = correlation.IterativeCorrelation(
            conv_func_1=get_vel_2dt, conv_func_2=get_acc_2dt, trunc=trunc
        )
        # aa_corr
        corr_objs["aa_corr"] = correlation.IterativeCorrelation(
            conv_func_1=get_acc, conv_func_2=get_acc, trunc=trunc
        )
        # to get the logic to set up potential Interpolation use XdUCorr and replace the conv_func1
        adu_corr_obj = correlation.XdUCorr(dt=dt, fe=fe, trunc=trunc)
        adu_corr_obj.conv_func_1 = get_acc_2dt
        corr_objs["adu_corr"] = adu_corr_obj
    else:
        raise ValueError(f"Unknown extraction method: {extraction_method}")
    # iterate over trajectory and compute correlation functions
    for traj in trajs:
        for corr_obj in corr_objs.values():
            corr_obj.add_data(traj)
    # check mass
    if mass is None:
        mass = kbt / corr_objs["vv_corr"].mean_vsqrd.get_result()
    # compute kernel
    corrs = {name: corr.get_result() for name, corr in corr_objs.items()}
    ker = compute_kernel(
        extraction_method, dt, trunc=trunc, mass=mass, device=device, verbose=verbose, **corrs
    )
    result = Result.default()
    if "g_method" in extraction_method.value:
        result.set_kernel(kernel_int=ker, kernel=None)
    else:
        result.set_kernel(kernel=ker, kernel_int=None)
    result.dt = dt
    # write corrs
    for name, corr in corrs.items():
        setattr(result, name, corr)
    # write mean_vsqrd and mean_du
    result.mean_vsqrd = corr_objs["vv_corr"].mean_vsqrd.get_result()
    for name, corr_obj in corr_objs.items():
        if "du" in name:
            setattr(result, f"mean_{name}", corr_obj.mean_du.get_result())
    return result
