from typing import Optional
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from .ref_kernels import RefKernel


class FitKernel:
    """
    Class to fit a kernel to reference data by considering the
    kernel and the running integral over the kernel.
    """

    def __init__(
        self,
        time_ker: np.ndarray,
        time_ker_int: np.ndarray,
        data_ker: np.ndarray,
        data_ker_int: np.ndarray,
        ker_to_fit: RefKernel,
        alpha: float = 1.0,
    ) -> None:
        if time_ker.shape != data_ker.shape:
            raise ValueError(f"{time_ker.shape=} must be equal to {data_ker.shape=}.")
        if time_ker_int.shape != data_ker_int.shape:
            raise ValueError(f"{time_ker_int.shape=} must be equal to {data_ker_int.shape=}.")
        self.time_ker = time_ker
        self.time_ker_int = time_ker_int
        self.data_ker = data_ker
        self.data_ker_int = data_ker_int
        self.ker_to_fit = ker_to_fit
        self.alpha = alpha
        self.result = None

    @classmethod
    def from_end_time(
        cls,
        end_time_ker: float,
        end_time_ker_int: float,
        dt: float,
        data_ker: np.ndarray,
        data_ker_int: np.ndarray,
        ker_to_fit: RefKernel,
        alpha: float = 1.0,
        n_elems_log: Optional[int] = False,
    ) -> "FitKernel":
        """
        Create a FitKernel object from the end time of the kernel and the
        running integral over the kernel.
        """
        # check if data is long enough and make time arrays
        if end_time_ker > len(data_ker) * dt:
            time_new = len(data_ker) * dt
            print(
                f"WARNING: {len(data_ker) * dt=} is less than {end_time_ker=}, changing to {time_new=}."
            )
            end_time_ker = time_new
        time_ker = np.arange(0, end_time_ker, dt)
        if end_time_ker_int > len(data_ker_int) * dt:
            time_new = len(data_ker_int) * dt
            print(
                f"WARNING: {len(data_ker_int) * dt=} is less than {end_time_ker_int=}, changing to {time_new=}."
            )
            end_time_ker_int = time_new
        time_ker_int = np.arange(0, end_time_ker_int, dt)
        # shorten input data end time given smaller end time of kernel data
        data_ker = data_ker[: len(time_ker)]
        data_ker_int = data_ker_int[: len(time_ker_int)]
        # subsample on log scale if requested
        if n_elems_log:
            idxs_ker = np.logspace(
                np.log2(1), np.log2(len(time_ker)), n_elems_log, dtype=int, base=2
            )
            idxs_ker = np.unique(idxs_ker - 1)
            time_ker, data_ker = time_ker[idxs_ker], data_ker[idxs_ker]
            idxs_ker_int = np.logspace(
                np.log2(1), np.log2(len(time_ker_int)), n_elems_log, dtype=int, base=2
            )
            idxs_ker_int = np.unique(idxs_ker_int - 1)
            time_ker_int, data_ker_int = time_ker_int[idxs_ker_int], data_ker_int[idxs_ker_int]
        # return instance
        return cls(
            time_ker,
            time_ker_int,
            data_ker,
            data_ker_int,
            ker_to_fit,
            alpha,
        )

    def get_loss(self, params, return_both=False):
        """Get the loss for the given parameters."""
        self.ker_to_fit.set_from_array(np.abs(params))
        loss_kernel = np.mean((self.ker_to_fit.get_kernel(self.time_ker) - self.data_ker) ** 2)
        loss_kernel_int = np.mean(
            (self.ker_to_fit.get_kernel_int(self.time_ker_int) - self.data_ker_int) ** 2
        )
        if return_both:
            return loss_kernel, loss_kernel_int
        return loss_kernel + self.alpha * loss_kernel_int

    def fit(self, x0, method="Nelder-Mead", maxiter=None, **kwargs):
        """Fit the kernel to the reference kernel."""
        if maxiter is not None:
            if "options" not in kwargs:
                kwargs["options"] = {}
            elif not isinstance(kwargs["options"], dict):
                raise ValueError("options must be a dictionary.")
            kwargs["options"]["maxiter"] = maxiter
        self.result = minimize(self.get_loss, x0, method=method, **kwargs)

    def plot(
        self,
        axes=None,
        plot_ref=True,
        color=None,
        linestyle=None,
        color_ref="black",
        linestyle_ref="--",
    ):
        """Plot the fitted kernel."""
        self.ker_to_fit.set_from_array(np.abs(self.result.x))
        if axes is None:
            __, axes = plt.subplots(1, 2)
        if len(self.time_ker) > len(self.data_ker_int):
            time = self.time_ker
        else:
            time = self.time_ker_int
        axes[0].semilogx(time, self.ker_to_fit.get_kernel(time), color=color, linestyle=linestyle)
        axes[1].plot(time, self.ker_to_fit.get_kernel_int(time), color=color, linestyle=linestyle)
        if plot_ref:
            # time_ref = np.arange(len(self.ref_ker)) * self.dt
            axes[0].semilogx(self.time_ker, self.data_ker, color=color_ref, linestyle=linestyle_ref)
            axes[1].plot(
                self.time_ker_int, self.data_ker_int, color=color_ref, linestyle=linestyle_ref
            )
        return axes
