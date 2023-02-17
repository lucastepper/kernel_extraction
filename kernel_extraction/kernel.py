from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from .ref_kernels import RefKernel


@dataclass
class Kernel:
    """
    Container to hold and plot the kernel and its integral.
    Later, computing the kernel from a trajectory will be added.
    """

    dt: float
    kernel: np.ndarray
    kernel_int: np.ndarray
    kernel_ref: Optional[RefKernel] = None

    @classmethod
    def from_kernel(cls, kernel, dt, ref_kernel=None):
        """Compute the running integral from the kernel."""
        kernel_int = cumtrapz(kernel, dx=dt, initial=0)
        return cls(dt, kernel, kernel_int, ref_kernel)

    @classmethod
    def from_kernel_int(cls, kernel_int, dt, halfstep=False, ref_kernel=None):
        """Compute the kernel from its running integral via numerical gradeint."""
        kernel = np.gradient(kernel_int, dt)
        if halfstep:
            # When using halfstep method, first frame is G_0, then G_0.5, G_1.5 ...
            # We need to correct the gradient of the first two frames accordingly
            kernel[0] = (kernel_int[1] - kernel_int[0]) / (0.5 * dt)
            kernel[1] = (kernel_int[2] - kernel_int[0]) / (1.5 * dt)
        return cls(dt, kernel, kernel_int, ref_kernel)

    def plot(self, axes=None, end=None, logx_kernel=True, logx_kernel_int=False, **kwargs):
        """Plot the kernel and its integral. If set, also plot the reference kernel."""
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        if end is None:
            end = len(self.kernel)
        time = np.arange(end) * self.dt
        axes[0].plot(time, self.kernel[:end], **kwargs)
        if logx_kernel:
            axes[0].set_xscale("log")
        axes[1].plot(time, self.kernel_int[:end], **kwargs)
        axes[0].set(xlabel=r"$t$ [ps]", ylabel=r"$\Gamma$ [u/ps$^{2}$]")
        if logx_kernel_int:
            axes[1].set_xscale("log")
        axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axes[1].set(xlabel=r"$t$ [ps]", ylabel=r"$G$ [u/ps]")
        if self.kernel_ref:
            self.kernel_ref.plot(time, axes=axes)
        return axes


class Kernel2D:
    """
    A wrapper for a position dependent memory kernel.
    Handles computation and plotting. Assummes kernel (ncorr, nbins).
    """

    def __init__(
        self, dt=None, kernel=None, kernel_integral=None, cache=True, bins=None, fe=None
    ) -> None:
        if not isinstance(dt, (float, int)):
            raise TypeError(f"Dt needs to be float or int.")
        if kernel is None and kernel_integral is None:
            raise TypeError("Either kernel or kernel_integral needs to be provided.")
        self.dt = float(dt)
        self._kernel = kernel
        self._kernel_integral = kernel_integral
        # set values for plotting
        self.cache = cache
        # make the color map last used for plotting available
        self.sm = None
        # attributes for plotting
        self._bins = bins
        self._fe = fe

    @property
    def bins(self):
        if self._bins is None:
            raise ValueError("Bins not set.")
        return self._bins

    @property
    def fe(self):
        if self._fe is None:
            raise ValueError("fe not set.")
        if self._fe.ndim == 1:
            print(
                "Assuming fe is evaluated at the bins of the kernel. To use different"
                "positons for fe, pass 2d fe, shape: (2, ninbs) that is (values, fe)."
            )
            return self._bins, self._fe
        return self._fe[0], self._fe[1]

    @property
    def kernel(self):
        if not self._kernel is None:
            return self._kernel
        if not self._kernel_integral is None:
            kernel = np.gradient(self._kernel_integral, self.dt, axis=0)
            if self.cache:
                self._kernel = kernel
            return kernel
        raise ValueError("Either Gamma or G to be in self.")

    @property
    def kernel_integral(self):
        if not self._kernel_integral is None:
            return self._kernel_integral
        if not self._kernel is None:
            kernel_int = cumtrapz(self._kernel, dx=self.dt, axis=0, initial=0.0)
            if self.cache:
                self._kernel_integral = kernel_int
            return kernel_int
        raise ValueError("Either Gamma or G need to be in self.")

    @property
    def nbins(self):
        if not self._kernel is None:
            return self._kernel.shape[1]
        if not self._kernel_integral is None:
            return self._kernel_integral.shape[1]
        raise ValueError("Either Gamma or G need to be in self.")

    @property
    def ncorr(self):
        if not self._kernel is None:
            return self._kernel.shape[0]
        if not self._kernel_integral is None:
            return self._kernel_integral.shape[0]
        raise ValueError("Either Gamma or G need to be in self.")

    @property
    def time(self):
        return np.arange(self.ncorr) * self.dt

    def get_sm(self, min, max, cm="viridis"):
        cm = plt.cm.get_cmap(cm)
        norm = plt.Normalize(vmin=min, vmax=max)
        self.sm = plt.cm.ScalarMappable(norm=norm, cmap=cm)
        return self.sm

    def get_averaged(self, obs: np.ndarray, range_obs: tuple[float] = (-np.inf, np.inf)):
        """Compute a purely time-dependent observable averaged by averaging
        over all bins for an observable that depends on time and space.
        Expects bins and free energy to be in self.
        obs: (ncorr, nbins): data to be averaged
        range (2,): range of bins to be considered, in units of bins
        """
        values, fe = self.fe
        if obs.shape[1] != self.bins.shape[0] != values.shape[0] != fe.shape[0]:
            raise ValueError("Dimension mismatch between number of bins.")
        mask = (self.bins >= range_obs[0]) & (self.bins < range_obs[1])
        weights = np.exp(-np.interp(self.bins[mask], values, fe))
        weights /= np.trapz(weights, self.bins[mask])
        obs_averaged = np.nansum(weights.reshape(1, -1) * obs[:, mask], axis=1) / np.nansum(weights)
        return obs_averaged

    def plot_vs_time(self, bins=None, axes=None, format="None", dx=None, start=None):
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if bins is None:
            bins = np.arange(self.nbins)
        sm = self.get_sm(self.values[0], self.values[-1])
        for ibin in bins:
            axes[0].plot(
                self.time[:-1], self.kernel[:-1, int(ibin)], color=sm.to_rgba(self.values[ibin])
            )
            axes[1].plot(
                self.time, self.kernel_integral[:, int(ibin)], color=sm.to_rgba(self.values[ibin])
            )
        plt.colorbar(sm, ax=axes[1])
        return axes

    def plot_vs_pos(self, tcorrs=None, axes=None, format="None", start=None):
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if tcorrs is None:
            tcorrs = np.arange(self.ncorr)
        if np.any(tcorrs > self.ncorr):
            raise ValueError("All tcorrs must be smaller than ncorr.")
        sm = self.get_sm(self.dt * tcorrs[0], self.dt * tcorrs[-1])
        if (bins := self.bins) is None:
            bins = np.arange(0, self.nbins, 1.0)
        for icorr in tcorrs:
            axes[0].plot(bins, self.kernel[int(icorr)], color=sm.to_rgba(self.dt * icorr))
            axes[1].plot(bins, self.kernel_integral[int(icorr)], color=sm.to_rgba(self.dt * icorr))
        plt.colorbar(sm, ax=axes[1], label="$t$ [ps]")
        return axes

    def plot_averaged(
        self, axes=None, kernel_ref=None, kernel_int_ref=None, range_obs=(-np.inf, np.inf)
    ):
        if kernel_ref is not None and kernel_int_ref is None:
            kernel_int_ref = cumtrapz(kernel_ref, dx=self.dt, axis=0, initial=0.0)
        if kernel_int_ref is not None and kernel_ref is None:
            kernel_ref = np.gradient(kernel_int_ref, self.dt)
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].semilogx(self.time[:-1], self.get_averaged(self.kernel, range_obs)[:-1])
        axes[1].plot(self.time, self.get_averaged(self.kernel_integral, range_obs))
        if kernel_ref is not None and kernel_int_ref is not None:
            axes[0].semilogx(
                self.time[:-1], kernel_ref[: len(self.time) - 1], color="black", linestyle="--"
            )
            axes[1].plot(self.time, kernel_int_ref[: len(self.time)], color="black", linestyle="--")
        return axes
