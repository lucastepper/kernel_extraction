from dataclasses import dataclass
from typing import Optional, Iterable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import plot_utils
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
        if self.kernel_ref is not None:
            self.kernel_ref.plot(time, axes=axes)
        return axes


class Kernel2D:
    """
    A wrapper for a position dependent memory kernel.
    Handles computation and plotting. Assummes kernel (ncorr, nbins).
    """

    def __init__(
        self,
        dt: Optional[float] = None,
        kernel: Optional[np.ndarray] = None,
        kernel_integral: Optional[np.ndarray] = None,
        cache: Optional[bool] = True,
        bins: Optional[np.ndarray] = None,
        fe: Optional[np.ndarray] = None,
        ends: Optional[Iterable] = None,
    ) -> None:
        if not isinstance(dt, (float, int)):
            raise TypeError(f"Dt needs to be float or int, but found {type(dt)}.")
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
        self._ends = ends

    @property
    def bins(self):
        """Check if bins are set, otherwise raise error."""
        if self._bins is None:
            raise ValueError("Bins not set.")
        return self._bins

    @property
    def fe(self):
        """Check if fe is set, otherwise raise error. If set, check dimension.
        If dim is one, return self._bins, self._fe. If dim is two, return *self._fe.
        """
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
        """If self._kernel is set, return it. Otherwise, compute it from self._kernel_integral
        via numerical gradient. If neither is set, raise error.
        """
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
        """If self._kernel_integral is set, return it. Otherwise, compute it from self._kernel
        via numerical integration. If neither is set, raise error.
        """
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
        """Get the number of bins of kernel, either from self._kernel.shape[1]
        or self._kernel_integral.shape[1]. If neither is set, raise error.
        """
        if not self._kernel is None:
            return self._kernel.shape[1]
        if not self._kernel_integral is None:
            return self._kernel_integral.shape[1]
        raise ValueError("Either Gamma or G need to be in self.")

    @property
    def ncorr(self):
        """Get the number of time corrpoints from either self._kernel or self._kernel_integral"""
        if not self._kernel is None:
            return self._kernel.shape[0]
        if not self._kernel_integral is None:
            return self._kernel_integral.shape[0]
        raise ValueError("Either Gamma or G need to be in self.")

    @property
    def time(self):
        """Compute a time array for the kernel."""
        return np.arange(self.ncorr) * self.dt

    @property
    def ends(self):
        """Get the maximum time that the indivudal bins are considered relevant.
        Defaults to the full time range.
        """
        if self._ends is None:
            return np.ones(self.kernel.shape[1], dtype=int) * self.kernel.shape[0]
        return self._ends

    def get_averaged(
        self,
        obs: np.ndarray,
        range_obs: tuple[float] = (-np.inf, np.inf),
        _idxs_obs: Optional[np.ndarray] = None,
    ):
        """Compute a purely time-dependent observable averaged by averaging
        over all bins for an observable that depends on time and space.
        For each obs that is a function of position, only consider to time indexes
        that are lower than self.ends[idx_pos].
        Expects bins and free energy to be in self.
        obs: (ncorr, nbins): data to be averaged
        range (2,): range of bins to be considered, in units of bins
        idxs_obs: (np.ndarray): indexes (with corresponding times = dt * indexes) at which
            the observable to average is evaluated, needed only when leaving out time indexes
            according to self.ends. Defaults to np.arange(len(obs)).
        """
        values, fe = self.fe
        ends = self.ends
        if obs.shape[1] != self.bins.shape[0] != values.shape[0] != fe.shape[0]:
            raise ValueError("Dimension mismatch between number of bins.")
        mask_bins = (self.bins >= range_obs[0]) & (self.bins < range_obs[1])
        weights = np.exp(-np.interp(self.bins, values, fe))
        weights /= np.trapz(weights[mask_bins], self.bins[mask_bins])
        obs_summed = np.zeros(obs.shape[0])
        weights_summed = np.zeros(obs.shape[0])
        for idx_pos in range(obs.shape[1]):
            if not mask_bins[idx_pos]:
                continue
            # select all elements from idx_pos that are not nan
            mask = obs[:, idx_pos] == obs[:, idx_pos]
            # select all elements from idx_pos that have time < ends[idx_pos]
            # subsampling did not work, maybe todo to fix this, later
            # if idxs_obs is None:
            #     nframes_obs = self.ends[idx_pos]
            # else:
            #     nframes_obs = np.sum(idxs_obs < self.ends[idx_pos])
            mask[int(ends[idx_pos]) :] = False
            obs_summed[mask] += weights[idx_pos] * obs[:, idx_pos][mask]
            weights_summed[: int(ends[idx_pos])] += weights[idx_pos]
        return obs_summed / weights_summed

    def get_conv_func(self, nsamples_logspace: Optional[int] = None):
        """Get a function that subsamples the kernel in logspace."""
        if nsamples_logspace is not None:
            return lambda t, x: plot_utils.subsample_logspace(
                x, dt=self.dt, n=nsamples_logspace, time=t
            )
        return lambda t, x: (t, x)

    def plot_vs_time(
        self,
        bins=None,
        axes=None,
        cm="viridis_r",
        nsamples_logspace: Optional[int] = None,
    ):
        """Plot kernel vs time at given bins."""
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if bins is None:
            bins = np.arange(self.nbins)
        bin_values = self.bins[bins]
        conv_func = self.get_conv_func(nsamples_logspace)
        self.sm = plot_utils.get_colormap(cm, vmin=np.min(bin_values), vmax=np.max(bin_values))
        for ibin, bin_val in zip(bins, bin_values):
            time_ker, ker = conv_func(self.time, self.kernel[: int(self.ends[ibin]), int(ibin)])
            t_int, ker_int = conv_func(
                self.time, self.kernel_integral[: int(self.ends[ibin]), int(ibin)]
            )
            if np.any(np.isnan(ker)) or np.any(np.isnan(ker_int)):
                print(f"Skipping bin {ibin} because it contains nan.")
                continue
            axes[0].plot(time_ker[:-1], ker[:-1], color=self.sm.to_rgba(bin_val))
            axes[1].plot(
                t_int,
                ker_int,
                color=self.sm.to_rgba(bin_val),
            )
        plt.colorbar(self.sm, ax=axes[1])
        return axes

    def plot_vs_pos(self, tcorrs=None, axes=None, cm="viridis_r"):
        """Plot kernel vs position at given time points."""
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if tcorrs is None:
            tcorrs = np.arange(self.ncorr)
        if np.any(np.array(tcorrs) > self.ncorr):
            raise ValueError("All tcorrs must be smaller than ncorr.")
        self.sm = plot_utils.get_colormap(cm, vmin=tcorrs[0] * self.dt, vmax=tcorrs[-1] * self.dt)
        if (bins := self.bins) is None:
            bins = np.arange(0, self.nbins, 1.0)
        mask = self.bins < self.ends
        for icorr in tcorrs:
            kernel, kernel_int = self.kernel[int(icorr)], self.kernel_integral[int(icorr)]
            axes[0].plot(bins[mask], kernel[mask], color=self.sm.to_rgba(self.dt * icorr))
            axes[1].plot(bins[mask], kernel_int[mask], color=self.sm.to_rgba(self.dt * icorr))
        plt.colorbar(self.sm, ax=axes[1], label="$t$ [ps]")
        return axes

    def plot_averaged(
        self,
        axes=None,
        kernel_ref=None,
        kernel_int_ref=None,
        range_obs=(-np.inf, np.inf),
        nsamples_logspace: Optional[int] = None,
        label=None,
        label_ref=None,
    ):
        """Compute the kernel avearged over space, weighted by self.fe, and plot it vs time."""
        if kernel_ref is not None and kernel_int_ref is None:
            kernel_int_ref = cumtrapz(kernel_ref, dx=self.dt, axis=0, initial=0.0)
        if kernel_int_ref is not None and kernel_ref is None:
            kernel_ref = np.gradient(kernel_int_ref, self.dt)
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # subsampling on a logspace does not work right now, as selg.get_averaged looks
        # at how many indices of the kernel are smaller than self.ends[idx_pos]
        # which breaks if the kernel is subsampled
        conv_func = self.get_conv_func(nsamples_logspace)
        # # average kernel for the requested times and range of bins
        # time, kernel = conv_func(self.time, self.kernel)
        # time_int, kernel_int = conv_func(self.time, self.kernel_integral)
        # # convert the time arrays to a time index array so we can compare it to self.ends
        # time_idxs = time / self.dt
        # time_int_idxs = time_int / self.dt
        # axes[0].semilogx(
        #     time[:-1], self.get_averaged(kernel, range_obs, idxs_obs=time_idxs)[:-1], label=label
        # )
        # axes[1].plot(
        #     time_int, self.get_averaged(kernel_int, range_obs, idxs_obs=time_int_idxs),
        # label=label
        # )
        # The subsampling for computing average still did not work
        axes[0].semilogx(
            self.time[:-1], self.get_averaged(self.kernel, range_obs)[:-1], label=label
        )
        axes[1].plot(self.time, self.get_averaged(self.kernel_integral, range_obs), label=label)
        if kernel_ref is not None and kernel_int_ref is not None:
            axes[0].semilogx(
                *conv_func(self.time[:-1], kernel_ref[: len(self.time) - 1]),
                label=label_ref,
                color="black",
                linestyle="--",
            )
            axes[1].plot(
                *conv_func(self.time, kernel_int_ref[: len(self.time)]),
                label=label_ref,
                color="black",
                linestyle="--",
            )
        return axes
