from dataclasses import dataclass
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import cumtrapz


@dataclass
class RefKernelExponential:
    """
    Exponential Reference kernel for a Markovian embedding. 
    Gamma(t) = sum_i k_i exp(-t / tau_i), with tau_i = gamma_i / k_i
    """
    gammas: np.ndarray
    ks: np.ndarray

    @classmethod
    def from_params(cls, gammas, ks, taus):
        """ Create from parameters, check that the right ones are not None. """
        if gammas is None:
            gammas = np.array([])
            ks = np.array([])
        elif ks is None and taus is not None:
            ks = gammas / taus
        elif ks is not None:
            ks = np.asarray(ks)
        else:
            raise ValueError("Either ks or taus must be provided if gammas is provided")
        if len(gammas) != len(ks):
            raise ValueError(
                "gammas and ks must have the same length, "
                f"but have lengths {len(gammas)} and {len(ks)}"
            )
        return cls(gammas, ks)

    @classmethod
    def from_gammas_ks(cls, gammas, ks):
        """ Create from gammas and ks, no checks. """
        return cls(gammas, ks)
    
    def get_kernel(self, time):
        """ Compute the analytical kernel. """
        kernel = np.zeros_like(time)
        for gamma, k in zip(self.gammas, self.ks):
            tau = gamma / k
            kernel += k * np.exp(-time / tau)
        return kernel

    def get_kernel_int(self, time):
        """ Compute the analytical running integral over the kernel. """
        kernel_int = np.zeros_like(time)
        for gamma, k in zip(self.gammas, self.ks):
            tau = gamma / k
            kernel_int += gamma * (1 - np.exp(-time / tau))
        return kernel_int
    
    def plot(self, time, axes=None, color="black", ls="--", label="ref", **kwargs):
        """ Add plot of the kernel and its integral to the given axes. """
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(time, self.get_kernel(time), color=color, ls=ls, label=label, **kwargs)
        axes[1].plot(time, self.get_kernel_int(time), color=color, ls=ls, label=label, **kwargs)
        return axes


@dataclass
class Kernel:
    """ 
    Container to hold and plot the kernel and its integral.
    Later, computing the kernel from a trajectory will be added.
    """
    dt: float
    kernel: np.ndarray
    kernel_int: np.ndarray
    kernel_ref: Optional[RefKernelExponential] = None

    @classmethod
    def from_kernel(cls, kernel, dt):
        """ Compute the running integral from the kernel. """
        kernel_int = cumtrapz(kernel, dx=dt, initial=0)
        return cls(dt, kernel, kernel_int)

    @classmethod
    def from_kernel_int(cls, kernel_int, dt, halfstep=False):
        """ Compute the kernel from its running integral via numerical gradeint. """
        kernel = np.gradient(kernel_int, dt)
        if halfstep:
            # When using halfstep method, first frame is G_0, then G_0.5, G_1.5 ...
            # We need to correct the gradient of the first two frames accordingly
            kernel[0] = (kernel_int[1] - kernel_int[0]) / (0.5 * dt)
            kernel[1] = (kernel_int[2] - kernel_int[0]) / (1.5 * dt)
        return cls(dt, kernel, kernel_int)

    def set_params_kernel_ref(self, gammas=None, ks=None, taus=None):
        """ Set the reference kernel parameters, to allow plotting it. """
        self.kernel_ref = RefKernelExponential.from_params(gammas, ks, taus)

    def plot(self, axes=None, end=None, logx_kernel=True, logx_kernel_int=False, **kwargs):
        """ Plot the kernel and its integral. If set, also plot the reference kernel. """
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