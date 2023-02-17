import numpy as np
import matplotlib.pyplot as plt
from abc import ABC


class RefKernel(ABC):

    """Marker class."""

    def set_from_array(self, array):
        """Set the kernel parameters from a single array."""
        raise NotImplementedError

    def get_kernel(self, time):
        """Compute the analytical kernel."""
        raise NotImplementedError

    def get_kernel_int(self, time):
        """Compute the analytical running integral over the kernel."""
        raise NotImplementedError


class RefKernelExponential(RefKernel):
    """
    Exponential Reference kernel for a Markovian embedding.
    Gamma(t) = sum_i k_i exp(-t / tau_i), with tau_i = gamma_i / k_i
    """

    def __init__(self, gammas: np.ndarray, ks: np.ndarray):
        self.gammas = gammas
        self.ks = ks

    @classmethod
    def from_params(cls, gammas, ks, taus):
        """Create from parameters, check that the right ones are not None."""
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
        """Create from gammas and ks, no checks."""
        return cls(gammas, ks)

    def set_from_array(self, array):
        """Set the kernel parameters from a single array."""
        if len(array) % 2 != 0:
            raise ValueError("The array must have an even length")
        self.gammas = array[: len(array) // 2]
        self.ks = array[len(array) // 2 :]

    def get_kernel(self, time):
        """Compute the analytical kernel."""
        kernel = np.zeros_like(time)
        for gamma, k in zip(self.gammas, self.ks):
            tau = gamma / k
            kernel += k * np.exp(-time / tau)
        return kernel

    def get_kernel_int(self, time):
        """Compute the analytical running integral over the kernel."""
        kernel_int = np.zeros_like(time)
        for gamma, k in zip(self.gammas, self.ks):
            tau = gamma / k
            kernel_int += gamma * (1 - np.exp(-time / tau))
        return kernel_int

    def plot(self, time, axes=None, color="black", ls="--", label="ref", **kwargs):
        """Add plot of the kernel and its integral to the given axes."""
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(time, self.get_kernel(time), color=color, ls=ls, label=label, **kwargs)
        axes[1].plot(time, self.get_kernel_int(time), color=color, ls=ls, label=label, **kwargs)
        return axes


class RefKernelOscillating(RefKernel):
    """
    Oscillating, exponentially decaying Reference kernel for a Markovian embedding.
    Gamma(t) = sum_i k_i exp(-t / tau_i) (cos(omega_i t) + 1 / (tau_i omega_i) sin(omega_i t))
    Internally, we store the simulation parameters.
    """

    def __init__(self, ks: np.ndarray, taus: np.ndarray, omegas: np.ndarray) -> None:
        if len(ks) != len(taus) != len(omegas):
            raise ValueError(
                "ks, taus, omegas must have the same length, "
                f"but have lengths {len(ks)=}, {len(taus)=} and {len(omegas)=}."
            )
        self.ks = ks
        self.taus = taus
        self.omegas = omegas

    @classmethod
    def from_sim_params(cls, ks, gammas, masses):
        """Create from the fit parameters."""
        if len(ks) != len(gammas) != len(masses):
            raise ValueError(
                "ks, gammas, masses must have the same length, "
                f"but have lengths {len(ks)=}, {len(gammas)=} and {len(masses)=}."
            )
        taus = 2 * masses / gammas
        omegas = np.sqrt(ks / masses - 1 / taus**2)
        return cls(ks, taus, omegas)

    def get_sim_params(self, as_dict=False):
        """Return the fit parameters."""
        masses = self.ks / (self.omegas**2 + 1 / self.taus**2)
        gammas = 2 * masses / self.taus
        if as_dict:
            return dict(ks=self.ks, gammas=gammas, masses=masses)
        return self.ks, gammas, masses

    def set_from_array(self, array):
        """Set the kernel parameters from a single array."""
        if len(array) % 3 != 0:
            raise ValueError("The length or athe array must be divisible by 3.")
        self.ks = array[: len(array) // 3]
        self.taus = array[len(array) // 3 : 2 * len(array) // 3]
        self.omegas = array[2 * len(array) // 3 :]

    def get_kernel(self, time):
        """Compute the analytical kernel."""
        kernel = np.zeros_like(time)
        for i in range(len(self.taus)):
            kernel += (
                self.ks[i]
                * np.exp(-time / self.taus[i])
                * (
                    np.cos(self.omegas[i] * time)
                    + 1 / (self.taus[i] * self.omegas[i]) * np.sin(self.omegas[i] * time)
                )
            )
        return kernel

    def get_kernel_int(self, time):
        """Compute the analytical running integral over the kernel."""
        _ks, gammas, _masses = self.get_sim_params()
        kernel_int = np.zeros_like(time)
        for i in range(len(self.taus)):
            kernel_int += gammas[i] + self.ks[i] * np.exp(-time / self.taus[i]) * (
                (self.taus[i] ** 2 * self.omegas[i] ** 2 - 1) * np.sin(self.omegas[i] * time)
                - 2 * self.taus[i] * self.omegas[i] * np.cos(self.omegas[i] * time)
            ) / (self.taus[i] ** 2 * self.omegas[i] ** 3 + self.omegas[i])
        return kernel_int

    def plot(self, time, axes=None, color="black", ls="--", label="ref", **kwargs):
        """Add plot of the kernel and its integral to the given axes."""
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(time, self.get_kernel(time), color=color, ls=ls, label=label, **kwargs)
        axes[1].plot(time, self.get_kernel_int(time), color=color, ls=ls, label=label, **kwargs)
        return axes
