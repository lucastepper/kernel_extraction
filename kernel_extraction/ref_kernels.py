from abc import ABC
from typing import Callable, Optional, Union
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


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

    def set_params(self, params: np.ndarray):
        """Set the kernel parameters from a single array."""
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
    def from_gammas_ks_taus(
        cls,
        gammas: Optional[np.ndarray] = None,
        ks: Optional[np.ndarray] = None,
        taus: Optional[np.ndarray] = None,
    ):
        """Create from gammas, and either ks or taus."""
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

    def set_params(self, params: np.ndarray):
        """Set the kernel parameters from a single array."""
        self.gammas = params[: len(params) // 2]
        self.ks = params[len(params) // 2 :]

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

    def __init__(self, gammas: np.ndarray, ks: np.ndarray, masses: np.ndarray) -> None:
        if len(gammas) != len(ks) != len(masses):
            raise ValueError(
                "gammas, ks, masses must have the same length, "
                f"but have lengths {len(gammas)=}, {len(ks)=} and {len(masses)=}."
            )
        self.gammas = gammas
        self.ks = ks
        self.masses = masses

    @classmethod
    def from_ks_taus_omegas(cls, ks: np.ndarray, taus: np.ndarray, omegas: np.ndarray):
        """Create from the fit parameters."""
        if len(ks) != len(taus) != len(omegas):
            raise ValueError(
                "ks, taus, omegas must have the same length, "
                f"but have lengths {len(ks)=}, {len(taus)=} and {len(omegas)=}."
            )
        masses = ks / (omegas**2 + 1 / taus**2)
        gammas = 2 * masses / taus
        return cls(gammas, ks, masses)

    def get_ks_taus_omegas(self, as_dict: bool = False):
        """Return the fit parameters."""
        taus = 2 * self.masses / self.gammas
        omegas = np.sqrt(self.ks / self.masses - (0.5 * self.gammas / self.masses) ** 2)
        if as_dict:
            return {"ks": self.ks, "taus": taus, "omegas": omegas}
        return self.ks, taus, omegas

    def set_params(self, params: np.ndarray):
        """Set the kernel parameters from a single array."""
        if len(params) % 3 != 0:
            raise ValueError("The length or athe array must be divisible by 3.")
        self.gammas = params[: len(params) // 3]
        self.ks = params[len(params) // 3 : 2 * len(params) // 3]
        self.masses = params[2 * len(params) // 3 :]

    def get_kernel(self, time):
        """Compute the analytical kernel."""
        kernel = np.zeros_like(time)
        ks, taus, omegas = self.get_ks_taus_omegas()
        for i in range(len(taus)):
            kernel += (
                ks[i]
                * np.exp(-time / taus[i])
                * (np.cos(omegas[i] * time) + 1 / (taus[i] * omegas[i]) * np.sin(omegas[i] * time))
            )
        return kernel

    def get_kernel_int(self, time):
        """Compute the analytical running integral over the kernel."""
        kernel_int = np.zeros_like(time)
        ks, taus, omegas = self.get_ks_taus_omegas()
        for i in range(len(taus)):
            kernel_int += self.gammas[i] + ks[i] * np.exp(-time / taus[i]) * (
                (taus[i] ** 2 * omegas[i] ** 2 - 1) * np.sin(omegas[i] * time)
                - 2 * taus[i] * omegas[i] * np.cos(omegas[i] * time)
            ) / (taus[i] ** 2 * omegas[i] ** 3 + omegas[i])
        return kernel_int

    def plot(self, time, axes=None, color="black", ls="--", label="ref", **kwargs):
        """Add plot of the kernel and its integral to the given axes."""
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(time, self.get_kernel(time), color=color, ls=ls, label=label, **kwargs)
        axes[1].plot(time, self.get_kernel_int(time), color=color, ls=ls, label=label, **kwargs)
        return axes


class RefKernelOscillatingPosMemSplines(RefKernel):
    """
    Reference kernel for a Markovian embedding with position dependent memory.
    We use the zwanzig model, were the coupling between the auxiliary variable y and the
    central variable x is given by V(x, y) = k_i^2 / 2 (y_i - alpha_i(x)) and the dynamics of y is
    given by a underdamped Langevin equation with friction gammas and masses masses.
    tau_i is given by 2 * masses_i / gammas_i and omega is given by
    omega_i = sqrt(k_i^2 / masses_i - 1 / taus_i^2).
    The position dependence of alpha_i is given by cubic splines.
    The kernel is given by:
    Gamma(x, y, t) = sum_i k_i exp(-t / tau_i) (cos(omega_i t) + sin(omega_i t) / (omega_i tau_i))
        * (dalpha_i_dx)^2
    Arguments:
        ks (np.ndarray): The coupling constants k_i.
        taus (np.ndarray): The memory times tau_i.
        omegas (np.ndarray): The frequencies omega_i.
        xvals_alpha (np.ndarray): The x values at which the alpha_i are given.
        yvals_alpha (np.ndarray): The y values at which the alpha_i are given.
        mass_x (float, np.ndarray): The mass of the central variable x, we divide the kernel by it.
            if it is an array, it needs to be a 2d array with [xvals, masses] that is then
            interpolated linearly to get the mass as a function of x.
    """

    def __init__(
        self,
        ks: np.ndarray,
        taus: np.ndarray,
        omegas: np.ndarray,
        xvals_alpha: np.ndarray,
        yvals_alpha: np.ndarray,
        mass_x: Union[float, np.ndarray] = 1.0,
    ) -> None:
        super().__init__()
        self.ks = ks
        self.taus = taus
        self.omegas = omegas
        self.xvals_alpha = xvals_alpha
        self.yvals_alpha = yvals_alpha
        self.mass_x = mass_x
        self.sm = None
        self.check_length()

    def get_color(self, value: float):
        """Get the color map value for a given value, which could be time or position."""
        if self.sm is None:
            raise ValueError("Need to call plot_vs_time or plot_vs_pos first.")
        return self.sm.to_rgba(value)

    def get_mass(self, pos: Union[float, np.ndarray]):
        """Get the mass at a given position."""
        if isinstance(self.mass_x, float):
            return self.mass_x
        return np.interp(pos, self.mass_x[0], self.mass_x[1])

    @classmethod
    def from_sim_params(
        cls,
        ks: np.ndarray,
        gammas: np.ndarray,
        masses: np.ndarray,
        xvals_alpha: np.ndarray,
        yvals_alpha: np.ndarray,
        mass_x: Union[float, np.ndarray] = 1.0,
    ):
        """Create from the fit parameters."""
        taus = 2 * masses / gammas
        omegas = np.sqrt(ks / masses - 1 / taus**2)
        instance = cls(ks, taus, omegas, xvals_alpha, yvals_alpha, mass_x=mass_x)
        instance.check_length()
        return instance

    def check_length(self) -> None:
        """Make sure all lenght are the same."""
        if (
            len(self.ks)
            != len(self.taus)
            != len(self.omegas)
            != len(self.xvals_alpha)
            != len(self.yvals_alpha)
        ):
            raise ValueError(
                "ks, taus, omegas, x/yvals_alpha must have the same length, "
                f"but have lengths {len(self.ks)=}, {len(self.taus)=}, {len(self.omegas)=}, "
                f"{len(self.xvals_alpha)=}, {len(self.yvals_alpha)=}."
            )

    def get_sim_params(self, as_dict: bool = False):
        """Return the fit parameters."""
        masses = self.ks / (self.omegas**2 + 1 / self.taus**2)
        gammas = 2 * masses / self.taus
        if as_dict:
            return {
                "ks": self.ks,
                "gammas": gammas,
                "masses": masses,
                "xvals_alpha": self.xvals_alpha,
                "yvals_alpha": self.yvals_alpha,
            }
        return self.ks, gammas, masses, self.xvals_alpha, self.yvals_alpha

    def set_params(self, params: np.ndarray):
        """Set the kernel parameters from a single array."""
        naux = len(self.xvals_alpha)
        params_array = params[: 3 * naux]
        nvals_alpha = int(len(params[3 * naux :]) / 2 / naux)
        alpha_array = params[3 * naux :]
        if nvals_alpha < 2:
            raise ValueError(f"Need at least 2 nvals_alpha, but got {nvals_alpha}")
        if not len(params) == 3 * params_array + 2 * params_array * nvals_alpha:
            raise ValueError(
                f"Got {naux=}, {len(params)=} and computed {nvals_alpha=}. "
                "We need len(array) to be 3 * params_array + 2 * params_array * nvals_alpha."
            )
        self.ks = params_array[: naux // 3]
        self.taus = params_array[naux // 3 : 2 * naux // 3]
        self.omegas = params_array[2 * naux // 3 :]
        self.xvals_alpha = alpha_array[: len(alpha_array) // 2].reshape(naux, nvals_alpha)
        self.yvals_alpha = alpha_array[len(alpha_array) // 2 :].reshape(naux, nvals_alpha)

    def get_interpolation_objs(self):
        """Interpolate the different xvals and yvals using cubic splines."""
        return [CubicSpline(x, y) for x, y in zip(self.xvals_alpha, self.yvals_alpha)]

    def get_kernel(
        self,
        time: Union[float, np.ndarray],
        pos: Union[float, np.ndarray],
        dalphas_dx_overwrite: Optional[Callable] = None,
    ):
        """Compute the analytical kernel.
        Arguments:
            time (float, np.ndarray): The time points at which to evaluate the kernel.
            pos (float, np.ndarray): The time points at which to evaluate the kernel.
            dalphas_dx_overwrite (Optional[Callable]): If given, use this instead of the
                interpolation objects. Needs to be a list of functions, taking only a pos array
                as input, with len == len(self.masses).
        Returns:
            kernel (np.ndarray[len(time), len(pos)]): The kernel evaluated at the
                given time and pos points. If input is float, return float.
        """
        time = np.atleast_1d(time)
        pos = np.atleast_1d(pos)
        if dalphas_dx_overwrite is None:
            interpolation_objs = self.get_interpolation_objs()
        kernel = np.zeros((len(time), len(pos)))
        for i in range(len(self.taus)):
            if dalphas_dx_overwrite is not None:
                dalphas_dx = dalphas_dx_overwrite[i](pos)
            else:
                dalphas_dx = interpolation_objs[i](pos, nu=1)
            pos_dependence = dalphas_dx**2 / self.get_mass(pos)
            time_dependence = (
                self.ks[i]
                # pylint: disable=invalid-unary-operand-type
                * np.exp(-time / self.taus[i])
                * (
                    np.cos(self.omegas[i] * time)
                    + np.sin(self.omegas[i] * time) / (self.omegas[i] * self.taus[i])
                )
            )
            kernel += pos_dependence.reshape(1, -1) * time_dependence.reshape(-1, 1)
        if kernel.shape == (1, 1):
            return kernel[0, 0]
        return kernel

    def get_kernel_int(
        self,
        time: Union[float, np.ndarray],
        pos: Union[float, np.ndarray],
        dalphas_dx_overwrite: Optional[Callable] = None,
    ):
        """Compute the analytical running integral over the kernel.
        Arguments:
            time (float, np.ndarray): The time points at which to evaluate the kernel.
            pos (float, np.ndarray): The time points at which to evaluate the kernel.
            dalphas_dx_overwrite (Optional[Callable]): If given, use this instead of the
                interpolation objects. Needs to be a list of functions, taking only a pos array
                as input, with len == len(self.masses).
        Returns:
            kernel_int (np.ndarray[len(time), len(pos)]): The kernel evaluated at the
                given time and pos points. If input is float, return float.
        """
        time = np.atleast_1d(time)
        pos = np.atleast_1d(pos)
        _ks, gammas, _masses, _xvals_alpha, _yvals_alpha = self.get_sim_params()
        if dalphas_dx_overwrite is None:
            interpolation_objs = self.get_interpolation_objs()
        kernel_int = np.zeros((len(time), len(pos)))
        for i in range(len(self.taus)):
            if dalphas_dx_overwrite is not None:
                dalphas_dx = dalphas_dx_overwrite[i](pos)
            else:
                dalphas_dx = interpolation_objs[i](pos, nu=1)
            pos_dependence = dalphas_dx**2 / self.get_mass(pos)
            time_dependence = (
                # pylint: disable=invalid-unary-operand-type
                gammas[i]
                + self.ks[i]
                * np.exp(-time / self.taus[i])
                * (
                    (self.taus[i] ** 2 * self.omegas[i] ** 2 - 1) * np.sin(self.omegas[i] * time)
                    - 2 * self.taus[i] * self.omegas[i] * np.cos(self.omegas[i] * time)
                )
                / (self.taus[i] ** 2 * self.omegas[i] ** 3 + self.omegas[i])
            )
            kernel_int += pos_dependence.reshape(1, -1) * time_dependence.reshape(-1, 1)
        if kernel_int.shape == (1, 1):
            return kernel_int[0, 0]
        return kernel_int

    def plot_vs_time(
        self, time, pos, axes=None, cm="nipy_spectral", lognorm=True, ls="--", label=None, **kwargs
    ):
        """Add plot of the kernel and its integral to the given axes."""
        time = np.array(time)
        pos = np.array(pos)
        if "dalphas_dx_overwrite" in kwargs:
            dalphas_dx_overwrite = kwargs.pop("dalphas_dx_overwrite")
        else:
            dalphas_dx_overwrite = None
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        kernel = self.get_kernel(time, pos, dalphas_dx_overwrite=dalphas_dx_overwrite)
        kernel_int = self.get_kernel_int(time, pos, dalphas_dx_overwrite=dalphas_dx_overwrite)
        cm_base = plt.cm.get_cmap(cm)
        if lognorm:
            norm = mpl.colors.LogNorm(vmin=np.log(pos.min()), vmax=np.log(pos.max()))
        else:
            norm = mpl.colors.Normalize(vmin=pos.min(), vmax=pos.max())
        self.sm = mpl.cm.ScalarMappable(cmap=cm_base, norm=norm)
        # actual plotting
        for i, x in enumerate(pos):
            if label is not None:
                label_plot = f"x={x:.2e}"
            else:
                label_plot = None
            axes[0].plot(
                time, kernel[:, i], color=self.sm.to_rgba(x), ls=ls, label=label_plot, **kwargs
            )
            axes[1].plot(
                time, kernel_int[:, i], color=self.sm.to_rgba(x), ls=ls, label=label_plot, **kwargs
            )
        plt.colorbar(self.sm, ax=axes[1])
        return axes

    def plot_vs_pos(
        self, time, pos, axes=None, cm="nipy_spectral", lognorm=True, ls="--", label=None, **kwargs
    ):
        """Add plot of the kernel and its integral to the given axes."""
        time = np.array(time)
        pos = np.array(pos)
        if "dalphas_dx_overwrite" in kwargs:
            dalphas_dx_overwrite = kwargs.pop("dalphas_dx_overwrite")
        else:
            dalphas_dx_overwrite = None
        if axes is None:
            _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        kernel = self.get_kernel(time, pos, dalphas_dx_overwrite=dalphas_dx_overwrite)
        kernel_int = self.get_kernel_int(time, pos, dalphas_dx_overwrite=dalphas_dx_overwrite)
        cm_base = plt.cm.get_cmap(cm)
        if lognorm:
            norm = mpl.colors.LogNorm(vmin=np.log(time.min()), vmax=np.log(time.max()))
        else:
            norm = mpl.colors.Normalize(vmin=time.min(), vmax=time.max())
        self.sm = mpl.cm.ScalarMappable(cmap=cm_base, norm=norm)
        # actual plotting
        for i, t in enumerate(time):
            if label is not None:
                label_plot = f"t={t:.2e}"
            else:
                label_plot = None
            axes[0].plot(
                pos, kernel[i], color=self.sm.to_rgba(t), ls=ls, label=label_plot, **kwargs
            )
            axes[1].plot(
                pos, kernel_int[i], color=self.sm.to_rgba(t), ls=ls, label=label_plot, **kwargs
            )
        plt.colorbar(self.sm, ax=axes[1])
        return axes
