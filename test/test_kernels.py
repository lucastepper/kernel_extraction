import os
from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction

CORRS_ARGS = {
    "g_method_xdu": ("vv_corr", "xdu_corr", "dt", "mass", "trunc"),
    "g_method_xdu_nomass": ("vv_corr", "xdu_corr", "dt", "trunc"),
    "g_method_xdu_halfstep": ("vv_corr", "xdu_corr", "dt", "mass", "trunc"),
    "g_method_vdu": ("vv_corr", "vdu_corr", "dt", "mass", "trunc"),
    "gamma_method": ("vv_corr", "va_corr", "aa_corr", "adu_corr", "dt", "mass", "trunc"),
}


@lru_cache(maxsize=1)
def cuda_skip_reason():
    if os.environ.get("SKIPCUDA") == "true":
        return "CUDA tests disabled by SKIPCUDA"

    # A one-sample output initializes the CUDA backend without launching an extraction step.
    corr = np.ones(1, dtype=np.float64)
    try:
        kernel_extraction.g_method_xdu_nomass(
            corr, corr, dt=1.0, trunc=1, device="cuda", verbose=False
        )
    except RuntimeError as error:
        return f"CUDA unavailable ({error})"
    return None


def g_method_xdu(vv_corr, xdu_corr, dt, mass, trunc, loop=True, verbose=False):
    """Gmethod with external potential."""
    trunc = np.min([len(vv_corr), len(xdu_corr), trunc])
    kernel_i = np.zeros(trunc)
    prefac = 2.0 / (vv_corr[0] * dt)
    for i in range(1, len(kernel_i)):
        # analytically, the difference between mass * vv_corr and xdu_corr is zero,
        # as they are both equal to kbt. Numerically, it helps to keep it in, though.
        kernel_i[i] = mass * (vv_corr[0] - vv_corr[i]) + xdu_corr[i] - xdu_corr[0]
        if loop:
            conv = 0.0
            for j in range(1, i):
                conv += kernel_i[i - j] * vv_corr[j]
        else:
            conv = np.sum(kernel_i[:i] * vv_corr[::-1][len(vv_corr) - i - 1 : -1])
        kernel_i[i] -= dt * conv
        kernel_i[i] *= prefac
    return kernel_i


def g_method_xdu_nomass(vv_corr, xdu_corr, dt, trunc, verbose=False):
    """Compute the integral over the kernel."""
    trunc = np.min([len(vv_corr), len(xdu_corr), trunc])
    kernel_i = np.zeros(trunc)
    prefac = 2.0 / vv_corr[0]
    for i in range(1, trunc):
        kernel_i[i] = prefac * (
            (xdu_corr[i] - vv_corr[i] * xdu_corr[0] / vv_corr[0]) / dt
            - np.sum(kernel_i[1:i] * vv_corr[1:i][::-1])
        )
    return kernel_i


def g_method_xdu_halfstep(vv_corr, xdu_corr, dt, mass, trunc, loop=True, verbose=False):
    """Gmethod with external potential, evaluating kernel at half steps."""
    trunc = np.min([len(vv_corr) - 1, len(xdu_corr) - 1, trunc])
    kernel_i = np.zeros(trunc)
    prefac = 1.0 / (vv_corr[0] + vv_corr[1])
    prefac_vv_corr = 2.0 / dt
    for i in range(len(kernel_i)):
        kernel_i[i] = prefac_vv_corr * (
            mass * (vv_corr[0] - vv_corr[i]) + xdu_corr[i] - xdu_corr[0]
        )
        if loop:
            conv = 0.0
            for j in range(1, i):
                conv += kernel_i[i - j] * (vv_corr[j] + vv_corr[j + 1])
        else:
            conv = np.sum(
                kernel_i[:i]
                * (
                    vv_corr[::-1][len(vv_corr) - i - 1 : -1]
                    + vv_corr[::-1][len(vv_corr) - i - 2 : -2]
                )
            )
        kernel_i[i] -= conv
        kernel_i[i] *= prefac
    return kernel_i


def g_method_vdu(vv_corr, vdu_corr, dt, mass, trunc, verbose=False):
    r"""This is an alternatice volterra method extracting the running integral
    over the memory kernel G(t), where the integral $\int_0^t ds C^{vdU}(s)$ is
    pulled into the convolution integral $\int_0^t ds G(t - s) C^{vv}(s)$ prior
    to approximation with trapezoidal rule.
    Arguments:
        vv_corr (np.ndarray): velocity autocorrelation function
        vdu_corr (np.ndarray): velocity gradient of PMF correlation function
        dt (float): time step
        mass (float): mass
        trunc (int): number of frames of G(t) to compute
        loop (bool): whether to loop to compute covolution integral
        verbose (bool): argument not used, I just added it for consistency with the module to test
    """
    trunc = np.min([len(vv_corr), len(vdu_corr), trunc])
    kernel_int = np.zeros(trunc)
    # to streamline the computation of all terms for the trapezodial rule
    # of the integral over vdu_corr, we precompute the cum sum of vdu_corr
    # where we set the first element to zero
    vdu_corr_cumsum = np.cumsum(np.concatenate([[0.0], vdu_corr[1:trunc]]))
    prefac = 1 / vv_corr[0]
    for i in range(1, len(kernel_int)):
        kernel_int[i] = -vdu_corr[0] - vdu_corr[i] + (2 * mass / dt) * (vv_corr[0] - vv_corr[i])
        conv = 0.0
        for j in range(1, i):
            # conv += kernel_int[j] * vv_corr[i - j] + vdu_corr[j]
            conv += kernel_int[j] * vv_corr[i - j]
        kernel_int[i] -= 2 * vdu_corr_cumsum[i - 1]
        kernel_int[i] -= 2 * conv
        kernel_int[i] *= prefac
    return kernel_int


def gamma_method(vv_corr, va_corr, aa_corr, adu_corr, dt, mass, trunc, verbose=False):
    """This is an alternatice volterra method extracting the memory kernel
    $Gamma(t)$ directly. It is based on a volterra method of the second kind.
    Arguments:
        vv_corr (np.ndarray): velocity autocorrelation function
        va_corr (np.ndarray): velocity acceleration correlation function
        aa_corr (np.ndarray): acceleration autocorrelation function
        adu_corr (np.ndarray): acceleration gradient of PMF correlation function
        dt (float): time step
        mass (float): mass
        trunc (int): number of frames of G(t) to compute
        verbose (bool): argument not used, I just added it for consistency with the module to test
    """
    aa_corr_times_mass = mass * aa_corr
    trunc = np.min([len(vv_corr), len(va_corr), len(aa_corr), len(adu_corr), trunc])
    kernel = np.zeros(trunc)
    prefac = 1.0 / (vv_corr[0] + va_corr[0] * dt * 0.5)
    kernel[0] = (aa_corr_times_mass[0] + adu_corr[0]) / vv_corr[0]
    for i in range(1, trunc):
        num = aa_corr_times_mass[i] + adu_corr[i]
        # do first instance of loop where j = 0 and we scale
        # with 0.5 due to trapezoidal rule
        num -= 0.5 * dt * va_corr[i] * kernel[0]
        for j in range(1, i):
            num -= dt * va_corr[i - j] * kernel[j]
        kernel[i] = num
        kernel[i] = prefac * num
    return kernel


def run_tests_slow(device, test_data, extraction_method):
    """Test kernel simulating new data, then also test kernel_from_traj."""
    test_data = utils.simulate_test_data()
    kernel_test_obj = kernel_extraction.kernel_from_traj(
        trajs=test_data.traj,
        dt=test_data.dt,
        trunc=test_data.trunc,
        kbt=test_data.kbt,
        bins=test_data.bins,
        extraction_method=extraction_method,
        device=device,
    )
    # plot
    time = np.arange(test_data.trunc - 1) * test_data.dt
    gammas = np.array(test_data.gammas)
    taus = gammas / np.array(test_data.ks)
    kernel_ref_obj = kernel_extraction.RefKernelExponential(gammas=gammas, taus=taus)
    if os.environ.get("PLOT") == "true":
        _fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(time, kernel_ref_obj.get_kernel(time), label="ref")
        axes[0].plot(time, kernel_test_obj.kernel[: len(time)], label="sim")
        axes[0].legend()
        axes[1].plot(time, kernel_ref_obj.get_kernel_int(time), label="ref")
        axes[1].plot(time, kernel_test_obj.kernel_int[: len(time)], label="sim")
        axes[1].set_title(f"{extraction_method.value} {device}")
        plt.show()
    # test vs analytical reference kernel
    rtol, atol = 5e-1, 1e-2
    # I think there is still a small error in the derivation of halfstep kernel
    if "halfstep" in extraction_method.value:
        rtol, atol = 1.0, 1e-2
    if "g_method" in extraction_method.value:
        np.testing.assert_allclose(
            kernel_test_obj.kernel[: len(time)],
            kernel_ref_obj.get_kernel(time),
            rtol=rtol,
            atol=atol,
        )
    else:
        np.testing.assert_allclose(
            kernel_test_obj.kernel_int[: len(time)],
            kernel_ref_obj.get_kernel_int(time),
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize(
    "method, device",
    [
        ("g_method_xdu", "cpu"),
        ("g_method_xdu", "cuda"),
        ("g_method_xdu_nomass", "cpu"),
        ("g_method_xdu_nomass", "cuda"),
        ("g_method_xdu_halfstep", "cpu"),
        ("g_method_xdu_halfstep", "cuda"),
        ("g_method_vdu", "cpu"),
        ("g_method_vdu", "cuda"),
        ("gamma_method", "cpu"),
        ("gamma_method", "cuda"),
    ],
)
def test_kernel(method, device):
    """Test extraction method versus reference implementation, loading correlations."""
    if device == "cuda":
        reason = cuda_skip_reason()
        if reason is not None:
            pytest.skip(reason)
    test_data = utils.load_test_data()
    # test that the length of the correlation functions is taken into account
    trunc = 2 * len(test_data.vv_corr)
    test_data.trunc = trunc
    compute_ker_args = {arg: getattr(test_data, arg) for arg in CORRS_ARGS[method]}
    compute_ker_args["verbose"] = False
    get_ref_corr = globals()[method]
    get_test_corr = getattr(kernel_extraction, method)
    kernel_i_ref = get_ref_corr(**compute_ker_args)
    # check that kernel_extraction.method function works directly
    kernel_i_test_1 = get_test_corr(**compute_ker_args, device=device)
    extraction_method = kernel_extraction.ExtractionMethod(method)
    compute_ker_args["kernel"] = None
    # check that kernel_extraction.compute_kernel works given method
    kernel_i_test_2 = kernel_extraction.compute_kernel(
        extraction_method, **compute_ker_args, device=device
    )
    compute_ker_args["kernel"] = kernel_i_test_2[:len(kernel_i_ref) // 2]
    # check that extending give the same result
    kernel_i_test_3 = kernel_extraction.compute_kernel(
        extraction_method, **compute_ker_args, device=device
    )
    # check that we actually use the input kernel
    compute_ker_args["kernel"] = np.zeros_like(kernel_i_ref)
    kernel_i_test_wrong = kernel_extraction.compute_kernel(
        extraction_method, **compute_ker_args, device=device
    )
    # Saved the data in singles, reduce rtol
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_1, rtol=5e-5)
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_2, rtol=5e-5)
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_3, rtol=5e-5)
    if np.allclose(kernel_i_ref, kernel_i_test_wrong, rtol=5e-5):
        raise AssertionError(f"We did not extend the input kernel for {method=} and {device=}.")
    # run tests with simulated data
    # check if we passed --runslow to pytest
    if os.environ.get("RUNSLOW") == "true":
        run_tests_slow(device, test_data, extraction_method)


@pytest.mark.parametrize("method", CORRS_ARGS)
def test_auto_device_matches_cpu(method):
    test_data = utils.load_test_data()
    args = {arg: getattr(test_data, arg) for arg in CORRS_ARGS[method]}
    args["trunc"] = 10
    args["verbose"] = False
    extraction_method = kernel_extraction.ExtractionMethod(method)
    expected = kernel_extraction.compute_kernel(extraction_method, **args, device="cpu")
    actual = kernel_extraction.compute_kernel(extraction_method, **args, device="auto")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
