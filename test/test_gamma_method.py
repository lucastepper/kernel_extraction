import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction


def gamma_method(vv_corr, va_corr, af_corr, adu_corr, dt, trunc):
    """This is an alternatice volterra method extracting the memory kernel
    $Gamma(t)$ directly. It is based on a volterra method of the second kind.
    Arguments:
        vv_corr (np.ndarray): velocity autocorrelation function
        va_corr (np.ndarray): velocity acceleration correlation function
        af_corr (np.ndarray): acceleration force correlation function (af_corr = mass * caa)
        adu_corr (np.ndarray): acceleration gradient of PMF correlation function
        dt (float): time step
        trunc (int): number of frames of G(t) to compute
    """
    trunc = np.min([len(vv_corr), len(va_corr), len(af_corr), len(adu_corr), trunc])
    kernel = np.zeros(trunc)
    prefac = 1.0 / (vv_corr[0] + va_corr[0] * dt * 0.5)
    kernel[0] = (af_corr[0] + adu_corr[0]) / vv_corr[0]
    for i in range(1, trunc):
        num = af_corr[i] + adu_corr[i]
        # do first instance of loop where j = 0 and we scale
        # with 0.5 due to trapezoidal rule
        num -= 0.5 * dt * va_corr[i] * kernel[0]
        for j in range(1, i):
            num -= dt * va_corr[i - j] * kernel[j]
        kernel[i] = num
        kernel[i] = prefac * num
    return kernel


@pytest.mark.parametrize("device", ["cpu", "cuda",],)
def test_g_method_vdu(device):
    """Test extraction method versus reference implementation."""
    test_data = utils.load_test_data()
    # test that the length of the correlation functions is taken into account
    trunc = 2 * len(test_data.vv_corr)
    kernel_ref = gamma_method(
        test_data.vv_corr,
        test_data.va_corr,
        test_data.mass * test_data.aa_corr,
        test_data.adu_corr,
        test_data.dt,
        trunc,
    )
    kernel_test_1 = kernel_extraction.gamma_method(
        test_data.vv_corr,
        test_data.va_corr,
        test_data.aa_corr,
        test_data.adu_corr,
        test_data.dt,
        test_data.mass,
        trunc,
        device,
    )
    kernel_test_2 = kernel_extraction.compute_kernel(
        kernel_extraction.ExtractionMethod.GAMMA_METHOD,
        test_data.dt,
        vv_corr=test_data.vv_corr,
        va_corr=test_data.va_corr,
        aa_corr=test_data.aa_corr,
        adu_corr=test_data.adu_corr,
        mass=test_data.mass,
        trunc=trunc,
        device=device,
    )
    # Saved the data in singles, reduce rtol
    np.testing.assert_allclose(kernel_ref, kernel_test_1, rtol=5e-5)
    np.testing.assert_allclose(kernel_ref, kernel_test_2, rtol=5e-5)


@pytest.mark.slow
def test_sim_data_g_method_vdu():
    """Run GLE simulation, compute corrs from traj and extract kernel.
    Compare to the analytical reference kernel of the Markovian embedding.
    """
    test_data = utils.simulate_test_data()
    kernel = kernel_extraction.gamma_method(
        test_data.vv_corr,
        test_data.va_corr,
        test_data.aa_corr,
        test_data.adu_corr,
        test_data.dt,
        test_data.mass,
        test_data.trunc,
        "cpu",
    )
    # test
    time = np.arange(test_data.trunc) * test_data.dt
    kernel_ref = kernel_extraction.RefKernelExponential.from_gammas_ks(
        test_data.gammas, test_data.ks
    )
    # plot
    if os.environ.get("PLOT") == "true":
        kernel_obj = kernel_extraction.Kernel.from_kernel(kernel, test_data.dt)
        kernel_obj.set_params_kernel_ref(test_data.gammas, test_data.ks)
        axes = kernel_obj.plot(label="sim")
        axes[1].legend()
        axes[1].set_title(r"G method $C^{vdu}$.")
        plt.show()
    # test after plotting
    np.testing.assert_allclose(kernel, kernel_ref.get_kernel(time), rtol=5e-1, atol=1e-2)
