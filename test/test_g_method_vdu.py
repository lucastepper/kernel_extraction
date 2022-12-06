import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction


def g_method_vdu(vv_corr, vdu_corr, dt, mass, trunc, loop=False):
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
    """
    trunc = np.min([len(vv_corr), len(vdu_corr), trunc])
    kernel_int = np.zeros(trunc)
    # to streamline the computation of all terms for the trapezodial rule
    # of the integral over vdu_corr, we precompute the cum sum of vdu_corr
    # where we set the first element to zero
    vdu_corr_cumsum = np.cumsum(np.concatenate([[0.0], vdu_corr[1:trunc]]))
    prefac = 1 / vv_corr[0]
    for i in range(1, len(kernel_int)):
        kernel_int[i] = vdu_corr[0] + vdu_corr[i] + (2 * mass / dt) * (vv_corr[0] - vv_corr[i])
        conv = 0.0
        for j in range(1, i):
            # conv += kernel_int[j] * vv_corr[i - j] + vdu_corr[j]
            conv += kernel_int[j] * vv_corr[i - j]
        kernel_int[i] -= 2 * vdu_corr_cumsum[i - 1]
        kernel_int[i] -= 2 * conv
        kernel_int[i] *= prefac
    return kernel_int


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_g_method_vdu(device):
    """Test extraction method versus reference implementation."""
    test_data = utils.load_test_data()
    # test that the length of the correlation functions is taken into account
    trunc = 2 * len(test_data.vv_corr)
    kernel_i_ref = g_method_vdu(
        test_data.vv_corr, test_data.vdu_corr, test_data.dt, test_data.mass, trunc
    )
    kernel_i_test_1 = kernel_extraction.g_method_vdu(
        test_data.vv_corr, test_data.vdu_corr, test_data.dt, test_data.mass, trunc, device
    )
    kernel_i_test_2 = kernel_extraction.compute_kernel(
        kernel_extraction.ExtractionMethod.G_METHOD_VDU,
        test_data.dt,
        mass=test_data.mass,
        trunc = trunc,
        device=device,
        vv_corr=test_data.vv_corr,
        vdu_corr=test_data.vdu_corr,
    )
    # Saved the data in singles, reduce rtol
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_1, rtol=5e-5)
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_2, rtol=5e-5)


@pytest.mark.slow
def test_sim_data_g_method_vdu():
    """Run GLE simulation, compute corrs from traj and extract kernel.
    Compare to the analytical reference kernel of the Markovian embedding.
    """
    test_data = utils.simulate_test_data()
    kernel_i = kernel_extraction.g_method_vdu(
        test_data.vv_corr, test_data.vdu_corr, test_data.dt, test_data.mass, test_data.trunc, "cpu"
    )
    # test
    time = np.arange(test_data.trunc) * test_data.dt
    kernel_ref = kernel_extraction.RefKernelExponential.from_gammas_ks(
        test_data.gammas, test_data.ks
    )
    # plot
    if os.environ.get("PLOT") == "true":
        kernel = kernel_extraction.Kernel.from_kernel_int(kernel_i, test_data.dt)
        kernel.set_params_kernel_ref(test_data.gammas, test_data.ks)
        axes = kernel.plot(label="sim")
        axes[1].legend()
        axes[1].set_title(r"G method $C^{vdu}$.")
        plt.show()
    # test after plotting
    np.testing.assert_allclose(kernel_i, kernel_ref.get_kernel_int(time), rtol=5e-1, atol=1e-2)
