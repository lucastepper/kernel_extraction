import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction


def _calc_G_method(vv_corr, xdu_corr, dt, trunc):
    """Compute the integral over the kernel."""
    kernel_i = np.zeros(trunc)
    prefac = 2.0 / vv_corr[0]
    for i in range(1, trunc):
        kernel_i[i] = prefac * (
            (xdu_corr[i] - vv_corr[i] * xdu_corr[0] / vv_corr[0]) / dt
            - np.sum(kernel_i[1:i] * vv_corr[1:i][::-1])
        )
    return kernel_i


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_g_method_cpu(device):
    """Test extraction method versus reference implementation."""
    test_data = utils.load_test_data()
    kernel_i_ref = _calc_G_method(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, test_data.trunc
    )
    kernel_i_test = kernel_extraction.g_method_xdu_nomass(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, test_data.trunc, device
    )
    # Saved the data in sinlges, reduce rtol
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test, rtol=5e-5)


@pytest.mark.slow
def test_sim_data():
    """Run GLE simulation, compute corrs from traj and extract kernel.
    Compare to the analytical reference kernel of the Markovian embedding.
    """
    test_data = utils.simulate_test_data()
    kernel_i = kernel_extraction.g_method_xdu_nomass(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, test_data.trunc, "cpu"
    )
    # test
    time = np.arange(test_data.trunc) * test_data.dt
    kernel_ref = kernel_extraction.RefKernelExponential.from_gammas_ks(
        test_data.gammas, test_data.ks
    )
    np.testing.assert_allclose(kernel_i, kernel_ref.get_kernel_int(time), rtol=5e-2, atol=1e-2)
    # plot
    if os.environ.get("PLOT") == "true":
        kernel = kernel_extraction.Kernel.from_kernel_int(kernel_i, test_data.dt)
        kernel.set_params_kernel_ref(test_data.gammas, test_data.ks)
        axes = kernel.plot(label="sim")
        axes[1].legend()
        plt.show()
