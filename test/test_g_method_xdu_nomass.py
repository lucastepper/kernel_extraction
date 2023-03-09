import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction

SKIPCUDA = os.environ.get("SKIPCUDA", default="") == "true"


def g_method_nomass(vv_corr, xdu_corr, dt, trunc):
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_g_method_nomass(device):
    """Test extraction method versus reference implementation."""
    if device == "cuda" and SKIPCUDA:
        pytest.skip("Skipping CUDA test")
    test_data = utils.load_test_data()
    # test that the length of the correlation functions is taken into account
    trunc = 2 * len(test_data.vv_corr)
    kernel_i_ref = g_method_nomass(test_data.vv_corr, test_data.xdu_corr, test_data.dt, trunc)
    kernel_i_test_1 = kernel_extraction.g_method_xdu_nomass(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, trunc, device
    )
    kernel_i_test_2 = kernel_extraction.compute_kernel(
        kernel_extraction.ExtractionMethod.G_METHOD_XDU_NOMASS,
        test_data.dt,
        trunc=trunc,
        device=device,
        vv_corr=test_data.vv_corr,
        xdu_corr=test_data.xdu_corr,
    )
    # Saved the data in singles, reduce rtol
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_1, rtol=5e-5)
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test_2, rtol=5e-5)


@pytest.mark.slow
def test_sim_data_method_nomass():
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
        axes[1].set_title(r"G method $C^{xdu}$ no mass.")
        plt.show()
