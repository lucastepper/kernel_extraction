import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction


def g_method_xdu_halfstep(vv_corr, xdu_corr, dt, mass, trunc, loop=True):
    """Gmethod with external potential, evaluating kernel at half steps."""
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_g_method_xdu_halfstep(device):
    """Test extraction method versus reference implementation."""
    test_data = utils.load_test_data()
    kernel_i_ref = g_method_xdu_halfstep(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, test_data.mass, test_data.trunc
    )
    kernel_i_test = kernel_extraction.g_method_xdu_halfstep(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, test_data.mass, test_data.trunc, device
    )
    # Saved the data in singles, reduce rtol
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test, rtol=5e-5)


@pytest.mark.slow
def test_sim_data_g_method_xdu_halfstep():
    """Run GLE simulation, compute corrs from traj and extract kernel.
    Compare to the analytical reference kernel of the Markovian embedding.
    """
    test_data = utils.simulate_test_data()
    kernel_i = g_method_xdu_halfstep(
        test_data.vv_corr, test_data.xdu_corr, test_data.dt, test_data.mass, test_data.trunc
    )
    kernel_obj = kernel_extraction.Kernel.from_kernel_int(kernel_i, test_data.dt, halfstep=True)
    # test
    time = np.arange(test_data.trunc) * test_data.dt
    kernel_ref = kernel_extraction.RefKernelExponential.from_gammas_ks(
        test_data.gammas, test_data.ks
    )
    np.testing.assert_allclose(kernel_obj.kernel, kernel_ref.get_kernel(time), rtol=0.5, atol=1e-2)
    # plot
    if os.environ.get("PLOT") == "true":
        kernel_obj.set_params_kernel_ref(test_data.gammas, test_data.ks)
        axes = kernel_obj.plot(label="sim")
        axes[1].legend()
        axes[1].set_title(r"G method $C^{xdu}$ halfstep.")
        plt.show()
