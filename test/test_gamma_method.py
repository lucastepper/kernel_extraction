import os
import numpy as np
import matplotlib.pyplot as plt
import pytest
import utils
import kernel_extraction


def gamma_method(cvv, cva, caf, cadu, dt, trunc):
    """ This is an alternatice volterra method extracting the memory kernel
    $\Gamma(t)$ directly. It is based on a volterra method of the second kind.
    Arguments:
        cvv (np.ndarray): velocity autocorrelation function
        cva (np.ndarray): velocity acceleration correlation function
        caf (np.ndarray): acceleration force correlation function (caf = mass * caa)
        cadu (np.ndarray): acceleration gradient of PMF correlation function
        dt (float): time step
        trunc (int): number of frames of G(t) to compute
    """
    kernel = np.zeros(trunc)
    prefac = 1. / (cvv[0] + cva[0] * dt * 0.5)
    kernel[0] = (caf[0] + cadu[0]) / cvv[0]
    for i in range(1, trunc):
        num = caf[i] + cadu[i]
        # do first instance of loop where j = 0 and we scale
        # with 0.5 due to trapezoidal rule
        num -= 0.5 * dt * cva[i] * kernel[0]
        for j in range(1, i):
            num -= dt * cva[i - j] * kernel[j]
        kernel[i] = prefac * num
    return kernel


# @pytest.mark.parametrize("device", ["cpu", "cuda"])
# def test_g_method_vdu(device):
#     """Test extraction method versus reference implementation."""
#     test_data = utils.load_test_data()
#     # test that the length of the correlation functions is taken into account
#     trunc = 2 * len(test_data.vv_corr)
#     kernel_i_ref = g_method_vdu(
#         test_data.vv_corr, test_data.vdu_corr, test_data.dt, test_data.mass, trunc
#     )
#     kernel_i_test = kernel_extraction.g_method_vdu(
#         test_data.vv_corr, test_data.vdu_corr, test_data.dt, test_data.mass, trunc, device
#     )
#     # Saved the data in singles, reduce rtol
#     np.testing.assert_allclose(kernel_i_ref, kernel_i_test, rtol=5e-5)


# @pytest.mark.slow
# def test_sim_data_g_method_vdu():
#     """Run GLE simulation, compute corrs from traj and extract kernel.
#     Compare to the analytical reference kernel of the Markovian embedding.
#     """
#     test_data = utils.simulate_test_data()
#     kernel_i = kernel_extraction.g_method_vdu(
#         test_data.vv_corr, test_data.vdu_corr, test_data.dt, test_data.mass, test_data.trunc, "cpu"
#     )
#     # test
#     time = np.arange(test_data.trunc) * test_data.dt
#     kernel_ref = kernel_extraction.RefKernelExponential.from_gammas_ks(
#         test_data.gammas, test_data.ks
#     )
#     # plot
#     if os.environ.get("PLOT") == "true":
#         kernel = kernel_extraction.Kernel.from_kernel_int(kernel_i, test_data.dt)
#         kernel.set_params_kernel_ref(test_data.gammas, test_data.ks)
#         axes = kernel.plot(label="sim")
#         axes[1].legend()
#         axes[1].set_title(r"G method $C^{vdu}$.")
#         plt.show()
#     # test after plotting
#     np.testing.assert_allclose(kernel_i, kernel_ref.get_kernel_int(time), rtol=5e-1, atol=1e-2)
