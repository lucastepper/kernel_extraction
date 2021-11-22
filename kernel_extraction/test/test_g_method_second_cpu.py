import numpy as np
import kernel_extraction


def _calc_G_method(kernel_i, xu_cf, v_acf, dt, trunc, verbose=False):
    """Compute the integral over the kernel."""

    prefac = 2.0 / v_acf[0]
    for i in range(1, trunc):
        kernel_i[i] = prefac * (
            (xu_cf[i] - v_acf[i] * xu_cf[0] / v_acf[0]) / dt
            - np.sum(kernel_i[1:i] * v_acf[1:i][::-1])
        )
        if verbose and i % 10000 == 0:
            print("progress: ", round(i / trunc * 100, 3), "%")


def main():
    dt = 0.01
    trunc = 10
    vv_corr = np.load("test/vv_corr_ref.npy")[:10]
    xdu_corr = np.load("test/xdu_corr_ref.npy")[:10]
    kernel_i_ref = np.zeros(trunc)
    _calc_G_method(kernel_i_ref, xdu_corr, vv_corr, dt, trunc)
    kernel_i_test = kernel_extraction.g_method_second_cpu(vv_corr, xdu_corr, dt, trunc)
    np.testing.assert_allclose(kernel_i_ref, kernel_i_test)
    print("Test successful!")


if __name__ == "__main__":
    main()
