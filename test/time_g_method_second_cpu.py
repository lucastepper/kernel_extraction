from time import time
import os
import numpy as np
import kernel_extraction
import matplotlib.pyplot as plt


def main():
    dt = 0.001
    trunc_max = int(3e6)
    # set up data for benchmarks
    kernel_ref = np.load("kernel_ref.npy")
    xdu_corr_ref = np.load("xdu_corr_ref.npy")
    vv_corr_ref = np.load("vv_corr_ref.npy")
    truncs = []
    times = []
    truncs_to_run = [1, 2, 3, 4, 5]
    if os.environ.get("RUNLONG") == "true":
        truncs_to_run.extend([np.log10(1e6), np.log10(trunc_max)])
    # run benchmarks
    for trunc in truncs_to_run:
        trunc = int(10 ** trunc)
        t1 = time()
        # pad kernel if two few ref values
        if trunc > len(xdu_corr_ref):
            kernel_i_test = kernel_extraction.g_method_second_cpu(
                np.concatenate(
                    [vv_corr_ref, np.random.normal(scale=1e-7, size=(trunc - len(vv_corr_ref)))]
                ),
                np.concatenate(
                    [xdu_corr_ref, np.random.normal(scale=1e-7, size=(trunc - len(xdu_corr_ref)))]
                ),
                dt,
                trunc,
            )
        else:
            kernel_i_test = kernel_extraction.g_method_second_cpu(
                vv_corr_ref, xdu_corr_ref, dt, trunc,
            )
        time_run = time() - t1
        times.append(time_run)
        print(f"{trunc=}, {time_run=} s")
        truncs.append(trunc)
    # saved ref data in singles, reduce rtol, set atol so we do not get an error
    # from a slight shift in the kernels nodes.
    # dont test last element, as it comes from different numerical gradient
    kernel_test = np.gradient(kernel_i_test, dt)[:-1]
    np.testing.assert_allclose(kernel_test, kernel_ref[: len(kernel_test)], rtol=5e-5, atol=10)
    # make figure for kernel comparison and timings
    if not os.path.isdir("timings"):
        os.mkdir("timings")
    plt.plot(truncs, times, marker="x")
    plt.xlabel("trunc")
    plt.ylabel("time [s]")
    for i in range(1, 3):
        plt.annotate(str(round(times[-i], 1)), (truncs[-i], times[-i]))
    plt.savefig("timings/timings.pdf")
    plt.clf()
    plt.plot(np.arange(trunc)[:trunc] * dt, np.gradient(kernel_i_test, dt), label="test")
    plt.plot(np.arange(trunc) * dt, kernel_ref[:trunc], label="ref", linestyle="--")
    plt.semilogx()
    plt.legend()
    plt.savefig("kernel_compare.pdf")


if __name__ == "__main__":
    main()
