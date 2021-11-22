from time import time
import numpy as np
import kernel_extraction
import matplotlib.pyplot as plt


def main():
    dt = 0.001
    trunc_max = int(1e5)
    kernel_ref = np.load("test/kernel_ref.npy")
    xdu_corr_ref = np.load("test/xdu_corr_ref.npy")
    vv_corr_ref = np.load("test/vv_corr_ref.npy")

    truncs = []
    times = []
    for trunc in [1, 2, 3, 4, np.log10(trunc_max)]:
        trunc = int(10 ** trunc)
        t1 = time()
        kernel_i_test = kernel_extraction.g_method_second_cpu(
            vv_corr_ref, xdu_corr_ref, dt, trunc
        )
        time_run = time() - t1
        times.append(time_run)
        print(f"{trunc=}, {time_run=} s")
        truncs.append(trunc)

    plt.plot(truncs, times, marker="x")
    plt.xlabel("trunc")
    plt.ylabel("time [s]")
    for i in range(1, 3):
        plt.annotate(str(round(times[-i], 1)), (truncs[-i], times[-i]))
    plt.savefig("timings.pdf")
    plt.clf()

    plt.plot(
        np.arange(trunc)[:trunc] * dt, np.gradient(kernel_i_test, dt), label="test"
    )
    plt.plot(np.arange(trunc) * dt, kernel_ref[:trunc], label="ref", alpha=0.6)
    plt.semilogx()
    plt.legend()
    plt.savefig("kernel_compare.pdf")


if __name__ == "__main__":
    main()
