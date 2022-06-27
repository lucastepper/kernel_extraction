extern "C"
__global__ void kernel_step_double(double *dot, double *kernel, double *vv_corr_rev, double * xdu_corr, double dt, int i, int len) {
    kernel[i] = 2. / vv_corr_rev[len - 1] * ((
            xdu_corr[i]
            - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1]
        ) / dt - dot[0]);
}
