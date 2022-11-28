extern "C"
__global__ void kernel_step_g_method_nomass_double(
    double *conv, double *kernel, double *vv_corr_rev,
    double * xdu_corr, double dt, int i, int len
    ) {
    kernel[i] = 2. / vv_corr_rev[len - 1] * ((
            xdu_corr[i]
            - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1]
        ) / dt - conv[0]);
}

extern "C"
__global__ void kernel_step_g_method_double(
    double *conv, double *kernel, double *vv_corr_rev, double * xdu_corr,
    double mass, double dt, int i, int len
    ) {
    kernel[i] = 2. / vv_corr_rev[len - 1] * ((
        mass * (vv_corr_rev[len - 1] - vv_corr_rev[len - 1 - i]) 
        + xdu_corr[i] - xdu_corr[0]
        ) / dt - conv[0]);
}
