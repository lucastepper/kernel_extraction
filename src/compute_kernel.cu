extern "C"
__global__ void kernel_step_g_method_xdu_nomass_double(
    double *conv, double *kernel_i, double *vv_corr_rev,
    double *xdu_corr, double dt, int i, int len
    ) {
    kernel_i[i] = 2. / vv_corr_rev[len - 1] * ((
            xdu_corr[i]
            - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1]
        ) / dt - conv[0]);
}

extern "C"
__global__ void kernel_step_g_method_xdu_double(
    double *conv, double *kernel_i, double *vv_corr_rev, double *xdu_corr,
    double mass, double dt, int i, int len
    ) {
    kernel_i[i] = 2. / vv_corr_rev[len - 1] * ((
        mass * (vv_corr_rev[len - 1] - vv_corr_rev[len - 1 - i])
        + xdu_corr[i] - xdu_corr[0]
        ) / dt - conv[0]);
}

extern "C"
__global__ void kernel_step_g_method_xdu_halfstep_double(
    double *conv, double *kernel_i, double *vv_corr_rev, double *xdu_corr,
    double mass, double dt, int i, int len
    ) {
        kernel_i[i] = 1.0 / (vv_corr_rev[len - 1] + vv_corr_rev[len - 2]) * (
            2.0 / dt * (
                mass * (vv_corr_rev[len - 1] - vv_corr_rev[len - 1 - i]) + xdu_corr[i] - xdu_corr[0]
            ) - conv[0]
        );
}

extern "C"
__global__ void kernel_step_g_method_vdu_double(
    double *conv, double *kernel_i, double *vv_corr_rev, double *vdu_corr,
    double *vdu_corr_cumsum, double mass, double dt, int i, int len
    ) {
        kernel_i[i] = 1.0 / vv_corr_rev[len - 1] * (
            + 2.0 * mass / dt * (vv_corr_rev[len - 1] - vv_corr_rev[len - 1 - i])
            + vdu_corr[0] + vdu_corr[i]
            - 2.0 * vdu_corr_cumsum[i - 1]
            - 2.0 * conv[0]
        );
}
