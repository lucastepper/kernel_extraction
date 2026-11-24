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
        const int lag = i + 1;
        const double mass_term = __dmul_rn(mass, vv_corr_rev[len - 1] - vv_corr_rev[len - 1 - lag]);
        const double driving_term = __dmul_rn(
            2.0 / dt, __dadd_rn(mass_term, xdu_corr[lag]) - xdu_corr[0]
        );
        kernel_i[i] = (driving_term - conv[0]) * (1.0 / (vv_corr_rev[len - 1] + vv_corr_rev[len - 2]));
}

extern "C"
__global__ void kernel_step_g_method_vdu_double(
    double *conv, double *kernel_i, double *vv_corr_rev, double *vdu_corr,
    double *vdu_corr_cumsum, double mass, double dt, int i, int len
    ) {
        kernel_i[i] = 1.0 / vv_corr_rev[len - 1] * (
            + 2.0 * mass / dt * (vv_corr_rev[len - 1] - vv_corr_rev[len - 1 - i])
            - vdu_corr[0] - vdu_corr[i]
            - 2.0 * vdu_corr_cumsum[i - 1]
            - 2.0 * conv[0]
        );
}

extern "C"
__global__ void kernel_step_gamma_method_double(
    double *conv, double *kernel, double *va_corr_rev, double *aa_corr,
    double *adu_corr, double vv_corr_0, double mass, double dt, int i, int len
    ) {
        if (i == 0) {
            kernel[i] = (mass * aa_corr[0] + adu_corr[0]) / vv_corr_0;
        } else {
            kernel[i] = 1.0 / (vv_corr_0 + va_corr_rev[len - 1] * dt * 0.5) * (
                mass * aa_corr[i] + adu_corr[i]
                - 0.5 * dt * va_corr_rev[len - 1 - i] * kernel[0]
                - dt * conv[0]
            );
        }
}
