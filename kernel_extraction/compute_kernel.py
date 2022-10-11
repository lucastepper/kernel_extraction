import numpy as np
from numba import njit


@njit
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
