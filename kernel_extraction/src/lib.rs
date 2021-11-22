use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::{IntoPyArray, PyArray};
use ndarray::Dim;


fn _g_method_second_cpu(vv_corr: &[f64], xdu_corr: &[f64], dt: f64, trunc: usize) -> Vec<f64> {
    let mut kernel = vec![0.; trunc];
    for i in 1..trunc {
        let mut convolution = 0.;
        for j in 1..i {
            convolution += kernel[j] * vv_corr[i - j];
        }
        kernel[i] = 2. / vv_corr[0] * (
            (xdu_corr[i] - vv_corr[i] * xdu_corr[0] / vv_corr[0]) / dt
            - convolution
        );
    }
    kernel
}

#[pymodule]
fn kernel_extraction(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    #[pyo3(name = "g_method_second_cpu")]
    fn g_method_second_cpu<'py>(
        py: Python<'py>,
        vv_corr: Vec<f64>,
        xdu_corr: Vec<f64>,
        dt: f64,
        trunc: usize,
    ) -> &'py PyArray<f64, Dim<[usize; 1]>> {
        _g_method_second_cpu(&vv_corr, &xdu_corr, dt, trunc).into_pyarray(py)
    }
    Ok(())
}
