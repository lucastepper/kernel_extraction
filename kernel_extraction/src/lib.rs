use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::{IntoPyArray, PyArray};
use ndarray::Dim;


fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|v| v.0 * v.1).sum()
}

fn _g_method_second_cpu(vv_corr_rev: &[f64], xdu_corr: &[f64], dt: f64, trunc: usize) -> Vec<f64> {
    let len =  vv_corr_rev.len();
    let mut convolution: f64;
    let mut kernel = vec![0.; trunc];
    for i in 1..trunc {
        convolution = dot_prod(&kernel[1..i], &vv_corr_rev[(len - i)..(len - 1)]);
        kernel[i] = 2. / vv_corr_rev[len - 1] * ((
            xdu_corr[i]
            - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1]
        ) / dt - convolution);
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
        // flipping vv_corr once and compution convolution as dot prod is faster
        let vv_corr_rev: Vec<f64> = vv_corr.iter()
                                           .rev()
                                           .map(|x| *x)
                                           .collect();
        _g_method_second_cpu(&vv_corr_rev, &xdu_corr, dt, trunc).into_pyarray(py)
    }
    Ok(())
}
