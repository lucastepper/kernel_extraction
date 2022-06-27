// use crate::something;
mod kernel_cpu;
use ndarray::Dim;
use numpy::{IntoPyArray, PyArray};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};


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
        let vv_corr_rev: Vec<f64> = vv_corr.iter().rev().map(|x| *x).collect();
        kernel_cpu::_g_method_second_cpu(&vv_corr_rev, &xdu_corr, dt, trunc).into_pyarray(py)
    }
    Ok(())
}
