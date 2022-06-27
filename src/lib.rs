#[macro_use]
extern crate rustacuda;
mod kernel_cpu;
mod kernel_cuda;
use ndarray as nd;
use numpy as np;
use numpy::IntoPyArray;
use pyo3::prelude::*;

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
    ) -> &'py np::PyArray1<f64> {
        // flipping vv_corr once and compution convolution as dot prod is faster
        let vv_corr_rev: Vec<f64> = vv_corr.iter().rev().map(|x| *x).collect();
        kernel_cpu::_g_method_second_cpu(&vv_corr_rev, &xdu_corr, dt, trunc).into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "g_method_second_cuda")]
    fn g_method_second_cuda<'py>(
        py: Python<'py>,
        vv_corr: &'py np::PyArray1<f64>,
        xdu_corr: &'py np::PyArray1<f64>,
        dt: f64,
        trunc: usize,
    ) -> &'py np::PyArray1<f64> {
        // flipping vv_corr once and compution convolution as dot prod is faster
        let vv_corr = unsafe { vv_corr.as_array() };
        let vv_corr_rev: Vec<f64> = vv_corr.iter().rev().map(|x| *x).collect();
        let vv_corr_rev = nd::Array1::from_vec(vv_corr_rev);
        let xdu_corr = unsafe { xdu_corr.as_array() };
        kernel_cuda::compute_kernel(vv_corr_rev.view(), xdu_corr, dt, trunc).into_pyarray(py)
    }
    Ok(())
}
