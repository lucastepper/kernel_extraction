#[macro_use]
extern crate rustacuda;
mod kernel_cuda;
use ndarray::{s, Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1};
use packed_simd::f64x4;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

enum Device {
    Cpu,
    Cuda,
}
impl Device {
    fn from_str(s: &str) -> PyResult<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Device::Cpu),
            "cuda" => Ok(Device::Cuda),
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Device: {} not supported",
                    s
                )))
            }
        }
    }
}

/// Dot product of two vectors
pub fn dot_prod_par_simd(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    // code for avx_256 instructions, should probs require avx2
    let max_len = a.len() / 4 * 4;
    let remainder = a.len() % 4;
    // iterate trough elements at a stride of 4
    let mut dot_prod_simd = a[0..max_len]
        .par_chunks_exact(4)
        .map(f64x4::from_slice_unaligned)
        .zip(
            b[0..max_len]
                .par_chunks_exact(4)
                .map(f64x4::from_slice_unaligned),
        )
        .map(|(a, b)| a * b)
        .sum::<f64x4>()
        .sum();
    // add the last elems, up to three, if len(a) / 4 != 0
    for i in max_len..(max_len + remainder) {
        dot_prod_simd += a[i] * b[i];
    }
    dot_prod_simd
}

#[pyfunction]
#[pyo3(text_signature = "(vv_corr, xdu_corr, dt, trunc, device)")]
fn g_method_xdu_nomass<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    xdu_corr: &'py PyArray1<f64>,
    dt: f64,
    trunc: usize,
    device: String,
) -> PyResult<&'py PyArray1<f64>> {
    // flipping vv_corr once and compution convolution as dot prod is faster
    let vv_corr = unsafe { vv_corr.as_array() };
    let vv_corr_rev = Array1::from_iter(vv_corr.iter().rev().map(|x| *x));
    let xdu_corr = unsafe { xdu_corr.as_array() };
    // do actual computation
    let len = vv_corr_rev.shape()[0];
    let mut convolution: f64;
    let kernel_int = match Device::from_str(&device)? {
        Device::Cpu => {
            let mut kernel_int = Array1::zeros(trunc);
            for i in 1..trunc {
                convolution = dot_prod_par_simd(
                    kernel_int.slice(s![1..i]),
                    vv_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] = 2. / vv_corr_rev[len - 1]
                    * ((xdu_corr[i]
                        - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1])
                        / dt
                        - convolution);
            }
            kernel_int
        }
        Device::Cuda => kernel_cuda::g_method_xdu_nomass(vv_corr_rev.view(), xdu_corr, dt, trunc)?,
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(text_signature = "(vv_corr, xdu_corr, dt, mass, trunc, device)")]
fn g_method_xdu<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    xdu_corr: &'py PyArray1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
    device: String,
) -> PyResult<&'py PyArray1<f64>> {
    // flipping vv_corr once and compution convolution as dot prod is faster
    let vv_corr = unsafe { vv_corr.as_array() };
    let vv_corr_rev = Array1::from_iter(vv_corr.iter().rev().map(|x| *x));
    let xdu_corr = unsafe { xdu_corr.as_array() };
    // do actual computation
    let len = vv_corr_rev.shape()[0];
    let kernel_int = match Device::from_str(&device)? {
        Device::Cpu => {
            let mut kernel_int = Array1::zeros(trunc);
            let prefac = 2.0 / (dt * vv_corr_rev[len - 1]);
            for i in 1..trunc {
                let conv = dot_prod_par_simd(
                    kernel_int.slice(s![1..i]),
                    vv_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] = prefac
                    * (mass * (vv_corr[0] - vv_corr[i]) + xdu_corr[i] - xdu_corr[0] - dt * conv)
            }
            kernel_int
        }
        Device::Cuda => kernel_cuda::g_method_xdu(vv_corr_rev.view(), xdu_corr, dt, mass, trunc)?,
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pymodule]
fn kernel_extraction(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(g_method_xdu_nomass))?;
    m.add_wrapped(wrap_pyfunction!(g_method_xdu))?;
    Ok(())
}
