#[macro_use]
#[cfg(feature = "cuda")]
extern crate rustacuda;
#[cfg(feature = "cuda")]
mod kernel_cuda;
use ndarray::{s, Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1};
use packed_simd::f64x4;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

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

    #[cfg(not(feature = "cuda"))]
    fn raise_cuda_error<'py>() -> PyResult<&'py PyArray1<f64>> {
        return Err(PyValueError::new_err(
            "Cuda not supported, please compile with feature cuda",
        ));
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
    // check that corrs are long enough or lower trunc
    let trunc = vv_corr.len().min(xdu_corr.len()).min(trunc);
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
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (kernel_cuda::XDU_CORR, xdu_corr.view()),
            ]);
            let mut kernel_cuda = kernel_cuda::KernelCuda::new(corrs, dt, None, trunc)?;
            kernel_cuda.g_method_xdu_nomass()?;
            kernel_cuda.get_kernel()?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => return Device::raise_cuda_error(),
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
    // check that corrs are long enough or lower trunc
    let trunc = vv_corr.len().min(xdu_corr.len()).min(trunc);
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

        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (kernel_cuda::XDU_CORR, xdu_corr.view()),
            ]);
            let mut kernel_cuda = kernel_cuda::KernelCuda::new(corrs, dt, Some(mass), trunc)?;
            kernel_cuda.g_method_xdu()?;
            kernel_cuda.get_kernel()?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => return Device::raise_cuda_error(),
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(text_signature = "(vv_corr, xdu_corr, dt, mass, trunc, device)")]
fn g_method_xdu_halfstep<'py>(
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
    // check that corrs are long enough or lower trunc
    let trunc = (vv_corr.len() - 1).min(xdu_corr.len() - 1).min(trunc);
    // for the cpu version of the half step, we needan array containting the
    // sum of reversed(vv_corr[i] and vv_corr[i + i]) for the convolution
    let vv_corr_plus_next_rev = &vv_corr_rev.slice(s![1..]) + &vv_corr_rev.slice(s![..-1]);
    let xdu_corr = unsafe { xdu_corr.as_array() };
    // do actual computation
    let len = vv_corr_plus_next_rev.shape()[0];
    let kernel_int = match Device::from_str(&device)? {
        Device::Cpu => {
            let mut kernel_int = Array1::zeros(trunc);
            let prefac = 1.0 / (vv_corr[0] + vv_corr[1]);
            let prefac_vv_corr = 2.0 / dt;
            for i in 1..trunc {
                kernel_int[i] =
                    prefac_vv_corr * (mass * (vv_corr[0] - vv_corr[i]) + xdu_corr[i] - xdu_corr[0]);
                let conv = dot_prod_par_simd(
                    kernel_int.slice(s![1..i]),
                    vv_corr_plus_next_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] -= conv;
                kernel_int[i] *= prefac;
            }
            kernel_int
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (
                    kernel_cuda::VV_CORR_REV_PLUS_NEXT,
                    vv_corr_plus_next_rev.view(),
                ),
                (kernel_cuda::XDU_CORR, xdu_corr.view()),
            ]);
            let mut kernel_cuda = kernel_cuda::KernelCuda::new(corrs, dt, Some(mass), trunc)?;
            kernel_cuda.g_method_xdu_halfstep()?;
            kernel_cuda.get_kernel()?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => return Device::raise_cuda_error(),
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(text_signature = "(vv_corr, vdu_corr, dt, mass, trunc, device)")]
fn g_method_vdu<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    vdu_corr: &'py PyArray1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
    device: String,
) -> PyResult<&'py PyArray1<f64>> {
    // flipping vv_corr once and compution convolution as dot prod is faster
    let vv_corr = unsafe { vv_corr.as_array() };
    let vv_corr_rev = Array1::from_iter(vv_corr.iter().rev().map(|x| *x));
    let vdu_corr = unsafe { vdu_corr.as_array() };
    let len = vv_corr_rev.shape()[0];
    // check that corrs are long enough or lower trunc
    let trunc = vv_corr.len().min(vdu_corr.len()).min(trunc);
    // to streamline the computation of all terms for the trapezodial rule
    // of the integral over vdu_corr, we precompute the cum sum of vdu_corr
    // where we set the first element to zero
    let mut vdu_corr_cumsum: Array1<f64> = Array1::zeros(trunc);
    for i in 1..trunc {
        vdu_corr_cumsum[i] = vdu_corr_cumsum[i - 1] + vdu_corr[i];
    }
    // do actual computation
    let kernel_int = match Device::from_str(&device)? {
        Device::Cpu => {
            let mut kernel_int = Array1::zeros(trunc);
            let prefac = 1.0 / vv_corr[0];
            for i in 1..trunc {
                let conv = dot_prod_par_simd(
                    kernel_int.slice(s![1..i]),
                    vv_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] = prefac
                    * (2. * mass / dt * (vv_corr[0] - vv_corr[i]) + vdu_corr[0] + vdu_corr[i]
                        - 2. * conv
                        - 2. * vdu_corr_cumsum[i - 1]);
            }
            kernel_int
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (kernel_cuda::VDU_CORR, vdu_corr.view()),
                (kernel_cuda::VDU_CORR_CUMSUM, vdu_corr_cumsum.view()),
            ]);
            let mut kernel_cuda = kernel_cuda::KernelCuda::new(corrs, dt, Some(mass), trunc)?;
            kernel_cuda.g_method_vdu()?;
            kernel_cuda.get_kernel()?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => return Device::raise_cuda_error(),
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(text_signature = "(vv_corr, va_corr, aa_corr, adu_corr, dt, mass, trunc, device)")]
fn gamma_method<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    va_corr: &'py PyArray1<f64>,
    aa_corr: &'py PyArray1<f64>,
    adu_corr: &'py PyArray1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
    device: String,
) -> PyResult<&'py PyArray1<f64>> {
    // flipping vv_corr once and compution convolution as dot prod is faster
    let vv_corr = unsafe { vv_corr.as_array() };
    let va_corr = unsafe { va_corr.as_array() };
    let va_corr_rev = Array1::from_iter(va_corr.iter().rev().map(|x| *x));
    let aa_corr = unsafe { aa_corr.as_array() };
    let adu_corr = unsafe { adu_corr.as_array() };
    let len = va_corr_rev.shape()[0];
    // check that corrs are long enough or lower trunc
    let trunc = vv_corr
        .len()
        .min(va_corr.len())
        .min(aa_corr.len())
        .min(adu_corr.len())
        .min(trunc);
    // to streamline the computation of all terms for the trapezodial rule
    // of the integral over vdu_corr, we precompute the cum sum of vdu_corr
    // where we set the first element to zero
    let kernel_int = match Device::from_str(&device)? {
        Device::Cpu => {
            let mut kernel = Array1::zeros(trunc);
            let prefac = 1.0 / (vv_corr[0] + va_corr[0] * dt * 0.5);
            kernel[0] = (mass * aa_corr[0] + adu_corr[0]) / vv_corr[0];
            for i in 1..trunc {
                let conv = dot_prod_par_simd(
                    kernel.slice(s![1..i]),
                    va_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel[i] = prefac
                    * (mass * aa_corr[i] + adu_corr[i]
                        - 0.5 * dt * va_corr[i] * kernel[0]
                        - dt * conv);
            }
            kernel
        }
        #[cfg(feature = "cuda")]
        Device::Cuda => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VA_CORR_REV, va_corr_rev.view()),
                (kernel_cuda::AA_CORR, aa_corr),
                (kernel_cuda::ADU_CORR, adu_corr),
            ]);
            let mut kernel_cuda = kernel_cuda::KernelCuda::new(corrs, dt, Some(mass), trunc)?;
            kernel_cuda.set_vv_corr_0(vv_corr[0]);
            kernel_cuda.gamma_method()?;
            kernel_cuda.get_kernel()?
        }
        #[cfg(not(feature = "cuda"))]
        Device::Cuda => return Device::raise_cuda_error(),
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pymodule]
fn kernel_extraction(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(g_method_xdu_nomass))?;
    m.add_wrapped(wrap_pyfunction!(g_method_xdu))?;
    m.add_wrapped(wrap_pyfunction!(g_method_xdu_halfstep))?;
    m.add_wrapped(wrap_pyfunction!(g_method_vdu))?;
    m.add_wrapped(wrap_pyfunction!(gamma_method))?;
    Ok(())
}
