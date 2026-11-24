#[cfg(feature = "cuda")]
mod kernel_cuda;
use ndarray::{s, Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
#[cfg(feature = "cuda")]
use std::collections::HashMap;

enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(kernel_cuda::CudaBackend),
}

impl Device {
    fn resolve(device: &str, _method: &str) -> PyResult<Self> {
        match device.to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            #[cfg(feature = "cuda")]
            "cuda" => kernel_cuda::CudaBackend::new(_method).map(Self::Cuda),
            #[cfg(feature = "cuda")]
            "auto" => Ok(kernel_cuda::CudaBackend::new(_method)
                .map(Self::Cuda)
                .unwrap_or(Self::Cpu)),
            #[cfg(not(feature = "cuda"))]
            "auto" => Ok(Self::Cpu),
            #[cfg(not(feature = "cuda"))]
            "cuda" => Err(pyo3::exceptions::PyRuntimeError::new_err(
                "CUDA support was disabled when building this extension",
            )),
            _ => Err(PyValueError::new_err(format!(
                "Invalid device {device:?}. Expected 'cpu', 'cuda', or 'auto'."
            ))),
        }
    }
}

/// Dot product of two vectors
pub fn dot_prod_par(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let a = a.as_slice().unwrap();
    let b = b.as_slice().unwrap();
    a.par_iter().zip(b.par_iter()).map(|(a, b)| a * b).sum()
}

// Check if the kernel is already long enough, if not, pad with zeros
// Also return the start index for the kernel computation
pub fn get_kernel_array_start(
    kernel: Option<&PyArray1<f64>>,
    trunc: usize,
    first_sample: usize,
) -> (usize, Array1<f64>) {
    match kernel {
        Some(kernel) => {
            let kernel = unsafe { kernel.as_array() };
            if kernel.len() < trunc {
                let mut kernel_new = Array1::zeros(trunc);
                kernel_new.slice_mut(s![..kernel.len()]).assign(&kernel);
                (kernel.len().max(first_sample), kernel_new)
            } else {
                (kernel.len(), kernel.to_owned())
            }
        }
        None => (first_sample, Array1::zeros(trunc)),
    }
}
pub struct Pbar {
    bar: Option<indicatif::ProgressBar>,
}

impl Pbar {
    pub fn new(
        verbose: bool,
        message: &'static str,
        prefix: &'static str,
        mut len_ker: usize,
        start_val: usize,
    ) -> Self {
        if verbose == false {
            return Pbar { bar: None };
        }
        if len_ker > 1 {
            len_ker -= 1;
        };
        let bar = indicatif::ProgressBar::new(len_ker as u64);
        bar.inc(start_val as u64);
        bar.set_message(message);
        bar.set_prefix(prefix);
        Pbar { bar: Some(bar) }
    }
    pub fn increment(&mut self) {
        if let Some(bar) = &self.bar {
            bar.inc(1);
        }
    }
    pub fn finish(&mut self) {
        if let Some(bar) = &self.bar {
            bar.finish();
        }
    }
}

#[pyfunction]
#[pyo3(
    text_signature = "(vv_corr, xdu_corr, dt, trunc, device, verbose, kernel_int: Optional[ndarray])"
)]
fn g_method_xdu_nomass<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    xdu_corr: &'py PyArray1<f64>,
    dt: f64,
    trunc: usize,
    device: String,
    verbose: bool,
    kernel_int: Option<&'py PyArray1<f64>>,
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
    let kernel_int = match Device::resolve(&device, "g_method_xdu_nomass")? {
        Device::Cpu => {
            let (start, mut kernel_int) = get_kernel_array_start(kernel_int, trunc, 1);
            let mut bar = Pbar::new(
                verbose,
                "Computing kernel g_method_xdu_nomass",
                "G_method Cxdu nomass",
                trunc,
                start,
            );
            for i in start..trunc {
                convolution = dot_prod_par(
                    kernel_int.slice(s![1..i]),
                    vv_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] = 2. / vv_corr_rev[len - 1]
                    * ((xdu_corr[i]
                        - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1])
                        / dt
                        - convolution);
                bar.increment();
            }
            bar.finish();
            kernel_int
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(backend) => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (kernel_cuda::XDU_CORR, xdu_corr.view()),
            ]);
            let mut kernel_cuda =
                kernel_cuda::KernelCuda::new(backend, corrs, kernel_int, dt, trunc, 1, verbose)?;
            kernel_cuda.g_method_xdu_nomass()?;
            kernel_cuda.get_kernel()?
        }
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(
    text_signature = "(vv_corr, xdu_corr, dt, mass, trunc, device, verbose, kernel_int: Optional[ndarray])"
)]
fn g_method_xdu<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    xdu_corr: &'py PyArray1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
    device: String,
    verbose: bool,
    kernel_int: Option<&'py PyArray1<f64>>,
) -> PyResult<&'py PyArray1<f64>> {
    // flipping vv_corr once and compution convolution as dot prod is faster
    let vv_corr = unsafe { vv_corr.as_array() };
    let vv_corr_rev = Array1::from_iter(vv_corr.iter().rev().map(|x| *x));
    let xdu_corr = unsafe { xdu_corr.as_array() };
    // check that corrs are long enough or lower trunc
    let trunc = vv_corr.len().min(xdu_corr.len()).min(trunc);
    // do actual computation
    let len = vv_corr_rev.shape()[0];
    let kernel_int = match Device::resolve(&device, "g_method_xdu")? {
        Device::Cpu => {
            let (start, mut kernel_int) = get_kernel_array_start(kernel_int, trunc, 1);
            let mut bar = Pbar::new(
                verbose,
                "Computing kernel g_method_xdu",
                "G_method Cxdu",
                trunc,
                start,
            );
            let prefac = 2.0 / (dt * vv_corr_rev[len - 1]);
            for i in start..trunc {
                let conv = dot_prod_par(
                    kernel_int.slice(s![1..i]),
                    vv_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] = prefac
                    * (mass * (vv_corr[0] - vv_corr[i]) + xdu_corr[i] - xdu_corr[0] - dt * conv);
                bar.increment();
            }
            bar.finish();
            kernel_int
        }

        #[cfg(feature = "cuda")]
        Device::Cuda(backend) => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (kernel_cuda::XDU_CORR, xdu_corr.view()),
            ]);
            let mut kernel_cuda =
                kernel_cuda::KernelCuda::new(backend, corrs, kernel_int, dt, trunc, 1, verbose)?;
            kernel_cuda.g_method_xdu(mass, false)?;
            kernel_cuda.get_kernel()?
        }
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(
    text_signature = "(vv_corr, xdu_corr, dt, mass, trunc, device, verbose, kernel_int: Optional[ndarray])"
)]
fn g_method_xdu_halfstep<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    xdu_corr: &'py PyArray1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
    device: String,
    verbose: bool,
    kernel_int: Option<&'py PyArray1<f64>>,
) -> PyResult<&'py PyArray1<f64>> {
    let vv_corr = unsafe { vv_corr.as_array() };
    let xdu_corr = unsafe { xdu_corr.as_array() };
    let trunc = vv_corr
        .len()
        .min(xdu_corr.len())
        .saturating_sub(1)
        .min(trunc);
    let device = Device::resolve(&device, "g_method_xdu_halfstep")?;
    if trunc == 0 {
        let (_, kernel_int) = get_kernel_array_start(kernel_int, trunc, 0);
        return Ok(kernel_int.into_pyarray(py));
    }

    let kernel_int = match device {
        Device::Cpu => {
            let (start, mut kernel_int) = get_kernel_array_start(kernel_int, trunc, 0);
            let mut bar = Pbar::new(
                verbose,
                "Computing kernel g_method_xdu_halfstep",
                "G_method Cxdu halfstep",
                trunc,
                start,
            );
            let prefac = 1.0 / (vv_corr[0] + vv_corr[1]);
            let prefac_vv_corr = 2.0 / dt;
            for i in start..trunc {
                let driving_term = prefac_vv_corr
                    * (mass * (vv_corr[0] - vv_corr[i + 1]) + xdu_corr[i + 1] - xdu_corr[0]);
                let mut conv = 0.0;
                for j in 1..=i {
                    conv += kernel_int[i - j] * (vv_corr[j] + vv_corr[j + 1]);
                }
                kernel_int[i] = (driving_term - conv) * prefac;
                bar.increment();
            }
            bar.finish();
            kernel_int
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(backend) => {
            let vv_corr_rev = Array1::from_iter(vv_corr.iter().rev().copied());
            let vv_corr_plus_next_rev = &vv_corr_rev.slice(s![1..]) + &vv_corr_rev.slice(s![..-1]);
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (
                    kernel_cuda::VV_CORR_REV_PLUS_NEXT,
                    vv_corr_plus_next_rev.view(),
                ),
                (kernel_cuda::XDU_CORR, xdu_corr),
            ]);
            let mut kernel_cuda =
                kernel_cuda::KernelCuda::new(backend, corrs, kernel_int, dt, trunc, 0, verbose)?;
            kernel_cuda.g_method_xdu(mass, true)?;
            kernel_cuda.get_kernel()?
        }
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(
    text_signature = "(vv_corr, vdu_corr, dt, mass, trunc, device, verbose, kernel_int: Optional[ndarray])"
)]
fn g_method_vdu<'py>(
    py: Python<'py>,
    vv_corr: &'py PyArray1<f64>,
    vdu_corr: &'py PyArray1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
    device: String,
    verbose: bool,
    kernel_int: Option<&'py PyArray1<f64>>,
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
    let kernel_int = match Device::resolve(&device, "g_method_vdu")? {
        Device::Cpu => {
            let (start, mut kernel_int) = get_kernel_array_start(kernel_int, trunc, 1);
            let mut bar = Pbar::new(
                verbose,
                "Computing kernel g_method_vdu",
                "G_method Cvdu",
                trunc,
                start,
            );
            let prefac = 1.0 / vv_corr[0];
            for i in start..trunc {
                let conv = dot_prod_par(
                    kernel_int.slice(s![1..i]),
                    vv_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel_int[i] = prefac
                    * (2. * mass / dt * (vv_corr[0] - vv_corr[i])
                        - vdu_corr[0]
                        - vdu_corr[i]
                        - 2. * conv
                        - 2. * vdu_corr_cumsum[i - 1]);
                bar.increment();
            }
            bar.finish();
            kernel_int
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(backend) => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VV_CORR_REV, vv_corr_rev.view()),
                (kernel_cuda::VDU_CORR, vdu_corr.view()),
                (kernel_cuda::VDU_CORR_CUMSUM, vdu_corr_cumsum.view()),
            ]);
            let mut kernel_cuda =
                kernel_cuda::KernelCuda::new(backend, corrs, kernel_int, dt, trunc, 1, verbose)?;
            kernel_cuda.g_method_vdu(mass)?;
            kernel_cuda.get_kernel()?
        }
    };
    Ok(kernel_int.into_pyarray(py))
}

#[pyfunction]
#[pyo3(
    text_signature = "(vv_corr, va_corr, aa_corr, adu_corr, dt, mass, trunc, device, verbose, kernel: Optional[ndarray])"
)]
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
    verbose: bool,
    kernel: Option<&'py PyArray1<f64>>,
) -> PyResult<&'py PyArray1<f64>> {
    // flipping vv_corr once and compution convolution as dot prod is faster
    let vv_corr = unsafe { vv_corr.as_array() };
    let va_corr = unsafe { va_corr.as_array() };
    let va_corr_rev = Array1::from_iter(va_corr.iter().rev().map(|x| *x));
    let aa_corr = unsafe { aa_corr.as_array() };
    let adu_corr = unsafe { adu_corr.as_array() };
    let len = va_corr_rev.shape()[0];
    if vv_corr.is_empty() || va_corr.is_empty() || aa_corr.is_empty() || adu_corr.is_empty() {
        return Err(PyValueError::new_err(
            "Gamma method requires nonempty correlations",
        ));
    }
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
    let kernel = match Device::resolve(&device, "gamma_method")? {
        Device::Cpu => {
            let (start, mut kernel) = get_kernel_array_start(kernel, trunc, 1);
            let mut bar = Pbar::new(
                verbose,
                "Computing kernel gamma_method",
                "Gamma_method",
                trunc,
                start,
            );
            let prefac = 1.0 / (vv_corr[0] + va_corr[0] * dt * 0.5);
            if let Some(first) = kernel.first_mut() {
                *first = (mass * aa_corr[0] + adu_corr[0]) / vv_corr[0];
            }
            for i in start..trunc {
                let conv = dot_prod_par(
                    kernel.slice(s![1..i]),
                    va_corr_rev.slice(s![(len - i)..(len - 1)]),
                );
                kernel[i] = prefac
                    * (mass * aa_corr[i] + adu_corr[i]
                        - 0.5 * dt * va_corr[i] * kernel[0]
                        - dt * conv);
                bar.increment();
            }
            bar.finish();
            kernel
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(backend) => {
            let corrs = HashMap::from_iter(vec![
                (kernel_cuda::VA_CORR_REV, va_corr_rev.view()),
                (kernel_cuda::AA_CORR, aa_corr),
                (kernel_cuda::ADU_CORR, adu_corr),
            ]);
            let mut kernel_cuda =
                kernel_cuda::KernelCuda::new(backend, corrs, kernel, dt, trunc, 1, verbose)?;
            kernel_cuda.gamma_method(vv_corr[0], mass)?;
            kernel_cuda.get_kernel()?
        }
    };
    Ok(kernel.into_pyarray(py))
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
