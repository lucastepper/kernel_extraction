#[cfg(test)]
use cublas;
use cublas_sys;
use indicatif;
use ndarray::{Array1, ArrayView1};
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyResult;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use std::collections::HashMap;
use std::ffi::CString;
use std::fmt::Display;

use crate::{get_kernel_array_start, Pbar};
// define all corrs to use as keys in a hashmap
pub const VV_CORR_REV: &str = "vv_corr_rev";
pub const VV_CORR_REV_PLUS_NEXT: &str = "vv_corr_plus_next_rev";
pub const VA_CORR_REV: &str = "va_corr_rev";
pub const AA_CORR: &str = "aa_corr";
pub const XDU_CORR: &str = "xdu_corr";
pub const VDU_CORR: &str = "vdu_corr";
pub const ADU_CORR: &str = "adu_corr";
pub const VDU_CORR_CUMSUM: &str = "vdu_corr_cumsum";

// Convert a rustacuda::error::CudaError and cublas::Error to a PyResult
fn into_pyresult<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    match result {
        Ok(t) => Ok(t),
        Err(e) => Err(PyRuntimeError::new_err(format!("(CudaError) {}", e))),
    }
}

// Check that the correlation is present return the device pointer
fn get_device_pointer<T>(
    corrs: &mut HashMap<&'static str, DeviceBuffer<T>>,
    name: &'static str,
) -> PyResult<DevicePointer<T>> {
    match corrs.get_mut(name) {
        Some(d) => Ok(d.as_device_ptr()),
        None => Err(PyValueError::new_err(format!(
            "Correlation function {} not found",
            name
        ))),
    }
}
pub struct KernelCuda {
    kernel: DeviceBuffer<f64>,
    corrs: HashMap<&'static str, DeviceBuffer<f64>>,
    start: usize,
    dt: f64,
    verbose: bool,
    mass: Option<f64>,
    vv_corr_0: Option<f64>,
    len_data: usize,
    dot: DeviceBox<f64>,
    // cuda stuff the order matters, probably because it translates
    // to the order the fields are dropped when KernelCuda.drop is called
    module: Module,
    stream: Stream,
    cublas_context: cublas::Context,
    _cuda_context: Context,
}
impl KernelCuda {
    fn arr_to_gpu(arr: ArrayView1<f64>, name: &str) -> PyResult<DeviceBuffer<f64>> {
        let slice = match arr.as_slice() {
            Some(s) => s,
            None => {
                return Err(PyValueError::new_err(format!(
                    "arr_to_gpu: empyt array with name {}",
                    name
                )))
            }
        };
        Ok(into_pyresult(DeviceBuffer::from_slice(slice))?)
    }

    // This is the full constructor, which needs to set all fields, expecting some to be None
    // All kernel extraction methods get their own function to decide which fields to set
    pub fn new<'py>(
        corrs: HashMap<&'static str, ArrayView1<f64>>,
        kernel_to_extend: Option<&'py PyArray1<f64>>,
        dt: f64,
        mass: Option<f64>,
        trunc: usize,
        verbose: bool,
    ) -> PyResult<KernelCuda> {
        // Initialize the CUDA API
        into_pyresult(rustacuda::init(CudaFlags::empty()))?;
        // init device
        let device = into_pyresult(Device::get_device(0))?;
        // Create a context associated to this device
        let _cuda_context = into_pyresult(Context::create_and_push(
            ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO,
            device,
        ))?;
        // Load the module containing the function we want to call
        let module_data = CString::new(include_str!("kernel_steps.ptx"))?;
        let module = into_pyresult(Module::load_from_string(&module_data))?;
        // Create a stream to submit work to
        let stream = into_pyresult(Stream::new(StreamFlags::NON_BLOCKING, None))?;
        // Create cublas context
        let cublas_context = into_pyresult(cublas::API::create())?;
        // Create a device buffer for each corr and get length, set to length of vv_corr
        let len_data = KernelCuda::get_len(&corrs)?;
        let corrs = corrs
            .into_iter()
            .map(|(k, v)| Ok((k, KernelCuda::arr_to_gpu(v, k)?)))
            .collect::<PyResult<HashMap<&'static str, DeviceBuffer<f64>>>>()?;
        let (start, kernel) = get_kernel_array_start(kernel_to_extend, trunc);
        let kernel_slice = match kernel.as_slice() {
            Some(s) => s,
            None => {
                // We should never really get here, unless somebody sets trunc to 0
                return Err(PyValueError::new_err("kernel_to_extend is empty"));
            }
        };
        // initialize output
        Ok(KernelCuda {
            kernel: into_pyresult(DeviceBuffer::from_slice(kernel_slice))?,
            corrs,
            start,
            dot: into_pyresult(DeviceBox::new(&0.0))?,
            dt,
            verbose,
            mass,
            vv_corr_0: None,
            len_data,
            module,
            stream,
            cublas_context,
            _cuda_context,
        })
    }

    fn get_len(corrs: &HashMap<&'static str, ArrayView1<f64>>) -> PyResult<usize> {
        match corrs.get(VV_CORR_REV) {
            Some(d) => Ok(d.len()),
            None => match corrs.get(VA_CORR_REV) {
                Some(d) => return Ok(d.len()),
                None => {
                    return Err(PyValueError::new_err(format!(
                        "{} or {} not found while trying to compute 'len_data'",
                        VV_CORR_REV, VA_CORR_REV
                    )))
                }
            },
        }
    }

    fn get_mass(&self) -> PyResult<f64> {
        match self.mass {
            Some(m) => Ok(m),
            None => Err(PyValueError::new_err("mass in KernelCuda is None")),
        }
    }

    fn get_vv_corr_0(&self) -> PyResult<f64> {
        match self.vv_corr_0 {
            Some(vv) => Ok(vv),
            None => Err(PyValueError::new_err("vv_corr_0 in KernelCuda is None")),
        }
    }

    pub fn set_vv_corr_0(&mut self, vv_corr_0: f64) {
        self.vv_corr_0 = Some(vv_corr_0);
    }

    pub fn g_method_xdu_nomass(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let vv_corr_rev = get_device_pointer(&mut self.corrs, VV_CORR_REV)?;
        let xdu_corr = get_device_pointer(&mut self.corrs, XDU_CORR)?;
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel g_method_xdu_nomass",
            "G_method Cxdu nomass",
            self.kernel.len(),
            self.start,
        );
        for i in self.start..self.kernel.len() {
            // compute the convolution via dot of cublas
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw().offset(1),
                    1i32,
                    vv_corr_rev.as_raw().offset((self.len_data - i) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_g_method_xdu_nomass_double<<<1, 1, 0, stream>>>(
                        self.dot.as_device_ptr(),
                        self.kernel.as_device_ptr(),
                        vv_corr_rev,
                        xdu_corr,
                        self.dt,
                        i,
                        self.len_data
                    )),
                )?;
                into_pyresult(stream.synchronize())?;
            }
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn g_method_xdu(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let vv_corr_rev = get_device_pointer(&mut self.corrs, VV_CORR_REV)?;
        let xdu_corr = get_device_pointer(&mut self.corrs, XDU_CORR)?;
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel g_method_xdu",
            "G_method Cxdu",
            self.kernel.len(),
            self.start,
        );
        for i in self.start..self.kernel.len() {
            // compute the convolution via dot of cublas
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw().offset(1),
                    1i32,
                    vv_corr_rev.as_raw().offset((self.len_data - i) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_g_method_xdu_double<<<1, 1, 0, stream>>>(
                        self.dot.as_device_ptr(),
                        self.kernel.as_device_ptr(),
                        vv_corr_rev,
                        xdu_corr,
                        self.get_mass()?,
                        self.dt,
                        i,
                        self.len_data
                    )),
                )?;
                into_pyresult(stream.synchronize())?;
            }
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn g_method_xdu_halfstep(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let vv_corr_rev = get_device_pointer(&mut self.corrs, VV_CORR_REV)?;
        let xdu_corr = get_device_pointer(&mut self.corrs, XDU_CORR)?;
        let vv_corr_plus_next_rev = get_device_pointer(&mut self.corrs, VV_CORR_REV_PLUS_NEXT)?;
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel g_method_xdu_halfstep",
            "G_method Cxdu halfstep",
            self.kernel.len(),
            self.start,
        );
        for i in self.start..self.kernel.len() {
            // compute the convolution via dot of cublas
            // We have to keep in mind that len belongs to the lenght of vv_corr_rev, which is
            // one element longer than the the array containing cvv[::-1][1:] + cvv[::-1][:-1]
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw().offset(1),
                    1i32,
                    vv_corr_plus_next_rev
                        .as_raw()
                        .offset((self.len_data - i - 1) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_g_method_xdu_halfstep_double<<<1, 1, 0, stream>>>(
                        self.dot.as_device_ptr(),
                        self.kernel.as_device_ptr(),
                        vv_corr_rev,
                        xdu_corr,
                        self.get_mass()?,
                        self.dt,
                        i,
                        self.len_data
                    )),
                )?;
                into_pyresult(stream.synchronize())?;
            }
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn g_method_vdu(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let vv_corr_rev = get_device_pointer(&mut self.corrs, VV_CORR_REV)?;
        let vdu_corr = get_device_pointer(&mut self.corrs, VDU_CORR)?;
        let vdu_corr_cumsum = get_device_pointer(&mut self.corrs, VDU_CORR_CUMSUM)?;
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel g_method_vdu",
            "G_method Cvdu",
            self.kernel.len(),
            self.start,
        );
        for i in self.start..self.kernel.len() {
            // compute the convolution via dot of cublas
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw().offset(1),
                    1i32,
                    vv_corr_rev.as_raw().offset((self.len_data - i) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_g_method_vdu_double<<<1, 1, 0, stream>>>(
                        self.dot.as_device_ptr(),
                        self.kernel.as_device_ptr(),
                        vv_corr_rev,
                        vdu_corr,
                        vdu_corr_cumsum,
                        self.get_mass()?,
                        self.dt,
                        i,
                        self.len_data
                    )),
                )?;
                into_pyresult(stream.synchronize())?;
            }
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn gamma_method(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let va_corr_rev = get_device_pointer(&mut self.corrs, VA_CORR_REV)?;
        let aa_corr = get_device_pointer(&mut self.corrs, AA_CORR)?;
        let adu_corr = get_device_pointer(&mut self.corrs, ADU_CORR)?;
        // Here we use the 0th loop iteration to compute the kernel at time 0
        // So in the case that start is 1 (which happens if we start kernel from scratch)
        // We set start to 0 and compute the kernel at time 0
        let start = if self.start == 1 { 0 } else { self.start };
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel gamma_method",
            "Gamma_method",
            self.kernel.len(),
            start,
        );
        for i in start..self.kernel.len() {
            // compute the convolution via dot of cublas
            if i > 0 {
                unsafe {
                    cublas_sys::cublasDdot_v2(
                        *self.cublas_context.id_c(),
                        i as i32,
                        self.kernel.as_device_ptr().as_raw().offset(1),
                        1i32,
                        va_corr_rev.as_raw().offset((self.len_data - i) as isize),
                        1i32,
                        self.dot.as_device_ptr().as_raw_mut(),
                    );
                }
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_gamma_method_double<<<1, 1, 0, stream>>>(
                        self.dot.as_device_ptr(),
                        self.kernel.as_device_ptr(),
                        va_corr_rev,
                        aa_corr,
                        adu_corr,
                        self.get_vv_corr_0()?,
                        self.get_mass()?,
                        self.dt,
                        i,
                        self.len_data
                    )),
                )?;
                into_pyresult(stream.synchronize())?;
            }
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn get_kernel(&self) -> PyResult<Array1<f64>> {
        let mut kernel_host = vec![0.; self.kernel.len()];
        into_pyresult(self.kernel.copy_to(&mut kernel_host))?;
        Ok(Array1::from_vec(kernel_host))
    }
}
