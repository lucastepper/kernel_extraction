#[cfg(test)]
use cublas;
use cublas_sys;
use ndarray as nd;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyResult;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use rustacuda_core::DevicePointer;
use std::ffi::CString;
use std::fmt::Display;

// Convert a rustacuda::error::CudaError and cublas::Error to a PyResult
fn into_pyresult<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    match result {
        Ok(t) => Ok(t),
        Err(e) => Err(PyRuntimeError::new_err(format!("(CudaError) {}", e))),
    }
}

// Check that the given Option is not none and return the device pointer
fn get_device_pointer<T>(
    opt: &mut Option<DeviceBuffer<T>>,
    name: &str,
) -> PyResult<DevicePointer<T>> {
    match opt {
        Some(d) => Ok(d.as_device_ptr()),
        None => Err(PyRuntimeError::new_err(format!(
            "DeviceBox {} is None",
            name,
        ))),
    }
}

pub struct KernelCuda {
    kernel: DeviceBuffer<f64>,
    vv_corr_rev: Option<DeviceBuffer<f64>>,
    xdu_corr: Option<DeviceBuffer<f64>>,
    dt: f64,
    mass: Option<f64>,
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
    fn arr_to_gpu(arr: Option<nd::ArrayView1<f64>>) -> PyResult<Option<DeviceBuffer<f64>>> {
        match arr {
            None => Ok(None),
            Some(arr) => {
                let slice = match arr.as_slice() {
                    Some(s) => s,
                    None => return Err(PyValueError::new_err("arr_to_gpu: empyt array")),
                };
                Ok(Some(into_pyresult(DeviceBuffer::from_slice(slice))?))
            }
        }
    }

    // This is the full constructor, which needs to set all fields, expecting some to be None
    // All kernel extraction methods get their own function to decide which fields to set
    fn new(
        vv_corr_rev: Option<nd::ArrayView1<f64>>,
        xdu_corr: Option<nd::ArrayView1<f64>>,
        dt: f64,
        mass: Option<f64>,
        trunc: usize,
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
        let module_data = CString::new(include_str!("compute_kernel.ptx"))?;
        let module = into_pyresult(Module::load_from_string(&module_data))?;
        // Create a stream to submit work to
        let stream = into_pyresult(Stream::new(StreamFlags::NON_BLOCKING, None))?;
        // Create cublas context
        let cublas_context = into_pyresult(cublas::API::create())?;
        // init obj
        Ok(KernelCuda {
            kernel: into_pyresult(DeviceBuffer::from_slice(&vec![0.; trunc]))?,
            vv_corr_rev: Self::arr_to_gpu(vv_corr_rev)?,
            xdu_corr: Self::arr_to_gpu(xdu_corr)?,
            dot: into_pyresult(DeviceBox::new(&0.0))?,
            dt,
            mass,
            len_data: vv_corr_rev.map(|a| a.len()).unwrap_or(0),
            module,
            stream,
            cublas_context,
            _cuda_context,
        })
    }

    fn get_mass(&self) -> PyResult<f64> {
        match self.mass {
            Some(m) => Ok(m),
            None => Err(PyValueError::new_err("mass in KernelCuda is None")),
        }
    }

    fn g_method_xdu_nomass(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let vv_corr_rev = get_device_pointer(&mut self.vv_corr_rev, "vv_corr_rev")?;
        let xdu_corr = get_device_pointer(&mut self.xdu_corr, "xdu_corr")?;
        for i in 1..self.kernel.len() {
            // compute the convolution via dot of cublas
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw(),
                    1i32,
                    vv_corr_rev
                        .as_raw()
                        .offset((self.len_data - i - 1) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_g_method_nomass_double<<<1, 1, 0, stream>>>(
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
        }
        Ok(())
    }

    fn g_method_xdu(&mut self) -> PyResult<()> {
        let module = &self.module;
        let stream = &self.stream;
        let vv_corr_rev = get_device_pointer(&mut self.vv_corr_rev, "vv_corr_rev")?;
        let xdu_corr = get_device_pointer(&mut self.xdu_corr, "xdu_corr")?;
        for i in 1..self.kernel.len() {
            // compute the convolution via dot of cublas
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw(),
                    1i32,
                    vv_corr_rev
                        .as_raw()
                        .offset((self.len_data - i - 1) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the precomputed convolution
                into_pyresult(
                    launch!(module.kernel_step_g_method_double<<<1, 1, 0, stream>>>(
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
        }
        Ok(())
    }

    fn get_kernel(&self) -> PyResult<nd::Array1<f64>> {
        let mut kernel_host = vec![0.; self.kernel.len()];
        into_pyresult(self.kernel.copy_to(&mut kernel_host))?;
        Ok(nd::Array1::from_vec(kernel_host))
    }
}

pub fn g_method_xdu_nomass(
    vv_corr: nd::ArrayView1<f64>,
    xdu_corr: nd::ArrayView1<f64>,
    dt: f64,
    trunc: usize,
) -> PyResult<nd::Array1<f64>> {
    let mut kernel_cuda = KernelCuda::new(Some(vv_corr), Some(xdu_corr), dt, None, trunc)?;
    kernel_cuda.g_method_xdu_nomass()?;
    kernel_cuda.get_kernel()
}

pub fn g_method_xdu(
    vv_corr: nd::ArrayView1<f64>,
    xdu_corr: nd::ArrayView1<f64>,
    dt: f64,
    mass: f64,
    trunc: usize,
) -> PyResult<nd::Array1<f64>> {
    let mut kernel_cuda = KernelCuda::new(Some(vv_corr), Some(xdu_corr), dt, Some(mass), trunc)?;
    kernel_cuda.g_method_xdu()?;
    kernel_cuda.get_kernel()
}
