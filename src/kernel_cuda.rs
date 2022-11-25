#[cfg(test)]
use cublas;
use cublas_sys;
use ndarray as nd;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyResult;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::ffi::CString;
use std::fmt::Display;

fn into_pyresult<T, E: Display>(result: Result<T, E>) -> PyResult<T> {
    match result {
        Ok(t) => Ok(t),
        Err(e) => Err(PyRuntimeError::new_err(format!("(CudaError) {}", e))),
    }
}

struct KernelCuda {
    kernel: DeviceBuffer<f64>,
    vv_corr_rev: Option<DeviceBuffer<f64>>,
    xdu_corr: Option<DeviceBuffer<f64>>,
    dt: f64,
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

    fn new(
        vv_corr_rev: Option<nd::ArrayView1<f64>>,
        xdu_corr: Option<nd::ArrayView1<f64>>,
        dt: f64,
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
            module,
            stream,
            cublas_context,
            _cuda_context,
        })
    }

    fn g_method_xdu_nomass(&mut self) -> PyResult<()> {
        // check that the kernels are present
        let vv_corr_rev = match self.vv_corr_rev {
            Some(ref mut vv_corr_rev) => vv_corr_rev,
            None => return Err(PyValueError::new_err("vv_corr_rev not present")),
        };
        let xdu_corr = match self.xdu_corr {
            Some(ref mut xdu_corr) => xdu_corr,
            None => return Err(PyValueError::new_err("xdu_corr not present")),
        };
        let len = vv_corr_rev.len();
        let module = &self.module;
        let stream = &self.stream;
        for i in 1..self.kernel.len() {
            // compute dot via cublas
            unsafe {
                cublas_sys::cublasDdot_v2(
                    *self.cublas_context.id_c(),
                    i as i32,
                    self.kernel.as_device_ptr().as_raw(),
                    1i32,
                    vv_corr_rev
                        .as_device_ptr()
                        .as_raw()
                        .offset((len - i - 1) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the dot product
                into_pyresult(launch!(module.kernel_step_double<<<1, 1, 0, stream>>>(
                    self.dot.as_device_ptr(),
                    self.kernel.as_device_ptr(),
                    vv_corr_rev.as_device_ptr(),
                    xdu_corr.as_device_ptr(),
                    self.dt,
                    i,
                    len
                )))?;
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
    let mut kernel_cuda = KernelCuda::new(Some(vv_corr), Some(xdu_corr), dt, trunc)?;
    kernel_cuda.g_method_xdu_nomass()?;
    kernel_cuda.get_kernel()
}
