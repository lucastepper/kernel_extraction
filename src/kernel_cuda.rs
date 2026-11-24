use cudarc::cublas::{sys as blas_sys, CudaBlas};
use cudarc::driver::{
    self, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use ndarray::{Array1, ArrayView1};
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::PyResult;
use std::collections::HashMap;
use std::fmt::Display;
use std::sync::Arc;

use crate::{get_kernel_array_start, Pbar};

pub const VV_CORR_REV: &str = "vv_corr_rev";
pub const VV_CORR_REV_PLUS_NEXT: &str = "vv_corr_plus_next_rev";
pub const VA_CORR_REV: &str = "va_corr_rev";
pub const AA_CORR: &str = "aa_corr";
pub const XDU_CORR: &str = "xdu_corr";
pub const VDU_CORR: &str = "vdu_corr";
pub const ADU_CORR: &str = "adu_corr";
pub const VDU_CORR_CUMSUM: &str = "vdu_corr_cumsum";

// Each PTX entry point computes a single sample and runs on one thread.
const STEP_LAUNCH: LaunchConfig = LaunchConfig {
    grid_dim: (1, 1, 1),
    block_dim: (1, 1, 1),
    shared_mem_bytes: 0,
};

fn cuda_error(error: impl Display) -> pyo3::PyErr {
    PyRuntimeError::new_err(format!("CUDA error: {error}"))
}

fn copy_to_device(stream: &Arc<CudaStream>, array: ArrayView1<f64>) -> PyResult<CudaSlice<f64>> {
    let buffer = if array.is_empty() {
        stream.null()
    } else if let Some(slice) = array.as_slice() {
        stream.clone_htod(slice)
    } else {
        stream.clone_htod(&array.to_vec())
    };
    buffer.map_err(cuda_error)
}

pub struct CudaBackend {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    step: CudaFunction,
    _previous_context: PreviousContext,
}

struct PreviousContext(driver::sys::CUcontext);

impl Drop for PreviousContext {
    fn drop(&mut self) {
        // cudarc binds its context to this thread. Restore the caller's context.
        let _ = unsafe { driver::result::ctx::set_current(self.0) };
    }
}

impl CudaBackend {
    pub fn new(method: &str) -> PyResult<Self> {
        // cudarc's loader panics if a shared library cannot be opened.
        if !unsafe { driver::sys::is_culib_present() } {
            return Err(cuda_error("NVIDIA driver library could not be loaded"));
        }
        if !unsafe { blas_sys::is_culib_present() } {
            return Err(cuda_error("cuBLAS library could not be loaded"));
        }

        driver::result::init().map_err(cuda_error)?;
        let previous_context = PreviousContext(
            driver::result::ctx::get_current()
                .map_err(cuda_error)?
                .unwrap_or(std::ptr::null_mut()),
        );
        let context = CudaContext::new(0).map_err(cuda_error)?;
        let module = context
            .load_module(Ptx::from_src(include_str!("kernel_steps.ptx")))
            .map_err(cuda_error)?;
        let step = module
            .load_function(&format!("kernel_step_{method}_double"))
            .map_err(cuda_error)?;
        let stream = context.new_stream().map_err(cuda_error)?;
        let blas = CudaBlas::new(stream.clone()).map_err(cuda_error)?;
        blas.set_pointer_mode(blas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE)
            .map_err(cuda_error)?;
        Ok(Self {
            stream,
            blas,
            step,
            _previous_context: previous_context,
        })
    }

    fn convolution(
        &self,
        kernel: &CudaSlice<f64>,
        corr: &CudaSlice<f64>,
        dot: &mut CudaSlice<f64>,
        i: usize,
        first_sample: usize,
    ) -> PyResult<()> {
        if i <= first_sample {
            // The sum is empty and dot still contains its initial zero.
            return Ok(());
        }
        let count = i - first_sample;
        let corr_end = corr.len() - 1;
        let kernel = kernel.slice(first_sample..i);
        let corr = corr.slice((corr_end - count)..corr_end);
        let (kernel_ptr, _kernel_sync) = kernel.device_ptr(&self.stream);
        let (corr_ptr, _corr_sync) = corr.device_ptr(&self.stream);
        let (dot_ptr, _dot_sync) = dot.device_ptr_mut(&self.stream);
        // Both input views contain count elements. cuBLAS writes one f64 to dot,
        // using the same stream as the kernel step that consumes it.
        unsafe {
            blas_sys::cublasDdot_v2(
                *self.blas.handle(),
                count as i32,
                kernel_ptr as *const f64,
                1,
                corr_ptr as *const f64,
                1,
                dot_ptr as *mut f64,
            )
            .result()
            .map_err(cuda_error)
        }
    }
}

pub struct KernelCuda {
    kernel: CudaSlice<f64>,
    corrs: HashMap<&'static str, CudaSlice<f64>>,
    dot: CudaSlice<f64>,
    backend: CudaBackend,
    start: usize,
    first_sample: usize,
    trunc: usize,
    len_data: i32,
    dt: f64,
    verbose: bool,
}

impl KernelCuda {
    pub fn new(
        backend: CudaBackend,
        corrs: HashMap<&'static str, ArrayView1<f64>>,
        kernel_to_extend: Option<&PyArray1<f64>>,
        dt: f64,
        trunc: usize,
        first_sample: usize,
        verbose: bool,
    ) -> PyResult<Self> {
        let len_data = corrs
            .get(VV_CORR_REV)
            .or_else(|| corrs.get(VA_CORR_REV))
            .ok_or_else(|| PyValueError::new_err("Missing reversed correlation"))?
            .len();
        let len_data = i32::try_from(len_data)
            .map_err(|_| PyValueError::new_err("Correlation is too long for CUDA"))?;
        let stream = &backend.stream;
        let corrs = corrs
            .into_iter()
            .map(|(name, corr)| Ok((name, copy_to_device(stream, corr)?)))
            .collect::<PyResult<_>>()?;
        let (start, kernel) = get_kernel_array_start(kernel_to_extend, trunc, first_sample);
        let kernel = copy_to_device(stream, kernel.view())?;
        let dot = stream.alloc_zeros(1).map_err(cuda_error)?;
        Ok(Self {
            backend,
            kernel,
            corrs,
            dot,
            start,
            first_sample,
            trunc,
            len_data,
            dt,
            verbose,
        })
    }

    pub fn g_method_xdu_nomass(&mut self) -> PyResult<()> {
        let vv_corr_rev = &self.corrs[VV_CORR_REV];
        let xdu_corr = &self.corrs[XDU_CORR];
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel g_method_xdu_nomass",
            "G_method Cxdu nomass",
            self.trunc,
            self.start,
        );
        for i in self.start..self.trunc {
            self.backend.convolution(
                &self.kernel,
                vv_corr_rev,
                &mut self.dot,
                i,
                self.first_sample,
            )?;
            let index = i as i32;
            let mut launch = self.backend.stream.launch_builder(&self.backend.step);
            launch
                .arg(&self.dot)
                .arg(&mut self.kernel)
                .arg(vv_corr_rev)
                .arg(xdu_corr)
                .arg(&self.dt)
                .arg(&index)
                .arg(&self.len_data);
            // The arguments match the embedded PTX and trunc bounds all array accesses.
            unsafe { launch.launch(STEP_LAUNCH) }.map_err(cuda_error)?;
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn g_method_xdu(&mut self, mass: f64, halfstep: bool) -> PyResult<()> {
        let vv_corr_rev = &self.corrs[VV_CORR_REV];
        let xdu_corr = &self.corrs[XDU_CORR];
        let (dot_corr, message, prefix) = if halfstep {
            (
                &self.corrs[VV_CORR_REV_PLUS_NEXT],
                "Computing kernel g_method_xdu_halfstep",
                "G_method Cxdu halfstep",
            )
        } else {
            (
                vv_corr_rev,
                "Computing kernel g_method_xdu",
                "G_method Cxdu",
            )
        };
        let mut bar = Pbar::new(self.verbose, message, prefix, self.trunc, self.start);
        for i in self.start..self.trunc {
            self.backend.convolution(
                &self.kernel,
                dot_corr,
                &mut self.dot,
                i,
                self.first_sample,
            )?;
            let index = i as i32;
            let mut launch = self.backend.stream.launch_builder(&self.backend.step);
            launch
                .arg(&self.dot)
                .arg(&mut self.kernel)
                .arg(vv_corr_rev)
                .arg(xdu_corr)
                .arg(&mass)
                .arg(&self.dt)
                .arg(&index)
                .arg(&self.len_data);
            // Both XDU variants share the same PTX argument layout.
            unsafe { launch.launch(STEP_LAUNCH) }.map_err(cuda_error)?;
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn g_method_vdu(&mut self, mass: f64) -> PyResult<()> {
        let vv_corr_rev = &self.corrs[VV_CORR_REV];
        let vdu_corr = &self.corrs[VDU_CORR];
        let vdu_corr_cumsum = &self.corrs[VDU_CORR_CUMSUM];
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel g_method_vdu",
            "G_method Cvdu",
            self.trunc,
            self.start,
        );
        for i in self.start..self.trunc {
            self.backend.convolution(
                &self.kernel,
                vv_corr_rev,
                &mut self.dot,
                i,
                self.first_sample,
            )?;
            let index = i as i32;
            let mut launch = self.backend.stream.launch_builder(&self.backend.step);
            launch
                .arg(&self.dot)
                .arg(&mut self.kernel)
                .arg(vv_corr_rev)
                .arg(vdu_corr)
                .arg(vdu_corr_cumsum)
                .arg(&mass)
                .arg(&self.dt)
                .arg(&index)
                .arg(&self.len_data);
            // The arguments match the embedded PTX and trunc bounds all array accesses.
            unsafe { launch.launch(STEP_LAUNCH) }.map_err(cuda_error)?;
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn gamma_method(&mut self, vv_corr_0: f64, mass: f64) -> PyResult<()> {
        let va_corr_rev = &self.corrs[VA_CORR_REV];
        let aa_corr = &self.corrs[AA_CORR];
        let adu_corr = &self.corrs[ADU_CORR];
        let mut bar = Pbar::new(
            self.verbose,
            "Computing kernel gamma_method",
            "Gamma_method",
            self.trunc,
            self.start,
        );
        // Like the CPU method, compute step zero even when extending a kernel.
        for i in (0..self.kernel.len().min(1)).chain(self.start..self.trunc) {
            self.backend.convolution(
                &self.kernel,
                va_corr_rev,
                &mut self.dot,
                i,
                self.first_sample,
            )?;
            let index = i as i32;
            let mut launch = self.backend.stream.launch_builder(&self.backend.step);
            launch
                .arg(&self.dot)
                .arg(&mut self.kernel)
                .arg(va_corr_rev)
                .arg(aa_corr)
                .arg(adu_corr)
                .arg(&vv_corr_0)
                .arg(&mass)
                .arg(&self.dt)
                .arg(&index)
                .arg(&self.len_data);
            // Step zero uses the first correlation samples. Later steps are bounded by trunc.
            unsafe { launch.launch(STEP_LAUNCH) }.map_err(cuda_error)?;
            bar.increment();
        }
        bar.finish();
        Ok(())
    }

    pub fn get_kernel(&self) -> PyResult<Array1<f64>> {
        if self.kernel.is_empty() {
            return Ok(Array1::zeros(0));
        }
        let stream = &self.backend.stream;
        let kernel = stream.clone_dtoh(&self.kernel).map_err(cuda_error)?;
        stream.synchronize().map_err(cuda_error)?;
        Ok(Array1::from_vec(kernel))
    }
}
