#[cfg(test)]
use super::kernel_cpu;
use cublas;
use cublas_sys;
use ndarray as nd;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

struct KernelCuda {
    kernel: DeviceBuffer<f64>,
    vv_corr_rev: DeviceBuffer<f64>,
    xdu_corr: DeviceBuffer<f64>,
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
    fn new(
        vv_corr_rev: nd::ArrayView1<f64>,
        xdu_corr: nd::ArrayView1<f64>,
        dt: f64,
        trunc: usize,
    ) -> Result<KernelCuda, Box<dyn Error>> {
        // Initialize the CUDA API
        rustacuda::init(CudaFlags::empty())?;
        // init device
        let device = Device::get_device(0)?;
        // Create a context associated to this device
        let _cuda_context =
            Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;
        // Load the module containing the function we want to call
        let module_data = CString::new(include_str!("compute_kernel.ptx"))?;
        let module = Module::load_from_string(&module_data)?;
        // Create a stream to submit work to
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        // Create cublas context
        let cublas_context = cublas::API::create().unwrap();
        // init obj
        Ok(KernelCuda {
            kernel: DeviceBuffer::from_slice(&vec![0.; trunc])?,
            vv_corr_rev: DeviceBuffer::from_slice(&vv_corr_rev.as_slice().unwrap())?,
            xdu_corr: DeviceBuffer::from_slice(&xdu_corr.as_slice().unwrap())?,
            dot: DeviceBox::new(&0.0)?,
            dt,
            module,
            stream,
            cublas_context,
            _cuda_context,
        })
    }
    fn compute(&mut self) {
        let len = self.vv_corr_rev.len();
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
                    self.vv_corr_rev
                        .as_device_ptr()
                        .as_raw()
                        .offset((len - i - 1) as isize),
                    1i32,
                    self.dot.as_device_ptr().as_raw_mut(),
                );
            }
            unsafe {
                // compute the kernel at time i * dt using the dot product
                // launch!(module.kernel_step_double<<<1, 1, 0, stream>>>(
                launch!(module.kernel_step_double<<<1, 1, 0, stream>>>(
                    self.dot.as_device_ptr(),
                    self.kernel.as_device_ptr(),
                    self.vv_corr_rev.as_device_ptr(),
                    self.xdu_corr.as_device_ptr(),
                    self.dt,
                    i,
                    len
                ))
                .unwrap();
                // }
                stream.synchronize().unwrap();
            }
        }
    }
    fn get_kernel(&self) -> nd::Array1<f64> {
        let mut kernel_host = vec![0.; self.kernel.len()];
        self.kernel.copy_to(&mut kernel_host).unwrap();
        nd::Array1::from_vec(kernel_host)
    }
}

pub fn compute_kernel(
    vv_corr: nd::ArrayView1<f64>,
    xdu_corr: nd::ArrayView1<f64>,
    dt: f64,
    trunc: usize,
) -> nd::Array1<f64> {
    let mut kernel_cuda = KernelCuda::new(vv_corr, xdu_corr, dt, trunc).unwrap();
    kernel_cuda.compute();
    kernel_cuda.get_kernel()
}

#[cfg(test)]
mod tests {
    const EPS: f64 = 1e-6;
    use super::*;

    #[test]
    fn test_kernel() {
        // init test data
        let dt = 0.01;
        let trunc = (1. / dt) as usize;
        let time = nd::Array1::range(0., 1., dt);
        let vv_corr = time.mapv(|t| f64::cos(100. * t) * f64::exp(-t / 0.2));
        let vv_corr_rev: Vec<f64> = vv_corr.iter().rev().map(|x| *x).collect();
        let vv_corr_rev = nd::Array1::from_vec(vv_corr_rev);
        let xdu_corr = time.mapv(|t| 2.494 * f64::exp(-t));
        // data ref
        let kernel_ref = kernel_cpu::_g_method_second_cpu(
            &vv_corr_rev.as_slice().unwrap(),
            &xdu_corr.as_slice().unwrap(),
            dt,
            trunc,
        );
        let kernel_ref = nd::Array1::from_vec(kernel_ref);
        // compute test data
        let mut kernel_cuda =
            KernelCuda::new(vv_corr_rev.view(), xdu_corr.view(), dt, trunc).unwrap();
        kernel_cuda.compute();
        let kernel_test = kernel_cuda.get_kernel();
        println!("Time: {:?}", time);
        println!("Cvv: {:?}", vv_corr);
        println!("xdU_corr {:?}", xdu_corr);
        assert!(&kernel_ref.abs_diff_eq(&kernel_test, EPS));
    }
}
