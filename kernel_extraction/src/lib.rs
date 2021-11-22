use pyo3::prelude::{pymodule, PyModule, PyResult, Python};
use numpy::{IntoPyArray, PyArray};
use ndarray::Dim;
use packed_simd::f64x4;
use rayon::prelude::*;


#[allow(dead_code)]
fn dot_prod(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|v| v.0 * v.1).sum()
}

#[allow(dead_code)]
fn dot_prod_par(a: &[f64], b: &[f64]) -> f64 {
    a.par_iter()
     .zip(b.par_iter())
     .map(|(x, y)| x * y)
     .sum()
}

#[allow(dead_code)]
fn dot_prod_par_simd(a: &[f64], b: &[f64]) -> f64 {
    // code for avx_256 instructions, should probs require avx2
    let max_len = a.len() / 4 * 4;
    let remainder = a.len() % 4;
    // iterate trough elements at a stride of 4
    let mut dot_prod_simd = a[0..max_len].par_chunks_exact(4)
        .map(f64x4::from_slice_unaligned)
        .zip(b[0..max_len].par_chunks_exact(4).map(f64x4::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f64x4>()
        .sum();
    // add the last elems, up to three, if len(a) / 4 != 0
    for i in max_len..(max_len + remainder) {
        dot_prod_simd += a[i] * b[i];
    }
    dot_prod_simd
}

#[allow(dead_code)]
fn dot_prod_simd(a: &[f64], b: &[f64]) -> f64 {
    // code for avx_256 instructions, should probs require avx2
    let max_len = a.len() / 4 * 4;
    let remainder = a.len() % 4;
    // iterate trough elements at a stride of 4, could in future be done with iterator remaining somehow
    let mut dot_prod_simd = a[0..max_len].chunks_exact(4)
        .map(f64x4::from_slice_unaligned)
        .zip(b[0..max_len].chunks_exact(4).map(f64x4::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f64x4>()
        .sum();
    // add the last elems, up to three, if len(a) / 4 != 0
    for i in max_len..(max_len + remainder) {
        dot_prod_simd += a[i] * b[i];
    }
    dot_prod_simd
}

fn _g_method_second_cpu(vv_corr_rev: &[f64], xdu_corr: &[f64], dt: f64, trunc: usize) -> Vec<f64> {
    let len =  vv_corr_rev.len();
    let mut convolution: f64;
    let mut kernel = vec![0.; trunc];
    for i in 1..trunc {
        convolution = dot_prod_par_simd(&kernel[1..i], &vv_corr_rev[(len - i)..(len - 1)]);
        kernel[i] = 2. / vv_corr_rev[len - 1] * ((
            xdu_corr[i]
            - vv_corr_rev[len - 1 - i] * xdu_corr[0] / vv_corr_rev[len - 1]
        ) / dt - convolution);
    }
    kernel
}

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
        let vv_corr_rev: Vec<f64> = vv_corr.iter()
                                           .rev()
                                           .map(|x| *x)
                                           .collect();
        _g_method_second_cpu(&vv_corr_rev, &xdu_corr, dt, trunc).into_pyarray(py)
    }
    Ok(())
}
