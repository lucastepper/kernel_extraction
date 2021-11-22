use pyo3::prelude::{pymodule, PyModule, PyResult, Python};


fn _g_method_second_cpu() {
    panic!("Not implemented");
}

#[pymodule]
fn kernel_extraction(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // wrapper of `_g_method_second_cpu`
    #[pyfn(m)]
    #[pyo3(name = "g_method_second_cpu")]
    fn g_method_second_cpu<'py>(
        _py: Python<'py>,
    ) -> PyResult<()> {
        _g_method_second_cpu();
        Ok(())
    }
    Ok(())
}
