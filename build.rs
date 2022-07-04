#[allow(unused_imports)]
use std::process;

fn main() {
    let test_version = std::process::Command::new("nvcc").arg("--version").output();
    match test_version {
        Ok(_o) => (),
        Err(e) => {
            eprintln!("Cuda compiler did not work, is it installed. Err: {:?}", e);
        }
    }
    let build_output = std::process::Command::new("nvcc")
        .arg("--ptx")
        .arg("-o")
        .arg("src/compute_kernel.ptx")
        .arg("src/compute_kernel.cu")
        .output();
    match build_output {
        Ok(_o) => (),
        Err(e) => {
            eprintln!("Cuda compilation failed. Err: {:?}", e);
        }
    }
}
