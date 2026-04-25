fn main() {
    println!("cargo:rerun-if-changed=src/kernel.cu");

    cc::Build::new().cuda(true).flag("-arch=sm_89").file("src/kernel.cu").compile("moonai_simulation_cuda");
}
