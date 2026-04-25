fn main() {
    println!("cargo:rerun-if-changed=src/crossover.cu");
    println!("cargo:rerun-if-changed=src/mutation.cu");
    println!("cargo:rerun-if-changed=src/network_compilation.cu");

    cc::Build::new()
        .cuda(true)
        .flag("-arch=sm_89")
        .files(&["src/crossover.cu", "src/mutation.cu", "src/network_compilation.cu"])
        .compile("moonai_evolution_cuda");
}
