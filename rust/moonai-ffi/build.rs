fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let ffi_dir = std::path::Path::new(&manifest_dir);

    // Check if we should use CMake build (when CUDA is available)
    let use_cmake = std::env::var("MOONAI_USE_CMAKE").unwrap_or_default() == "1";

    if use_cmake {
        // Use CMake build
        let build_dir = std::path::Path::new(&manifest_dir).join("cmake_build");
        std::fs::create_dir_all(&build_dir).ok();

        let mut cmake = std::process::Command::new("cmake");
        cmake.arg("-S.").arg("-B").arg(&build_dir);
        cmake.arg("-DCMAKE_BUILD_TYPE=Release");
        cmake.arg("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON");

        if !cmake.output().map(|o| o.status.success()).unwrap_or(false) {
            println!("cargo:warning=CMake configuration failed");
        }

        let mut make = std::process::Command::new("cmake");
        make.arg("--build").arg(&build_dir).arg("--parallel");

        if !make.output().map(|o| o.status.success()).unwrap_or(false) {
            println!("cargo:warning=CMake build failed");
        }

        println!("cargo:rerun-if-changed=CMakeLists.txt");
        println!("cargo:rerun-if-changed=src_cpp/simulation.cpp");
        println!("cargo:rerun-if-changed=src_cpp/random.cpp");
    } else {
        // Fallback: use cxx-build with simple stubs
        cxx_build::bridge("src/lib.rs")
            .flag_if_supported("-std=c++17")
            .include("include")
            .file("src_cpp/random.cpp")
            .file("src_cpp/gpu_sim.cpp")
            .compile("moonai-ffi");

        println!("cargo:rerun-if-changed=src/lib.rs");
        println!("cargo:rerun-if-changed=src_cpp/random.cpp");
        println!("cargo:rerun-if-changed=src_cpp/gpu_sim.cpp");
        println!("cargo:rerun-if-changed=include/random.hpp");
        println!("cargo:rerun-if-changed=include/gpu_sim.hpp");
    }
}