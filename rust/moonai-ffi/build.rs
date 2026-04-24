fn main() {
    cxx_build::bridge("src/lib.rs")
        .flag_if_supported("-std=c++17")
        .include("include")
        .file("src_cpp/random.cpp")
        .compile("moonai-ffi");

    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src_cpp/random.cpp");
    println!("cargo:rerun-if-changed=include/random.hpp");
}
