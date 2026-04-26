use std::env;
use std::path::Path;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    let src_config = Path::new(&manifest_dir)
        .join("..")
        .join("..")
        .join("config")
        .join("config.lua");
    let dst_config = Path::new(&out_dir).join("config.lua");
    if src_config.exists() {
        println!("cargo:rerun-if-changed={}", src_config.display());
        std::fs::copy(&src_config, &dst_config).unwrap();
    }

    let src_settings = Path::new(&manifest_dir)
        .join("..")
        .join("..")
        .join("config")
        .join("settings.json");
    let dst_settings = Path::new(&out_dir).join("settings.json");
    if src_settings.exists() {
        println!("cargo:rerun-if-changed={}", src_settings.display());
        std::fs::copy(&src_settings, &dst_settings).unwrap();
    }
}