#[cxx::bridge]
mod random_ffi {
    unsafe extern "C++" {
        include!("random.hpp");

        type CRandom;

        unsafe fn c_random_create(seed: i32) -> *mut CRandom;
        unsafe fn c_random_destroy(rng: *mut CRandom);
        unsafe fn c_random_next_int(rng: *mut CRandom, min: i32, max: i32) -> i32;
        unsafe fn c_random_next_float(rng: *mut CRandom, min: f32, max: f32) -> f32;
        unsafe fn c_random_next_bool(rng: *mut CRandom, probability: f32) -> bool;
        unsafe fn c_random_weighted_select(rng: *mut CRandom, weights: *const f32, len: usize) -> i32;
    }
}

mod ffi;

pub use ffi::Random;
