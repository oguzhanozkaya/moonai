use crate::random_ffi;

pub struct Random {
    ptr: *mut random_ffi::CRandom,
}

impl Random {
    pub fn new(seed: i32) -> Self {
        let ptr = unsafe { random_ffi::c_random_create(seed) };
        Self { ptr }
    }

    pub fn next_int(&mut self, min: i32, max: i32) -> i32 {
        unsafe { random_ffi::c_random_next_int(self.ptr, min, max) }
    }

    pub fn next_float(&mut self, min: f32, max: f32) -> f32 {
        unsafe { random_ffi::c_random_next_float(self.ptr, min, max) }
    }

    pub fn next_bool(&mut self, probability: f32) -> bool {
        unsafe { random_ffi::c_random_next_bool(self.ptr, probability) }
    }

    pub fn weighted_select(&mut self, weights: &[f32]) -> i32 {
        let len = weights.len();
        let ptr = weights.as_ptr();
        unsafe { random_ffi::c_random_weighted_select(self.ptr, ptr, len) }
    }
}

impl Drop for Random {
    fn drop(&mut self) {
        unsafe { random_ffi::c_random_destroy(self.ptr) }
    }
}
