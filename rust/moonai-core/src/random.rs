use std::f32::consts::PI;

const W: usize = 64;
const N: usize = 312;
const M: usize = 156;
const R: usize = 31;
const A: u64 = 0xB5026F5AA96619E9;
const U: usize = 29;
const S: usize = 17;
const B: u64 = 0x71D67FFFEDA60000;
const T: usize = 37;
const C: u64 = 0xFFF7EEE000000000;
const L: usize = 43;
const MASK_LOWER: u64 = (1u64 << R) - 1;
const MASK_UPPER: u64 = !MASK_LOWER;

pub struct Random {
    seed: i32,
    state: [u64; N],
    index: usize,
}

impl Clone for Random {
    fn clone(&self) -> Self {
        Self {
            seed: self.seed,
            state: self.state,
            index: self.index,
        }
    }
}

impl Random {
    pub fn new(seed: i32) -> Self {
        let mut state = [0u64; N];
        state[0] = seed as u64;
        for i in 1..N {
            state[i] = 6364136223846793005u64
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> (W - 2)))
                .wrapping_add(i as u64);
        }
        Self {
            seed,
            state,
            index: N,
        }
    }

    fn twist(&mut self) {
        for i in 0..N {
            let x = (self.state[i] & MASK_UPPER) | (self.state[(i + 1) % N] & MASK_LOWER);
            let mut x_a = x >> 1;
            if x & 1 == 1 {
                x_a ^= A;
            }
            self.state[i] = self.state[(i + M) % N] ^ x_a;
        }
        self.index = 0;
    }

    fn next_u64(&mut self) -> u64 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        y ^= y >> U;
        y ^= (y << S) & B;
        y ^= (y << T) & C;
        y ^= y >> L;
        self.index += 1;
        y
    }

    pub fn next_int(&mut self, min: i32, max: i32) -> i32 {
        let range = (max - min + 1) as u64;
        let r = self.next_u64() % range;
        min + r as i32
    }

    pub fn next_float(&mut self, min: f32, max: f32) -> f32 {
        let r = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        min + r * (max - min)
    }

    pub fn next_gaussian(&mut self, mean: f32, stddev: f32) -> f32 {
        let u1 = self.next_float(0.0, 1.0);
        let u2 = self.next_float(0.0, 1.0);
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + z0 * stddev
    }

    pub fn next_bool(&mut self, probability: f32) -> bool {
        self.next_float(0.0, 1.0) < probability
    }

    pub fn weighted_select(&mut self, weights: &[f32]) -> i32 {
        if weights.is_empty() {
            return -1;
        }
        let total: f32 = weights.iter().sum();
        if total <= 0.0 {
            return self.next_int(0, weights.len() as i32 - 1);
        }
        let r = self.next_float(0.0, total);
        let mut cumulative = 0.0f32;
        for (i, &w) in weights.iter().enumerate() {
            cumulative += w;
            if r <= cumulative {
                return i as i32;
            }
        }
        (weights.len() - 1) as i32
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_float_range() {
        let mut rng = Random::new(42);
        for _ in 0..100 {
            let v = rng.next_float(0.0, 1.0);
            assert!(v >= 0.0 && v < 1.0, "value {} out of range [0, 1)", v);
        }
    }

    #[test]
    fn test_next_bool() {
        let mut rng = Random::new(42);
        let true_count = (0..1000).filter(|_| rng.next_bool(0.5)).count();
        assert!(true_count > 400 && true_count < 600, "true_count = {}", true_count);
    }

    #[test]
    fn test_deterministic_sequence() {
        let mut rng1 = Random::new(42);
        let mut rng2 = Random::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_int(0, 1000), rng2.next_int(0, 1000));
        }
    }

    #[test]
    fn test_weighted_select_empty() {
        let mut rng = Random::new(42);
        let weights: Vec<f32> = vec![];
        let idx = rng.weighted_select(&weights);
        assert_eq!(idx, -1);
    }

    #[test]
    fn test_weighted_select() {
        let mut rng = Random::new(42);
        let weights = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32];
        for _ in 0..100 {
            let idx = rng.weighted_select(&weights);
            assert!(idx >= 0 && idx < 5);
        }
    }
}
