#[inline]
pub fn hash_u32(x: u32) -> u32 {
    let mut x = x;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

#[inline]
pub fn base_seed(seed: i32, step_index: i32, item_id: u32) -> u32 {
    (seed as u32)
        ^ ((step_index as u32 + 1).wrapping_mul(0x9e3779b9))
            .wrapping_add(item_id.wrapping_mul(0x85ebca6b))
}

#[inline]
pub fn unit_float(seed: u32) -> f32 {
    hash_u32(seed) as f32 / u32::MAX as f32
}

#[inline]
pub fn should_respawn(seed: i32, step_index: i32, item_id: u32, respawn_rate: f32) -> bool {
    unit_float(base_seed(seed, step_index, item_id)) < respawn_rate
}

#[inline]
pub fn respawn_x(seed: i32, step_index: i32, item_id: u32, world_width: f32) -> f32 {
    unit_float(base_seed(seed, step_index, item_id) ^ 0x68bc21eb) * world_width
}

#[inline]
pub fn respawn_y(seed: i32, step_index: i32, item_id: u32, world_height: f32) -> f32 {
    unit_float(base_seed(seed, step_index, item_id) ^ 0x02e5be93) * world_height
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_u32() {
        let h = hash_u32(42);
        assert_ne!(h, 42);
        assert_eq!(hash_u32(0), 0);
    }

    #[test]
    fn test_unit_float_range() {
        let seed = base_seed(42, 100, 0);
        let u = unit_float(seed);
        assert!(u >= 0.0 && u < 1.0);
    }

    #[test]
    fn test_should_respawn_never() {
        let result = should_respawn(42, 0, 0, 0.0);
        assert!(!result);
    }

    #[test]
    fn test_should_respawn_always() {
        let result = should_respawn(42, 0, 0, 1.0);
        assert!(result);
    }

    #[test]
    fn test_respawn_x_in_range() {
        let x = respawn_x(42, 100, 0, 1000.0);
        assert!(x >= 0.0 && x < 1000.0);
    }

    #[test]
    fn test_respawn_y_in_range() {
        let y = respawn_y(42, 100, 0, 500.0);
        assert!(y >= 0.0 && y < 500.0);
    }

    #[test]
    fn test_deterministic() {
        let x1 = respawn_x(42, 100, 0, 1000.0);
        let x2 = respawn_x(42, 100, 0, 1000.0);
        assert_eq!(x1, x2);

        let y1 = respawn_y(42, 100, 0, 1000.0);
        let y2 = respawn_y(42, 100, 0, 1000.0);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_different_seeds_different_results() {
        let x1 = respawn_x(42, 100, 0, 1000.0);
        let x2 = respawn_x(43, 100, 0, 1000.0);
        assert_ne!(x1, x2);
    }
}