pub mod connection;
pub mod node;
pub mod vec2;

pub use connection::ConnectionGene;
pub use node::{NodeGene, NodeType};
pub use vec2::Vec2;

pub const SENSOR_COUNT: usize = 35;
pub const OUTPUT_COUNT: usize = 2;
pub const INVALID_ENTITY: u32 = u32::MAX;

pub fn deterministic_respawn(seed: u64, tick: u64, entity_id: u32) -> (f32, f32) {
    let mut h = seed;
    h = h.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add((tick << 32) | entity_id as u64);
    h ^= h >> 30;
    h = h.wrapping_mul(0xbf58476d1ce4e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d049bb133111eb);
    h ^= h >> 31;
    let x = (h & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
    let y = ((h >> 32) & 0xFFFF_FFFF) as f32 / u32::MAX as f32;
    (x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_respawn_determinism() {
        let (x1, y1) = deterministic_respawn(42, 100, 5);
        let (x2, y2) = deterministic_respawn(42, 100, 5);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn deterministic_respawn_different_ticks() {
        let (x1, y1) = deterministic_respawn(42, 100, 5);
        let (x2, y2) = deterministic_respawn(42, 101, 5);
        assert_ne!(x1, x2);
        assert_ne!(y1, y2);
    }

    #[test]
    fn deterministic_respawn_different_entities() {
        let (x1, y1) = deterministic_respawn(42, 100, 5);
        let (x2, y2) = deterministic_respawn(42, 100, 6);
        assert_ne!(x1, x2);
        assert_ne!(y1, y2);
    }

    #[test]
    fn deterministic_respawn_range() {
        for _ in 0..100 {
            let (x, y) = deterministic_respawn(42, 100, 5);
            assert!((0.0..=1.0).contains(&x));
            assert!((0.0..=1.0).contains(&y));
        }
    }
}
