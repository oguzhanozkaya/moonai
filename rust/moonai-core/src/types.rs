

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn length(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

pub const INVALID_ENTITY: u32 = u32::MAX;

pub const SENSOR_COUNT: usize = 35;
pub const OUTPUT_COUNT: usize = 2;
