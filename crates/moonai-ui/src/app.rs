use moonai_config::UiConfig;

pub struct App;

impl App {
    pub fn new(_ui_config: &UiConfig) -> anyhow::Result<Self> {
        Ok(Self)
    }

    pub fn run(&mut self) {}
}
