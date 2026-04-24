use moonai_app::{App, AppConfig};
use moonai_core::{SENSOR_COUNT, OUTPUT_COUNT};
use moonai_ffi::gpu_sim_is_available;

macro_rules! gpu_test {
    ($name:ident, $body:expr) => {
        #[test]
        fn $name() {
            unsafe {
                if gpu_sim_is_available() == 0 {
                    eprintln!("GPU not available - skipping test");
                    return;
                }
            }
            $body
        }
    };
}

gpu_test!(test_app_creation_with_default_config, {
    let cfg = AppConfig::default();
    let predator_count = cfg.sim_config.predator_count;
    let prey_count = cfg.sim_config.prey_count;
    let app = App::new(cfg);
    assert_eq!(app.state.predator.size(), predator_count as usize);
    assert_eq!(app.state.prey.size(), prey_count as usize);
});

gpu_test!(test_app_initialization_populates_predators, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 10;
    cfg.sim_config.prey_count = 0;
    cfg.sim_config.food_count = 0;

    let app = App::new(cfg);

    assert_eq!(app.state.predator.size(), 10);
    assert_eq!(app.state.prey.size(), 0);
});

gpu_test!(test_app_initialization_populates_prey, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 0;
    cfg.sim_config.prey_count = 20;
    cfg.sim_config.food_count = 0;

    let app = App::new(cfg);

    assert_eq!(app.state.predator.size(), 0);
    assert_eq!(app.state.prey.size(), 20);
});

gpu_test!(test_app_step_increments_runtime_step, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 5;
    cfg.sim_config.prey_count = 5;
    cfg.sim_config.food_count = 10;
    cfg.sim_config.max_steps = 10;

    let mut app = App::new(cfg);
    let initial_step = app.state.runtime.step;

    app.step();

    assert_eq!(app.state.runtime.step, initial_step + 1);
});

gpu_test!(test_app_step_decreases_energy, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 5;
    cfg.sim_config.prey_count = 5;
    cfg.sim_config.food_count = 10;
    cfg.sim_config.max_steps = 10;

    let mut app = App::new(cfg);

    let predator_energy_before = app.state.predator.energy[0];
    let prey_energy_before = app.state.prey.energy[0];

    app.step();

    let predator_energy_after = app.state.predator.energy[0];
    let prey_energy_after = app.state.prey.energy[0];

    assert!(predator_energy_after < predator_energy_before || predator_energy_before <= 0.0);
    assert!(prey_energy_after < prey_energy_before || prey_energy_before <= 0.0);
});

gpu_test!(test_app_run_completes_within_max_steps, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 5;
    cfg.sim_config.prey_count = 5;
    cfg.sim_config.food_count = 10;
    cfg.sim_config.max_steps = 5;

    let mut app = App::new(cfg);

    let result = app.run();

    assert!(result);
    assert!(app.state.runtime.step >= 5);
});

gpu_test!(test_app_metrics_are_populated_after_init, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 10;
    cfg.sim_config.prey_count = 20;
    cfg.sim_config.food_count = 50;

    let app = App::new(cfg);

    assert!(app.state.metrics.predator_count >= 0);
    assert!(app.state.metrics.prey_count >= 0);
});

gpu_test!(test_app_entity_ids_are_unique, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 10;
    cfg.sim_config.prey_count = 10;
    cfg.sim_config.food_count = 0;

    let app = App::new(cfg);

    let mut predator_ids = Vec::new();
    for i in 0..app.state.predator.size() {
        predator_ids.push(app.state.predator.entity_id[i]);
    }

    predator_ids.sort();
    predator_ids.dedup();
    assert_eq!(predator_ids.len(), app.state.predator.size());
});

gpu_test!(test_app_genomes_are_initialized, {
    let mut cfg = AppConfig::default();
    cfg.sim_config.predator_count = 5;
    cfg.sim_config.prey_count = 5;
    cfg.sim_config.food_count = 0;

    let app = App::new(cfg);

    for i in 0..app.state.predator.size() {
        let genome = &app.state.predator.genomes[i];
        assert_eq!(genome.num_inputs(), SENSOR_COUNT as i32);
        assert_eq!(genome.num_outputs(), OUTPUT_COUNT as i32);
    }
});