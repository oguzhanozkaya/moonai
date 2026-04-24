mod app;

use clap::Parser;
use moonai_core::{load_all_configs_lua, validate_config, SimulationConfig};
use std::path::Path;

#[derive(Parser, Debug)]
#[command(name = "MoonAI")]
#[command(about = "Predator-Prey Evolutionary Simulation")]
struct Args {
    #[arg(short = 'c', long = "config", default_value = "config.lua")]
    config_path: String,

    #[arg(short = 'n', long = "steps")]
    max_steps: Option<i32>,

    #[arg(long = "headless")]
    headless: bool,

    #[arg(short = 'v', long = "verbose")]
    verbose: bool,

    #[arg(long = "experiment")]
    experiment: Option<String>,

    #[arg(long = "all")]
    all: bool,

    #[arg(long = "list")]
    list: bool,

    #[arg(long = "name")]
    name: Option<String>,

    #[arg(long = "validate")]
    validate: bool,
}

fn run_experiment(
    name: &str,
    config: &SimulationConfig,
    headless: bool,
    max_steps_override: Option<i32>,
    run_name: Option<&str>,
) -> bool {
    let mut cfg = config.clone();
    if let Some(steps) = max_steps_override {
        cfg.max_steps = steps;
    }

    let app_cfg = app::AppConfig {
        sim_config: cfg,
        experiment_name: name.to_string(),
        headless,
        speed_multiplier: 1,
        run_name_override: run_name.map(String::from),
    };

    let mut app = app::App::new(app_cfg);
    app.run()
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(if true {
        "debug"
    } else {
        "info"
    }))
    .init();

    let args = Args::parse();

    let config_path = Path::new(&args.config_path);
    let configs = load_all_configs_lua(config_path);
    if configs.is_empty() {
        eprintln!("Error: No configs loaded from '{}'", args.config_path);
        std::process::exit(1);
    }

    if args.list {
        for name in configs.keys() {
            println!("{}", name);
        }
        return;
    }

    if args.validate {
        let config = if let Some(ref exp) = args.experiment {
            if let Some(cfg) = configs.get(exp) {
                cfg.clone()
            } else {
                eprintln!("Error: Experiment '{}' not found", exp);
                std::process::exit(1);
            }
        } else {
            configs.values().next().unwrap().clone()
        };

        let mut config = config;
        if let Some(steps) = args.max_steps {
            config.max_steps = steps;
        }

        let errors = validate_config(&config);
        if errors.is_empty() {
            println!("OK");
        } else {
            for error in &errors {
                eprintln!("ERROR [{}]: {}", error.field, error.message);
            }
            std::process::exit(1);
        }
        return;
    }

    if args.all {
        let mut failures = 0;
        for (name, config) in &configs {
            if !run_experiment(
                name,
                config,
                args.headless,
                args.max_steps,
                args.name.as_deref(),
            ) {
                failures += 1;
            }
        }
        std::process::exit(if failures == 0 { 0 } else { 1 });
    }

    let selected = if let Some(ref exp) = args.experiment {
        exp.clone()
    } else {
        let mut iter = configs.keys();
        let first = iter.next().unwrap().clone();
        if configs.len() > 1 {
            log::warn!(
                "Multiple experiments found; using '{}'. Use --experiment to select.",
                first
            );
        }
        first
    };

    let config = if let Some(cfg) = configs.get(&selected) {
        cfg
    } else {
        eprintln!("Error: Experiment '{}' not found.", selected);
        std::process::exit(1);
    };

    let success = run_experiment(
        &selected,
        config,
        args.headless,
        args.max_steps,
        args.name.as_deref(),
    );

    std::process::exit(if success { 0 } else { 1 });
}