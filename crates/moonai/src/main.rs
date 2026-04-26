pub mod signal;

use std::collections::HashMap;

use anyhow::Result;
use clap::Parser as _;
use moonai_config::{validate_config, CliArgs, ConfigError, SimulationConfig};

fn resolve_config_path(args: &CliArgs) -> Option<String> {
    if let Some(ref path) = args.config {
        return Some(path.clone());
    }
    moonai_config::config_path_from_binary().map(|p| p.to_string_lossy().to_string())
}

fn load_experiments(config_path: &str) -> Result<HashMap<String, SimulationConfig>, ConfigError> {
    moonai_config::lua::load_config(config_path)
}

fn do_list(config_path: &str) -> Result<(), ConfigError> {
    let experiments = load_experiments(config_path)?;
    let mut names: Vec<_> = experiments.keys().collect();
    names.sort();
    println!("Available experiments:");
    for name in names {
        println!("  {}", name);
    }
    Ok(())
}

fn do_validate(config_path: &str, experiment_name: Option<&str>) -> Result<(), ConfigError> {
    let experiments = load_experiments(config_path)?;

    let config = if let Some(name) = experiment_name {
        match experiments.get(name) {
            Some(cfg) => cfg.clone(),
            None => {
                eprintln!("Error: experiment '{}' not found", name);
                std::process::exit(1);
            }
        }
    } else if let Some(cfg) = experiments.get("default") {
        cfg.clone()
    } else if experiments.len() == 1 {
        experiments.values().next().unwrap().clone()
    } else {
        eprintln!("Error: no 'default' experiment found and multiple experiments exist");
        std::process::exit(1);
    };

    match validate_config(&config) {
        Ok(()) => {
            println!("Configuration is valid.");
            Ok(())
        }
        Err(ConfigError::InvalidConfig(msg)) => {
            eprintln!("Configuration is invalid: {}", msg);
            std::process::exit(1);
        }
        Err(e) => Err(e),
    }
}

fn select_experiment(
    experiments: &HashMap<String, SimulationConfig>,
    name: Option<&str>,
) -> Result<SimulationConfig, ConfigError> {
    match name {
        Some(n) => experiments
            .get(n)
            .cloned()
            .ok_or_else(|| ConfigError::InvalidConfig(format!("experiment '{}' not found", n))),
        None => {
            if let Some(cfg) = experiments.get("default") {
                return Ok(cfg.clone());
            }
            if experiments.len() == 1 {
                return Ok(experiments.values().next().unwrap().clone());
            }
            Err(ConfigError::InvalidConfig(
                "no 'default' experiment found and multiple experiments exist without --experiment flag"
                    .to_string(),
            ))
        }
    }
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    let Some(config_path) = resolve_config_path(&args) else {
        eprintln!("Error: config.lua not found. Provide with --config or place next to binary.");
        std::process::exit(1);
    };

    if args.list {
        do_list(&config_path)?;
        return Ok(());
    }

    if args.validate {
        do_validate(&config_path, args.experiment.as_deref())?;
        return Ok(());
    }

    let experiments = load_experiments(&config_path)?;

    if args.all {
        if !args.headless {
            eprintln!("Error: --all requires --headless");
            std::process::exit(1);
        }
        let mut names: Vec<_> = experiments.keys().collect();
        names.sort();
        for name in names {
            let config = experiments.get(name).unwrap();
            let config = if let Some(steps) = args.steps {
                let mut cfg = config.clone();
                cfg.max_steps = steps;
                cfg
            } else {
                config.clone()
            };
            println!("Running experiment: {}", name);
            println!("Config: {:?}", config);
        }
        println!("All experiments completed.");
        return Ok(());
    }

    let config = select_experiment(&experiments, args.experiment.as_deref())?;
    let mut config = config;
    if let Some(steps) = args.steps {
        config.max_steps = steps;
    }

    println!("MoonAI - GPU-first predator-prey evolution simulation");
    println!(
        "Loaded experiment: {:?}",
        args.experiment.as_deref().unwrap_or("default")
    );
    println!("Config: {:?}", config);

    Ok(())
}