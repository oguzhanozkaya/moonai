use clap::Parser;

#[derive(Debug, Parser)]
#[command(author, version, about)]
pub struct CliArgs {
    #[arg(short, long, default_value = "config.lua")]
    pub config: String,
    #[arg(short = 'n', long)]
    pub steps: Option<u64>,
    #[arg(long)]
    pub headless: bool,
    #[arg(short, long)]
    pub verbose: bool,
    #[arg(long)]
    pub experiment: Option<String>,
    #[arg(long)]
    pub all: bool,
    #[arg(long)]
    pub list: bool,
    #[arg(long)]
    pub name: Option<String>,
    #[arg(long)]
    pub validate: bool,
}
