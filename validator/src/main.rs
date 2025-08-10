use anyhow::Result;
use clap::Command;

use constants::{ABOUT, BIN_NAME, NAME, VERSION};

mod config;
mod constants;
mod log;

fn main() -> Result<()> {
    let about_text = format!("{NAME} {VERSION}\n{ABOUT}");
    let after_help_text =
        format!("See '{BIN_NAME} help <command>' for more information on a command");

    let storb = Command::new("storb")
        .bin_name(BIN_NAME)
        .name(NAME)
        .version(VERSION)
        .about(about_text)
        .after_help(after_help_text)
        // .args(cli::args::common_args())
        .arg_required_else_help(true)
        // .subcommands(cli::builtin())
        .subcommand_required(true);

    let matches = storb.get_matches();
    match matches.subcommand() {
        _ => {}
    }

    Ok(())
}
