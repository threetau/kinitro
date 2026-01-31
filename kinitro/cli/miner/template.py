"""Template initialization for miner policies."""

import shutil
from pathlib import Path

import typer


def init_miner(
    output_dir: str = typer.Argument(".", help="Directory to create template in"),
):
    """
    Initialize a new miner policy from template.

    Creates the necessary files for building a policy container.
    """
    kinitro_root = Path(__file__).parent.parent.parent
    template_dir = kinitro_root / "miner" / "template"
    rl_interface_src = kinitro_root / "rl_interface.py"
    output_path = Path(output_dir)

    if not template_dir.exists():
        typer.echo("Template directory not found!", err=True)
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy template files (skip directories and pycache)
    for file in template_dir.iterdir():
        # Skip directories and pycache
        if file.is_dir() or file.name.startswith("__"):
            continue
        # Skip .pyc files
        if file.suffix == ".pyc":
            continue

        dest = output_path / file.name
        if dest.exists():
            typer.echo(f"Skipping {file.name} (already exists)")
        else:
            shutil.copy(file, dest)
            typer.echo(f"Created {file.name}")

    # Copy rl_interface.py from main kinitro package (single source of truth)
    rl_interface_dest = output_path / "rl_interface.py"
    if rl_interface_dest.exists():
        typer.echo("Skipping rl_interface.py (already exists)")
    else:
        shutil.copy(rl_interface_src, rl_interface_dest)
        typer.echo("Created rl_interface.py")

    typer.echo("\nMiner template initialized!")
    typer.echo("Next steps:")
    typer.echo("  1. Edit policy.py to implement your policy")
    typer.echo("  2. Add your model weights to the directory")
    typer.echo("  3. Test locally: uvicorn server:app --port 8001")
    typer.echo("  4. Upload to HuggingFace: huggingface-cli upload user/repo .")
    typer.echo("  5. Deploy to Basilica: kinitro miner push --repo user/repo --revision SHA")
    typer.echo(
        "  6. Or use one-command deploy: kinitro miner deploy -r user/repo -p . --netuid ..."
    )
