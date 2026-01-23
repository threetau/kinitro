"""Initialize miner template command."""

import typer


def init_miner(
    output_dir: str = typer.Argument(".", help="Directory to create template in"),
):
    """
    Initialize a new miner policy from template.

    Creates the necessary files for building a policy container.
    """
    import shutil
    from pathlib import Path

    template_dir = Path(__file__).parent.parent / "miner" / "template"
    output_path = Path(output_dir)

    if not template_dir.exists():
        typer.echo("Template directory not found!", err=True)
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy template files
    for file in template_dir.iterdir():
        dest = output_path / file.name
        if dest.exists():
            typer.echo(f"Skipping {file.name} (already exists)")
        else:
            shutil.copy(file, dest)
            typer.echo(f"Created {file.name}")

    typer.echo("\nMiner template initialized!")
    typer.echo("Next steps:")
    typer.echo("  1. Edit policy.py to implement your policy")
    typer.echo("  2. Add your model weights to the directory")
    typer.echo("  3. Test locally: uvicorn server:app --port 8001")
    typer.echo("  4. Upload to HuggingFace: huggingface-cli upload user/repo .")
    typer.echo("  5. Deploy to Chutes: kinitro chutes-push --repo user/repo --revision SHA")
    typer.echo("  6. Commit to chain: kinitro commit --repo ... --revision ... --chute-id ...")
