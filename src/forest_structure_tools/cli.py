"""Command-line interface for forest structure tools."""

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Forest Structure Tools - LiDAR analysis for forest metrics."""
    console.print("[bold green]🌲 Forest Structure Tools[/bold green]")


@cli.command()
def dummy():
    console.print("[bold]DUMMY COMMAND[/bold]")


if __name__ == "__main__":
    cli()
