"""Command-line interface for forest structure tools."""

import click
from rich import print
from rich.console import Console

from . import create_sites, process_lidar_data, generate_metrics

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Forest Structure Tools - LiDAR analysis for forest metrics."""
    console.print("[bold green]🌲 Forest Structure Tools[/bold green]")


@cli.command()
@click.argument("name")
@click.option("--lat", type=float, required=True, help="Latitude coordinate")
@click.option("--lon", type=float, required=True, help="Longitude coordinate")
def create_site(name: str, lat: float, lon: float):
    """Create a new forest site for analysis."""
    create_sites.create_site(name, (lat, lon))


@cli.command()
def list_sites():
    """List all available forest sites."""
    create_sites.list_sites()


@cli.command()
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--height-threshold", default=2.0, help="Minimum height for vegetation (m)")
def process_lidar(file_path: str, height_threshold: float):
    """Process LiDAR data file."""
    # Process the point cloud
    results = process_lidar_data.process_point_cloud(file_path)
    
    # Filter vegetation
    filtered = process_lidar_data.filter_vegetation(results, height_threshold)
    
    console.print(f"✅ Processing complete: {filtered['filtered_points']:,} vegetation points")


@cli.command()
@click.option("--site-name", default="Example Site", help="Name of the forest site")
def generate_metrics_cmd(site_name: str):
    """Generate forest metrics and reports."""
    # Mock data for demonstration
    site_info = {"name": site_name, "latitude": 45.0, "longitude": -120.0}
    lidar_data = {"vegetation_points": 800000}
    
    # Calculate metrics
    height_metrics = generate_metrics.calculate_canopy_height(lidar_data)
    biomass = generate_metrics.calculate_biomass(height_metrics, site_info)
    
    # Generate report
    report = generate_metrics.generate_report(site_info, height_metrics, biomass)
    
    # Optionally save to file
    with open(f"{site_name.lower().replace(' ', '_')}_report.txt", "w") as f:
        f.write(report)
    
    console.print(f"📄 Report saved to: {site_name.lower().replace(' ', '_')}_report.txt")


@cli.command()
@click.argument("lidar_file", type=click.Path(exists=True))
@click.option("--site-name", required=True, help="Name of the forest site")
@click.option("--lat", type=float, required=True, help="Latitude coordinate")
@click.option("--lon", type=float, required=True, help="Longitude coordinate")
def full_analysis(lidar_file: str, site_name: str, lat: float, lon: float):
    """Run complete forest analysis pipeline."""
    console.print("[bold]🔄 Running Full Analysis Pipeline[/bold]")
    
    # Step 1: Create site
    site_info = create_sites.create_site(site_name, (lat, lon))
    
    # Step 2: Process LiDAR
    console.print("\n[bold]Step 2: Processing LiDAR Data[/bold]")
    lidar_results = process_lidar_data.process_point_cloud(lidar_file)
    filtered_data = process_lidar_data.filter_vegetation(lidar_results)
    
    # Step 3: Generate metrics
    console.print("\n[bold]Step 3: Generating Metrics[/bold]")
    height_metrics = generate_metrics.calculate_canopy_height(filtered_data)
    biomass = generate_metrics.calculate_biomass(height_metrics, site_info)
    
    # Step 4: Generate report
    console.print("\n[bold]Step 4: Generating Report[/bold]")
    report = generate_metrics.generate_report(site_info, height_metrics, biomass)
    
    report_file = f"{site_name.lower().replace(' ', '_')}_analysis.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    console.print(f"🎉 [bold green]Complete analysis saved to: {report_file}[/bold green]")


if __name__ == "__main__":
    cli()
