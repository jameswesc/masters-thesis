"""Site creation functionality for forest structure analysis."""

from rich import print
from rich.console import Console

console = Console()


def create_site(name: str, location: tuple[float, float]) -> dict:
    """Create a new forest site for analysis.
    
    Args:
        name: Name of the site
        location: Tuple of (latitude, longitude)
    
    Returns:
        Dictionary containing site information
    """
    site_info = {
        "name": name,
        "latitude": location[0],
        "longitude": location[1],
        "created": True
    }
    
    console.print(f"✅ Created site: [bold green]{name}[/bold green]")
    console.print(f"📍 Location: {location[0]}, {location[1]}")
    
    return site_info


def list_sites() -> list[dict]:
    """List all available forest sites."""
    # Placeholder implementation
    sites = [
        {"name": "Example Forest", "latitude": 45.0, "longitude": -120.0}
    ]
    
    console.print("[bold]Available Sites:[/bold]")
    for site in sites:
        console.print(f"  • {site['name']} ({site['latitude']}, {site['longitude']})")
    
    return sites
