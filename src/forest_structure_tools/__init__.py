"""Forest Structure Tools - LiDAR analysis for forest metrics."""

from rich import print

# Import submodules for easy access
from . import create_sites
from . import process_lidar_data  
from . import generate_metrics

# Version info
__version__ = "0.1.0"

# Original hello functions for backwards compatibility
def hello():
    return "Hello, [bold magenta]World[/bold magenta]!", ":vampire:"


def say_hello():
    print(*hello())

# Expose main functionality
__all__ = [
    "create_sites",
    "process_lidar_data", 
    "generate_metrics",
    "hello",
    "say_hello"
]
