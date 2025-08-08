from forest_structure_tools.pdal_utils import process_pdal_pipelines_parallel
from pathlib import Path

CYCLE = "cycle-2"

PIPELINE_PATH = Path(__file__).parent / "pdal_pipelines" / "process_sites.json"

data_dir = Path(__file__).parents[1] / "data"
input_lidar_dir = data_dir / "source" / CYCLE

outputs_dir = data_dir / "outputs"
lidar_outputs_dir = outputs_dir / "sites" / CYCLE


def main():
    print("TODO")


if __name__ == "__main__":
    main()
