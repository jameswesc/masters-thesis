# Masters Thesis

This repo contains the code used for data processing as part of my Masters of Environmental Geospatial Science at the University of Tasmania. This is hosted online for anyone who wishes to dig in, but it has no effort has been made to present it in an accessible way.

## ALS Metrics Script

The [metrics.py](./forest_structure_tools/metrics.py) Python function is presented as a proof of concept for generating a wide variety of structural metris from ALS data. It expects input data to be as a structured numpy array (e.g. from PDAL) and returns a set of a grid and voxels metrics as an Xarray dataset.

As mentioned, this is presented as a proof of concept and there is no intention to maintain or support it.

## Notebooks

Notebooks used for data processing ate available in the [notebooks](./notebooks/) directory.

## Data

Output CSVs are stored in this repository under [csvs](./csvs/). Lidar data and grid/voxel data are not stored here. Please get in contact for access.
