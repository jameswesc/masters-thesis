# Forest Structure Tools

## Typical Processing Pipeline

1. Create sites GeoJSON
2. Process lidar at site level. This includes a reprojection, clip to site boundary, calculate height above ground, removing noise, and saving as a cloud optimised point cloud (COPC).
3. Further site processing that isn't  automated (e.g. removing cloud noise).
4. Processing lidar at a plot level. This includes clipping to each plot, loading height above ground into Z dimension, setting points with Z < 0 as ground, and adding a weight dimension.
5. Deriving plot level datasets. This includes rasters, height profiles and voxels.