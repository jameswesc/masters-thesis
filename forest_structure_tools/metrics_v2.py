import numpy as np
import numpy.typing as npt

import xarray as xr
import rioxarray

from scipy.stats import kurtosis, skew
from .gini import gini
from .cv import cv


def create_points_ds(points_array: npt.NDArray) -> xr.Dataset:
    # First we create a basic dataset with only dimension is point index.
    # This datset will have the data variables x, y, z and weights
    point_ds_data_vars = {
        "x": ("idx", points_array["X"]),
        "y": ("idx", points_array["Y"]),
        "z": ("idx", points_array["Z"]),
        "return_number": ("idx", points_array["ReturnNumber"]),
        "number_of_returns": ("idx", points_array["NumberOfReturns"]),
    }

    points_ds = xr.Dataset(data_vars=point_ds_data_vars)
    return points_ds


def create_xy_grouping(points_ds: xr.Dataset, xy_bin_size: float = 1):
    x = points_ds["x"]
    y = points_ds["y"]

    # This aligns the grid with the xy_bin_size. This is not really needed
    # but makes coordinates easier to read. Also easier to merge datasets
    x_min = np.floor(x.min() / xy_bin_size) * xy_bin_size
    y_min = np.floor(y.min() / xy_bin_size) * xy_bin_size

    # Define bins for x and y. Essentially edges of grid cells
    x_bins = np.arange(x_min, x.max() + xy_bin_size, xy_bin_size)
    y_bins = np.arange(y_min, y.max() + xy_bin_size, xy_bin_size)

    # Use center of each cell as the label
    x_labels = (x_bins[1:] + x_bins[:-1]) / 2
    y_labels = (y_bins[1:] + y_bins[:-1]) / 2

    # Define groupers for binning
    x_bin_grouper = xr.groupers.BinGrouper(
        bins=x_bins, labels=x_labels, include_lowest=True
    )
    y_bin_grouper = xr.groupers.BinGrouper(
        bins=y_bins, labels=y_labels, include_lowest=True
    )

    return points_ds.groupby(x=x_bin_grouper, y=y_bin_grouper)


def forest_structure_metrics(
    points: np.ndarray,
    xy_bin_size: float = 1,
    z_bin_size: float = 1,
):
    # Turn points into a xarray dataset so we can use groupby
    points_ds = create_points_ds(points)

    # Group the dataset into x and y bins
    xy_grouping = create_xy_grouping(points_ds, xy_bin_size=xy_bin_size)

    # Calculate forest metrics for each x,y bin
    metrics_ds = xy_grouping.map(
        forest_z_metrics_ds,
        z_bin_size=z_bin_size,
        global_z_max=points_ds["z"]
        .max()
        .item(),  # Passing z max in makes calulating vertical profile easier
    )

    # Rename dimensions to x and y instead of x_bins and y_bins
    metrics_ds = metrics_ds.rename({"x_bins": "x", "y_bins": "y"})

    metrics_ds.attrs["xy_bin_size"] = str(xy_bin_size)
    metrics_ds.attrs["z_bin_size"] = str(z_bin_size)

    # TODO - Un hardcode the CRS
    # Add coordinate reference system and spatial dimensions
    metrics_ds.rio.write_crs(7855, inplace=True)
    metrics_ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    metrics_ds.rio.write_coordinate_system(inplace=True)

    return metrics_ds


def forest_z_metrics_ds(
    points_ds,
    z_bin_size: float = 1,
    global_z_max: float | None = None,
):
    grid_metrics = grid_metric_vars(points_ds)
    voxel_metrics, voxel_z_coords = voxel_metric_vars(
        points_ds, z_bin_size, global_z_max=global_z_max
    )

    metrics = {**grid_metrics, **voxel_metrics}

    return xr.Dataset(data_vars=metrics, coords=voxel_z_coords)


# Given a column of points to be reduced to a single value
# Each column is a cell in the xy grid
def grid_metric_vars(points_ds: xr.Dataset):
    z = points_ds["z"].values
    rn = points_ds["return_number"].values
    weights = (1 / points_ds["number_of_returns"]).values

    fr_mask = rn == 1
    veg_mask = z > 0
    ground_mask = z == 0

    # Total number of points and pulses
    num_points = len(z)
    num_pulses = fr_mask.sum()

    # Cover metrics for whole veg profile
    total_weight = weights.sum()
    ground_count = ground_mask.sum()
    ground_weight = weights[ground_mask].sum()
    veg_count = veg_mask.sum()
    veg_weight = weights[veg_mask].sum()

    lgap = ground_count / num_points if num_points > 0 else np.nan
    lgap_weight = ground_weight / total_weight if num_points > 0 else np.nan

    lcapture = 1 - lgap if num_points > 0 else np.nan
    lcapture_weight = 1 - lgap_weight if num_points > 0 else np

    # Max, mean, median height use all points
    # 0 when highest point is ground
    # nan when no points
    max_height = z.max() if num_points > 0 else np.nan
    mean_height = z.mean() if num_points > 0 else np.nan
    median_height = np.median(z) if num_points > 0 else np.nan

    crr = mean_height / max_height if max_height > 0 else np.nan

    # Basic vertical complexity summary stats for veg points only
    # e.g. not interested in standard deviation if all points are ground
    # points (and thus have z of 0)
    veg_z = z[veg_mask]
    num_veg_points = len(veg_z)

    sd_veg_height = veg_z.std() if num_veg_points >= 2 else np.nan
    var_veg_height = veg_z.var() if num_veg_points >= 2 else np.nan
    cv_veg_height = cv(veg_z) if num_veg_points >= 2 else np.nan
    skew_veg_height = skew(veg_z) if num_veg_points >= 3 else np.nan
    kurt_veg_height = kurtosis(veg_z) if num_veg_points >= 4 else np.nan
    gini_veg_height = gini(veg_z) if num_veg_points >= 2 else np.nan

    # Veg height for every 10th percentile
    percentile_metrics = {
        f"q{p}_veg_height": (np.percentile(veg_z, p) if num_veg_points > 0 else np.nan)
        for p in np.arange(10, 100, 10)
    }

    return {
        "num_points": num_points,
        "num_pulses": num_pulses,
        "total_weight": total_weight,
        "ground_count": ground_count,
        "ground_weight": ground_weight,
        "veg_count": veg_count,
        "veg_weight": veg_weight,
        "lgap": lgap,
        "lgap_weight": lgap_weight,
        "lcapture": lcapture,
        "lcapture_weight": lcapture_weight,
        "max_height": max_height,
        "mean_height": mean_height,
        "median_height": median_height,
        "crr": crr,
        "sd_veg_height": sd_veg_height,
        "var_veg_height": var_veg_height,
        "cv_veg_height": cv_veg_height,
        "skew_veg_height": skew_veg_height,
        "kurt_veg_height": kurt_veg_height,
        "gini_veg_height": gini_veg_height,
        **percentile_metrics,
    }


# Given a column of points to be reduced to a single value
# Each column is reduced to a vertical profile of voxels
def voxel_metric_vars(
    points_ds: xr.Dataset,
    z_bin_size: float,
    global_z_max: float | None = None,
):
    z = points_ds["z"].values
    weights = (1 / points_ds["number_of_returns"]).values

    z_max = global_z_max if global_z_max is not None else z.max()

    # We use digitize over np.histogram because we want
    # the special case of the first bin being just 0
    bins = np.arange(0, z_max + z_bin_size, z_bin_size)

    # right=True means [,0] -> 0, (0, bin_size] -> 1
    # i.e. z = 0 becomes index 0, z = 0.01 becomes index 1
    bin_indices = np.digitize(z, bins, right=True)

    # count # of returns inside each bin
    inside_count = np.bincount(bin_indices, minlength=len(bins)).astype(float)
    inside_weight = np.bincount(
        bin_indices, weights=weights, minlength=len(bins)
    ).astype(float)

    # Count the pulses that enter each voxel
    enter_count = np.nancumsum(inside_count)
    enter_weight = np.nancumsum(inside_weight)

    # Count the pulses that exit each voxel
    exit_count = np.concat(([np.nan], enter_count[:-1]))
    exit_weight = np.concat(([np.nan], enter_weight[:-1]))

    # Some cases enters will be 0 and the result will be NaN
    # This is desired as the voxel has been fully occluded
    # Separate from 0 when inside is 0 but some have passed
    with np.errstate(divide="ignore", invalid="ignore"):
        lgap = exit_count / enter_count
        lgap_weight = exit_weight / enter_weight
        # Could also be 1 - lgap
        lcapture = inside_count / enter_count
        lcapture_weight = inside_weight / enter_weight

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "vox_inside_count": ("z", inside_count),
        "vox_enter_count": ("z", enter_count),
        "vox_exit_count": ("z", exit_count),
        "vox_inside_weight": ("z", inside_weight),
        "vox_enter_weight": ("z", enter_weight),
        "vox_exit_weight": ("z", exit_weight),
        "vox_lgap": ("z", lgap),
        "vox_lgap_weight": ("z", lgap_weight),
        "vox_lcapture": ("z", lcapture),
        "vox_lcapture_weight": ("z", lcapture_weight),
    }

    coords = {"z": bins}

    return (metrics, coords)
