import numpy as np
import numpy.typing as npt

import xarray as xr

from scipy.stats import kurtosis, skew, entropy
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

    # Passing z max in makes calulating vertical profile easier
    global_z_max = points_ds["z"].max().item()

    # Calculate forest metrics for each x,y bin
    metrics_ds = xy_grouping.map(
        forest_z_metrics_ds,
        xy_bin_size=xy_bin_size,
        z_bin_size=z_bin_size,
        global_z_max=global_z_max,
    )

    # Rename dimensions to x and y instead of x_bins and y_bins
    metrics_ds = metrics_ds.rename({"x_bins": "x", "y_bins": "y"})

    metrics_ds.attrs["xy_bin_size"] = str(xy_bin_size)
    metrics_ds.attrs["z_bin_size"] = str(z_bin_size)

    return metrics_ds


def forest_z_metrics_ds(
    points_ds,
    xy_bin_size: float = 1,
    z_bin_size: float = 1,
    global_z_max: float | None = None,
):
    grid_metrics = grid_metric_vars(points_ds, xy_bin_size)
    voxel_metrics, voxel_z_coords = voxel_metric_vars(
        points_ds, z_bin_size, global_z_max=global_z_max
    )

    metrics = {**grid_metrics, **voxel_metrics}

    return xr.Dataset(data_vars=metrics, coords=voxel_z_coords)


# Given a column of points to be reduced to a single value
# Each column is a cell in the xy grid
def grid_metric_vars(points_ds: xr.Dataset, xy_bin_size=1):
    z = points_ds["z"].values
    rn = points_ds["return_number"].values
    weights = (1 / points_ds["number_of_returns"]).values

    xy_area = xy_bin_size * xy_bin_size

    fr_mask = rn == 1
    veg_mask = z > 0

    # Total number of points and pulses
    num_points = len(z)
    num_pulses = fr_mask.sum()

    # Cover metrics for whole veg profile
    total_weight = weights.sum()

    canopy_cover_1m = (z > 1).sum() / num_points
    canopy_cover_1m_w = weights[z > 1].sum() / total_weight

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
        # Ancillary
        "point_density": num_points / xy_area,
        "pulse": num_pulses / xy_area,
        # Height
        "chm": max_height,
        "mean_height": mean_height,
        "median_height": median_height,
        **percentile_metrics,
        # Vertical Complexity
        "crr": crr,
        "sd_veg_height": sd_veg_height,
        "cv_veg_height": cv_veg_height,
        "skew_veg_height": skew_veg_height,
        "kurt_veg_height": kurt_veg_height,
        "gini_veg_height": gini_veg_height,
        # Cover
        "canopy_cover_1m": canopy_cover_1m,
        "canopy_cover_1m_w": canopy_cover_1m_w,
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

    total_count = len(z)
    total_weight = weights.sum()

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

    rel_density = inside_count / total_count
    rel_density_w = inside_weight / total_weight

    # Count the pulses that enter each voxel
    enter_count = np.nancumsum(inside_count)
    enter_weight = np.nancumsum(inside_weight)

    # Count the pulses that exit each voxel
    # exit_count = np.concat(([np.nan], enter_count[:-1]))
    # exit_weight = np.concat(([np.nan], enter_weight[:-1]))

    fhd = entropy(inside_count, nan_policy="omit")
    fhd_w = entropy(inside_weight, nan_policy="omit")

    num_bins = len(bins)

    vci = fhd / np.log(num_bins)
    vci_w = fhd_w / np.log(num_bins)

    # Some cases enters will be 0 and the result will be NaN
    # This is desired as the voxel has been fully occluded
    # Separate from 0 when inside is 0 but some have passed
    with np.errstate(divide="ignore", invalid="ignore"):
        # Could also be 1 - lgap
        capture = inside_count / enter_count
        capture_w = inside_weight / enter_weight

    # Calc relative capture
    rel_capture = capture / capture.sum()
    rel_capture_w = capture_w / capture_w.sum()

    # Entropy will normalise capture and capture w
    shann_capture = entropy(capture, nan_policy="omit")
    shann_capture_w = entropy(capture_w, nan_policy="omit")

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "vox_inside": ("z", inside_count),
        "vox_inside_w": ("z", inside_weight),
        "vox_rel_density": ("z", rel_density),
        "vox_rel_density_w": ("z", rel_density_w),
        "vox_enter": ("z", enter_count),
        "vox_enter_w": ("z", enter_weight),
        "vox_capture": ("z", capture),
        "vox_capture_w": ("z", capture_w),
        "vox_rel_capture": ("z", rel_capture),
        "vox_rel_capture_w": ("z", rel_capture_w),
        "fhd": fhd,
        "fhd_w": fhd_w,
        "vci": vci,
        "vci_w": vci_w,
        "shann_capture": shann_capture,
        "shann_capture_w": shann_capture_w,
        "norm_shann_capture": shann_capture / np.log(num_bins),
        "norm_shann_capture_w": shann_capture_w / np.log(num_bins),
    }

    coords = {"z": bins}

    return (metrics, coords)
