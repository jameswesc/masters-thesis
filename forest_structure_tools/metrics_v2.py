import numpy as np
import numpy.typing as npt

import xarray as xr

from scipy.stats import kurtosis, skew, entropy
from .gini import gini
from .cv import cv

default_percentiles = np.arange(10, 100, 10)


def forest_structure_metrics(
    points: npt.NDArray,
    xy_bin_size: float | None = 1,
    z_bin_size: float = 1,
    percentiles: npt.NDArray[np.integer] | None = default_percentiles,
):

    # First we create a basic dataset with only dimension is point index.
    # This datset will have the data variables x, y, z and weights
    point_ds_data_vars = {
        "x": ("idx", points["X"]),
        "y": ("idx", points["Y"]),
        "z": ("idx", points["Z"]),
        "return_number": ("idx", points["ReturnNumber"]),
        "number_of_returns": ("idx", points["NumberOfReturns"]),
    }

    points_ds = xr.Dataset(data_vars=point_ds_data_vars)

    if xy_bin_size is None:
        metrics_ds = forest_z_metrics_ds(
            points_ds, percentiles=percentiles, z_bin_size=z_bin_size
        )
    else:
        x = points_ds["x"]
        y = points_ds["y"]
        # This algins the grid with the xy_bin_size
        # This ensures that datasets can be merged properly
        x_min = np.floor(x.min() / xy_bin_size) * xy_bin_size
        y_min = np.floor(y.min() / xy_bin_size) * xy_bin_size

        x_bins = np.arange(x_min, x.max() + xy_bin_size, xy_bin_size)
        y_bins = np.arange(y_min, y.max() + xy_bin_size, xy_bin_size)

        # TODO  - Make labels the center

        x_bin_grouper = xr.groupers.BinGrouper(
            bins=x_bins, labels=x_bins[:-1], include_lowest=True
        )
        y_bin_grouper = xr.groupers.BinGrouper(
            bins=y_bins, labels=y_bins[:-1], include_lowest=True
        )

        xy_grouping = points_ds.groupby(x=x_bin_grouper, y=y_bin_grouper)
        metrics_ds = xy_grouping.map(
            forest_z_metrics_ds, percentiles=percentiles, z_bin_size=z_bin_size
        )
        metrics_ds = metrics_ds.rename({"x_bins": "x", "y_bins": "y"})

    metrics_ds.attrs["xy_bin_size"] = str(xy_bin_size)
    metrics_ds.attrs["z_bin_size"] = str(z_bin_size)

    return metrics_ds


def forest_z_metrics_ds(
    points_ds,
    percentiles: npt.NDArray[np.integer] = None,
    z_bin_size: float | None = 1,
):
    z = points_ds["z"].values
    rn = points_ds["return_number"].values
    weights = (1 / points_ds["number_of_returns"]).values

    metrics = {}
    coords = {}

    veg_z = z[z > 0]

    metrics |= total_count_and_weight_metrics(z, rn, weights)
    metrics |= basic_veg_point_metrics(veg_z)

    if percentiles is not None and len(veg_z) > 0:
        metrics |= veg_point_percentiles(veg_z, percentiles=percentiles)

    if z_bin_size is not None:
        m, c = z_bin_metrics(z, z_bin_size, weights=weights)
        metrics |= m
        coords |= c

    return xr.Dataset(data_vars=metrics, coords=coords)


def total_count_and_weight_metrics(
    z: npt.NDArray[np.floating],
    rn: npt.NDArray[np.floating],
    weights: npt.NDArray[np.floating],
):
    fr_mask = rn == 1
    veg_mask = z > 0
    ground_mask = z == 0

    return {
        "num_points": len(z),
        "num_pulses": fr_mask.sum(),
        "total_count": len(z),
        "total_weight": weights.sum(),
        "ground_count": ground_mask.sum(),
        "ground_weight": weights[ground_mask].sum(),
        "veg_count": veg_mask.sum(),
        "veg_weight": weights[veg_mask].sum(),
    }


def basic_veg_point_metrics(veg_z: npt.NDArray[np.floating]):

    if len(veg_z) == 0:
        return {
            "max_veg": 0,
            "mean_veg": 0,
            "median_veg": 0,
            "sd_veg": np.nan,
            "var_veg": np.nan,
            "cv_veg": np.nan,
            "crr_veg": np.nan,
            "skew_veg": np.nan,
            "kurt_veg": np.nan,
            "gini_veg": np.nan,
        }

    max_veg = veg_z.max()
    mean_veg = veg_z.mean()
    median_veg = np.median(veg_z)
    sd_veg = veg_z.std()
    var_veg = veg_z.var()
    cv_veg = cv(veg_z)

    with np.errstate(divide="ignore", invalid="ignore"):
        crr_veg = mean_veg / max_veg

    gini_veg = gini(veg_z)

    if np.allclose(veg_z, veg_z[0]):
        skew_veg = np.nan
        kurt_veg = np.nan
    else:
        skew_veg = skew(veg_z)
        kurt_veg = kurtosis(veg_z)

    metrics = {
        "max_veg": max_veg,
        "mean_veg": mean_veg,
        "median_veg": median_veg,
        "sd_veg": sd_veg,
        "var_veg": var_veg,
        "cv_veg": cv_veg,
        "crr_veg": crr_veg,
        "skew_veg": skew_veg,
        "kurt_veg": kurt_veg,
        "gini_veg": gini_veg,
    }

    return metrics


def veg_point_percentiles(veg_z: npt.NDArray[np.floating], percentiles: npt.ArrayLike):

    percentile_metrics = {}
    percentile_values = np.percentile(veg_z, percentiles)

    for q, val in zip(percentiles, percentile_values):
        if np.isnan(val):
            raise ValueError("Val is nan q{q}: {val}")

        percentile_metrics[f"q{q}_veg"] = val

    return percentile_metrics


def z_bin_metrics(
    z: npt.NDArray[np.floating],
    z_bin_size: float,
    weights: npt.NDArray[np.floating],
):
    # We use digitize over np.histogram because we want
    # the special case of the first bin being just 0
    bins = np.arange(0, z.max() + z_bin_size, z_bin_size)

    # right=True means [,0] -> 0, (0, bin_size] -> 1
    # i.e. z = 0 becomes index 0, z = 0.01 becomes index 1
    bin_indices = np.digitize(z, bins, right=True)

    # count # of returns inside each bin
    inside_count = np.bincount(bin_indices, minlength=len(bins)).astype(float)
    inside_weight = np.bincount(
        bin_indices, weights=weights, minlength=len(bins)
    ).astype(float)

    # Set any mising counts as nan instead of 0 - why ?

    # inside[inside == 0] = np.nan
    # inside_p = inside / total

    # Index of the first non nan value of inside
    # first_valid_value = np.argmax(~np.isnan(inside))
    # entries = np.nancumsum(inside)
    # entries[:first_valid_value] = np.nan

    enter_count = np.nancumsum(inside_count)
    enter_weight = np.nancumsum(inside_weight)

    exit_count = np.concat(([np.nan], enter_count[:-1]))
    exit_weight = np.concat(([np.nan], enter_weight[:-1]))

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "inside_count": ("z", inside_count),
        "enter_count": ("z", enter_count),
        "exit_count": ("z", exit_count),
        "inside_weight": ("z", inside_weight),
        "enter_weight": ("z", enter_weight),
        "exit_weight": ("z", exit_weight),
    }

    coords = {"z": bins}

    return (metrics, coords)
