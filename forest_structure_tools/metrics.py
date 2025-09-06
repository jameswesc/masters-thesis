import numpy as np
import numpy.typing as npt

import xarray as xr

from scipy.stats import kurtosis, skew, entropy
from .gini import gini
from .cv import cv

default_percentiles = np.arange(10, 100, 10)


def forest_structure_metrics(
    z: npt.NDArray,
    x: npt.NDArray | None = None,
    y: npt.NDArray | None = None,
    weights: npt.NDArray | None = None,
    xy_bin_size: float | None = None,
    z_bin_size: float | None = None,
    include_basic=False,
    cover_threshold: float | None = None,
    percentiles: npt.NDArray[np.integer] = None,
):

    # First we create a basic dataset with only dimension is point index.
    # This datset will have the data variable z. It may also have x, y and weights
    point_ds_data_vars = {"z": ("point_idx", z)}

    if xy_bin_size is not None:
        if x is None or y is None:
            raise TypeError("x and y must be provided if xy_grain_size is provided")

        point_ds_data_vars["x"] = ("point_idx", x)
        point_ds_data_vars["y"] = ("point_idx", y)

    if weights is None:
        weights = np.ones(len(z))

    point_ds_data_vars["weights"] = ("point_idx", weights)
    points_ds = xr.Dataset(data_vars=point_ds_data_vars)

    # common keyword arguments for forest metrics
    forest_z_metrics_kwargs = {
        "include_basic": include_basic,
        "percentiles": percentiles,
        "z_bin_size": z_bin_size,
        "cover_threshold": cover_threshold,
    }

    if xy_bin_size is None:
        metrics_ds = forest_z_metrics_ds(points_ds, **forest_z_metrics_kwargs)
    else:
        # This algins the grid with the xy_bin_size
        # This ensures that datasets can be merged properly
        x_min = np.floor(x.min() / xy_bin_size) * xy_bin_size
        y_min = np.floor(y.min() / xy_bin_size) * xy_bin_size

        x_bins = np.arange(x_min, x.max() + xy_bin_size, xy_bin_size)
        y_bins = np.arange(y_min, y.max() + xy_bin_size, xy_bin_size)

        x_bin_grouper = xr.groupers.BinGrouper(
            bins=x_bins, labels=x_bins[:-1], include_lowest=True
        )
        y_bin_grouper = xr.groupers.BinGrouper(
            bins=y_bins, labels=y_bins[:-1], include_lowest=True
        )

        xy_grouping = points_ds.groupby(x=x_bin_grouper, y=y_bin_grouper)
        metrics_ds = xy_grouping.map(forest_z_metrics_ds, **forest_z_metrics_kwargs)
        metrics_ds = metrics_ds.rename({"x_bins": "x", "y_bins": "y"})

    metrics_ds.attrs["xy_bin_size"] = str(xy_bin_size)
    metrics_ds.attrs["z_bin_size"] = str(z_bin_size)

    return metrics_ds


def forest_z_metrics_ds(
    points_ds,
    include_basic=True,
    percentiles: npt.NDArray[np.integer] = None,
    z_bin_size: float | None = 1,
    cover_threshold: float | None = None,
):
    z = points_ds["z"].values
    weights = points_ds["weights"].values

    metrics = {}
    coords = {}

    veg_z = z[z > 0]

    if include_basic:
        metrics |= basic_veg_point_metrics(veg_z)

    if percentiles is not None and len(veg_z) > 0:
        metrics |= veg_point_percentiles(veg_z, percentiles=percentiles)

    if cover_threshold is not None:
        metrics |= cover_metrics(z, cover_threshold, weights=weights)

    if z_bin_size is not None:
        m, c = z_bin_metrics(z, z_bin_size, weights=weights)
        metrics |= m
        coords |= c

    return xr.Dataset(data_vars=metrics, coords=coords)


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


def cover_metrics(
    z: npt.NDArray[np.floating],
    t: float,
    weights: npt.NDArray[np.floating] | None = None,
):
    if len(z) == 0:
        return {}

    metrics = {}
    if weights is None:
        weights = np.ones(len(z))

    total = weights.sum()

    lgap = weights[z <= t].sum() / total
    lcover = (1 - lgap) * 100

    # If lgap is 0 (no penetration past threshold)
    # we cant calculate pai
    if lgap == 0:
        pai = np.nan
        pad = np.nan
    # If lgap is 1 (full penetration) then pai and pad are 0
    elif lgap == 1:
        pai = 0
        pad = 0
    else:
        pai = -np.log(lgap)
        pad = pai / z.max()

    metrics["lgap"] = lgap
    metrics["lcover"] = lcover
    metrics["pai"] = pai
    metrics["pad"] = pad

    return metrics


def z_bin_metrics(
    z: npt.NDArray[np.floating],
    z_bin_size: float,
    weights: npt.NDArray[np.floating] | None = None,
):
    if weights is None:
        weights = np.ones(len(z))

    total = weights.sum()

    # We use digitize over np.histogram because we want
    # the special case of the first bin being just 0
    bins = np.arange(0, z.max() + z_bin_size, z_bin_size)
    # right=True means [,0] -> 0, (0, bin_size] -> 1
    # i.e. z = 0 becomes index 0, z = 0.01 becomes index 1
    bin_indices = np.digitize(z, bins, right=True)

    # count # of returns inside each bin
    inside = np.bincount(bin_indices, weights=weights, minlength=len(bins)).astype(
        float
    )

    # Set any mising counts as nan instead of 0
    inside[inside == 0] = np.nan
    inside_p = inside / total

    # Index of the first non nan value of inside
    first_valid_value = np.argmax(~np.isnan(inside))
    entries = np.nancumsum(inside)
    entries[:first_valid_value] = np.nan

    # entries_pct = entries / total * 100

    # No values exit the ground
    # Use nan to avoid infinities
    exits = np.concat(([np.nan], entries[:-1]))
    lgap = exits / entries

    # Not in some literature k is used
    # however, its just a scalar so it can be applied in post if we want
    # similarly dz - though i think dz actually messes with the calculation
    # of pai and should not be used

    pai = -np.log(lgap)
    pad = pai / z_bin_size

    fhd = entropy(inside_p, nan_policy="omit")

    cv_inside_p = cv(inside_p)
    cv_lgap = cv(lgap)
    cv_pai = cv(pai)
    cv_pad = cv(pad)

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "inside_z": ("z", inside),
        "entries_z": ("z", entries),
        "exits_z": ("z", exits),
        "pct_inside_z": ("z", inside_p * 100),
        "lgap_z": ("z", lgap),
        "lcover_z": ("z", (1 - lgap) * 100),
        "pai_z": ("z", pai),
        "pad_z": ("z", pad),
        "fhd": fhd,
        "cv_pct_inside_z": cv_inside_p,
        "cv_lgap_z": cv_lgap,
        "cv_pai_z": cv_pai,
        "cv_pad_z": cv_pad,
    }

    coords = {"z": bins}

    return (metrics, coords)


# def z_percentage_metrics(
#     z: npt.NDArray[np.floating],
#     percentages: Percentages,
#     weights: npt.NDArray[np.floating] | None = None,
# ):
#     metrics = {}

#     if weights is None:
#         weights = np.ones(len(z))

#     total = weights.sum()

#     if weights is not None:
#         total = weights.sum()

#     for op, a, *rest in percentages:
#         b = rest[0] if rest else None
#         if op == "at":
#             metrics[f"pct_at_{a}m"] = weights[z == a].sum() / total * 100
#         elif op == "above":
#             metrics[f"pct_gt_{a}m"] = weights[z > a].sum() / total * 100
#         elif op == "above_inc":
#             metrics[f"pct_gte_{a}m"] = weights[z >= a].sum() / total * 100
#         elif op == "below":
#             metrics[f"pct_lt_{a}m"] = weights[z < a].sum() / total * 100
#         elif op == "below_inc":
#             metrics[f"pct_lte_{a}m"] = weights[z <= a].sum() / total * 100
#         elif op == "inside":
#             metrics[f"pct_inside_({a},{b}m]"] = (
#                 weights[(z > a) & (z <= b)].sum() / total * 100
#             )
#         elif op == "inside_inc":
#             metrics[f"pct_inside_[{a},{b}m]"] = (
#                 weights[(z >= a) & (z <= b)].sum() / total * 100
#             )

#     return metrics
