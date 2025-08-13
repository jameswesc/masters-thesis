import numpy as np
import numpy.typing as npt

import xarray as xr

from .utils import  add_suffix
from .typing import Percentages, Suffix


default_percentiles = np.arange(10, 100, 10)
default_percentages: Percentages = [
    ("at", 0),
    ("above", 1.5),
    ("inside", 0, 1),
    ("inside", 1, 5),
    ("inside", 5, 10),
    ("inside", 10, 30),
    ("above", 30)
]


def forest_structure_metrics(
    z: npt.NDArray,
    x: npt.NDArray | None = None,
    y: npt.NDArray | None = None,
    weights: npt.NDArray | None = None,
    xy_bin_size: float | None = None,
    z_bin_size: float | None = None,
    include_basic=True,
    percentiles: npt.NDArray[np.integer] = default_percentiles,
    percentages: Percentages | None = default_percentages,
    suffix: Suffix | None = None,
):
    point_ds_data_vars = {"z": ("point_idx", z)}

    if xy_bin_size is not None:
        if x is None or y is None:
            raise TypeError("x and y must be provided if xy_grain_size is provided")

        point_ds_data_vars["x"] = ("point_idx", x)
        point_ds_data_vars["y"] = ("point_idx", y)

    was_weighted = weights is None  # For use in attributes later
    if weights is None:
        weights = np.ones(len(z))

    point_ds_data_vars["weights"] = (
        ("point_idx", weights) if weights is not None else np.ones(len(z))
    )

    points_ds = xr.Dataset(data_vars=point_ds_data_vars)

    forest_z_metrics_kwargs = {
        "include_basic": include_basic,
        "percentiles": percentiles,
        "percentages": percentages,
        "z_bin_size": z_bin_size,
        "suffix": suffix,
    }

    if xy_bin_size is None:
        metrics_ds = forest_z_metrics_ds(points_ds, **forest_z_metrics_kwargs)
    else:
        x_bins = np.arange(x.min(), x.max() + xy_bin_size, xy_bin_size)
        y_bins = np.arange(y.min(), y.max() + xy_bin_size, xy_bin_size)

        x_bin_grouper = xr.groupers.BinGrouper(
            bins=x_bins, labels=x_bins[:-1], include_lowest=True
        )
        y_bin_grouper = xr.groupers.BinGrouper(
            bins=y_bins, labels=y_bins[:-1], include_lowest=True
        )

        xy_grouping = points_ds.groupby(x=x_bin_grouper, y=y_bin_grouper)
        metrics_ds = xy_grouping.map(forest_z_metrics_ds, **forest_z_metrics_kwargs)
        metrics_ds = metrics_ds.rename({"x_bins": "x", "y_bins": "y"})

    metrics_ds.attrs["xy_bin_size"] = xy_bin_size
    metrics_ds.attrs["z_bin_size"] = z_bin_size
    metrics_ds.attrs["suffix"] = suffix
    metrics_ds.attrs["was_weighted"] = was_weighted

    return metrics_ds


def forest_z_metrics_ds(
    points_ds,
    include_basic=True,
    percentiles: npt.NDArray[np.integer] = None,
    percentages: Percentages | None = None,
    z_bin_size: float | None = None,
    suffix: Suffix | None = None,
):
    z = points_ds["z"].values
    weights = points_ds["weights"].values

    metrics = {}
    coords = {}

    if include_basic:
        metrics |= basic_z_metrics(z)

    if percentiles is not None:
        metrics |= z_percentile_metrics(z, percentiles=percentiles)

    if percentages is not None:
        metrics |= z_percentage_metrics(z, percentages=percentages, weights=weights)

    if z_bin_size is not None:
        m, c = z_bin_metrics(z, z_bin_size, weights=weights)
        metrics |= m
        coords |= c

    metrics = add_suffix(metrics, suffix)

    return xr.Dataset(data_vars=metrics, coords=coords)


def basic_z_metrics(z: npt.NDArray[np.floating]):
    max = z.max()
    min = z.min()
    range = max - min
    mean = z.mean()
    median = np.median(z)
    sd = z.std()
    # TODO - suppress expected warning when nans
    cv = sd / mean
    # skew = skew(veg)  - TODO check these do what you think they do
    # kurt = kurtosis(veg) - TODO check these do what you think they do
    var = z.var()

    metrics = {
        "max": max,
        "min": min,
        "range": range,
        "mean": mean,
        "median": median,
        "sd": sd,
        "var": var,
        "cv": cv,
        # "skew": skew,
        # "kurt": kurt,
    }

    return metrics


def z_bin_metrics(
    z: npt.NDArray[np.floating],
    z_bin_size: float,
    weights: npt.NDArray[np.floating] | None = None,
    k=1,
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
    entries = inside.cumsum()
    # No values exit the ground
    # Use nan to avoid infinities
    exits = np.concat(([np.nan], entries[:-1]))
    ppi = exits / entries

    vad = -np.log(ppi) * (1 / k) * (1 / z_bin_size)
    vai = np.nansum(vad)

    fhd =  - np.sum(inside_p * np.log(inside_p))
    norm_fhd = fhd / len(inside_p)

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "inside": ("z", inside),
        "inside_%": ("z", inside_p * 100),
        "entries": ("z", entries), 
        "exits": ("z", exits),
        "ppi": ("z", ppi),
        "vad": ("z", vad),
        "vai": vai,
        "fhd": fhd,
        "norm_fhd": norm_fhd
    }

    coords = {"z": bins}

    return (metrics, coords)


def z_percentile_metrics(z: npt.NDArray[np.floating], percentiles: npt.ArrayLike):
    percentile_metrics = {}
    percentile_values = np.percentile(z, percentiles)

    for q, val in zip(percentiles, percentile_values):
        percentile_metrics[f"q{q}_h"] = val

    return percentile_metrics


def z_percentage_metrics(
    z: npt.NDArray[np.floating],
    percentages: Percentages,
    weights: npt.NDArray[np.floating] | None = None,
):
    metrics = {}

    if weights is None:
        weights = np.ones(len(z))

    total = weights.sum()

    if weights is not None:
        total = weights.sum()

    for op, a, *rest in percentages:
        b = rest[0] if rest else None
        if op == "at":
            metrics[f"%at_{a}m"] = weights[z == a].sum() / total * 100
        elif op == "above":
            metrics[f"%gt_{a}m"] = weights[z > a].sum() / total * 100
        elif op == "above_inc":
            metrics[f"%gte_{a}m"] = weights[z >= a].sum() / total * 100
        elif op == "below":
            metrics[f"%lt_{a}m"] = weights[z < a].sum() / total * 100
        elif op == "below_inc":
            metrics[f"%lte_{a}m"] = weights[z <= a].sum() / total * 100
        elif op == "inside":
            metrics[f"%inside_({a},{b}m]"] = (
                weights[(z > a) & (z <= b)].sum() / total * 100
            )
        elif op == "inside_inc":
            metrics[f"%inside_[{a},{b}m]"] = (
                weights[(z >= a) & (z <= b)].sum() / total * 100
            )

    return metrics
