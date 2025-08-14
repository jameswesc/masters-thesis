import numpy as np
import numpy.typing as npt

import xarray as xr

from .utils import add_suffix
from .typing import Percentages, Suffix


default_percentiles = np.arange(10, 100, 10)
default_percentages: Percentages = [
    ("at", 0),
    ("above", 1.5),
    ("inside", 0, 1),
    ("inside", 1, 5),
    ("inside", 5, 10),
    ("inside", 10, 30),
    ("above", 30),
]


def forest_structure_metrics(
    z: npt.NDArray,
    x: npt.NDArray | None = None,
    y: npt.NDArray | None = None,
    weights: npt.NDArray | None = None,
    xy_bin_size: float | None = None,
    z_bin_size: float | None = 1,
    include_basic=True,
    percentiles: npt.NDArray[np.integer] = default_percentiles,
    percentages: Percentages | None = default_percentages,
    suffix: Suffix | None = None,
    skip_encodings=False,
):
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

    # float32 is enough for all my datavars
    for name, var in metrics_ds.data_vars.items():
        if var.dtype == "float64":
            metrics_ds[name] = var.astype("float32")

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
    with np.errstate(divide="ignore", invalid="ignore"):
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

    # Index of the first non nan value of inside
    first_valid_value = np.argmax(~np.isnan(inside))
    entries = np.nancumsum(inside)
    entries[:first_valid_value] = np.nan

    entries_pct = entries / total * 100

    # No values exit the ground
    # Use nan to avoid infinities
    exits = np.concat(([np.nan], entries[:-1]))
    ppi = exits / entries

    vad = -np.log(ppi) / k
    vad_norm = vad / z_bin_size
    vai = np.nansum(vad)

    fhd = -np.sum(inside_p * np.log(inside_p))
    norm_fhd = fhd / len(inside_p)

    # Compute VAI profiles
    vai_profile = np.zeros(len(bins))
    vai_slice = np.zeros(len(bins))

    for i, threshold in enumerate(bins):
        count_lte = weights[z <= threshold].sum()
        if count_lte > 0:
            vai_profile[i] = -np.log(count_lte / total) * (1 / k)
        else:
            vai_profile[i] = np.nan

    vai_slice[0] = np.nan
    for i, vai in enumerate(vai_profile[:-1]):
        vai_above = vai_profile[i + 1]
        vai_slice[i + 1] = vai - vai_above

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "inside": ("z", inside),
        "inside_pct": ("z", inside_p * 100),
        "entries": ("z", entries),
        "entries_pct": ("z", entries_pct),
        "exits": ("z", exits),
        "ppi": ("z", ppi),
        "vad_norm": ("z", vad_norm),
        "vad": ("z", vad),
        "vai_profile": ("z", vai_profile),
        "vai_slice": ("z", vai_slice),
        "vai": vai,
        "fhd": fhd,
        "norm_fhd": norm_fhd,
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
            metrics[f"pct_at_{a}m"] = weights[z == a].sum() / total * 100
        elif op == "above":
            metrics[f"pct_gt_{a}m"] = weights[z > a].sum() / total * 100
        elif op == "above_inc":
            metrics[f"pct_gte_{a}m"] = weights[z >= a].sum() / total * 100
        elif op == "below":
            metrics[f"pct_lt_{a}m"] = weights[z < a].sum() / total * 100
        elif op == "below_inc":
            metrics[f"pct_lte_{a}m"] = weights[z <= a].sum() / total * 100
        elif op == "inside":
            metrics[f"pct_inside_({a},{b}m]"] = (
                weights[(z > a) & (z <= b)].sum() / total * 100
            )
        elif op == "inside_inc":
            metrics[f"pct_inside_[{a},{b}m]"] = (
                weights[(z >= a) & (z <= b)].sum() / total * 100
            )

    return metrics
