import numpy as np
import numpy.typing as npt

import xarray as xr

from scipy.stats import kurtosis, skew, entropy
from .typing import Percentages
from .gini import gini


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
    z_bin_count: float | None = 10,
    include_basic=True,
    percentiles: npt.NDArray[np.integer] = default_percentiles,
    percentages: Percentages | None = default_percentages,
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
        "z_bin_count": z_bin_count,
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
    percentages: Percentages | None = None,
    z_bin_size: float | None = 1,
    z_bin_count: float | None = 10,
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

    if z_bin_count is not None:
        m, c = qz_bin_metrics(z, z_bin_count, weights=weights)
        metrics |= m
        coords |= c

    return xr.Dataset(data_vars=metrics, coords=coords)


def cv(x: npt.NDArray) -> float:
    valid_x = x[~np.isnan(x)]
    if len(valid_x) >= 2 and valid_x.mean() != 0:
        return valid_x.std() / valid_x.mean()
    else:
        return np.nan


def basic_z_metrics(z: npt.NDArray[np.floating]):
    max = z.max()
    min = z.min()
    range = max - min
    mean = z.mean()
    median = np.median(z)
    sd = z.std()
    var = z.var()
    cv_val = cv(z)

    with np.errstate(divide="ignore", invalid="ignore"):
        crr = (mean - min) / range

    skew_val = skew(z)
    kurt_val = kurtosis(z)
    gini_val = gini(z)

    metrics = {
        "max": max,
        "min": min,
        "range": range,
        "mean": mean,
        "median": median,
        "sd": sd,
        "var": var,
        "cv": cv_val,
        "crr": crr,
        "skew": skew_val,
        "kurt": kurt_val,
        "gini": gini_val,
    }

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
    ppi = exits / entries

    # Not in some literature k is used
    # however, its just a scalar so it can be applied in post if we want
    # similarly dz - though i think dz actually messes with the calculation
    # of pai and should not be used
    pad = -np.log(ppi) / z_bin_size

    # Set ground to have 0 pulse penetration and 0 pad
    ppi[0] = 0
    pad[0] = 0

    pai = np.nansum(pad) * z_bin_size

    fhd = entropy(inside_p, nan_policy="omit")

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_fhd = fhd / np.log(((~np.isnan(inside_p)) & (inside_p > 0)).sum())

    cv_inside_p = cv(inside_p)
    cv_ppi = cv(ppi)
    cv_pad = cv(pad)

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "inside_pct": ("z", inside_p * 100),
        "ppi": ("z", ppi),
        "pad": ("z", pad),
        "pai": pai,
        "fhd": fhd,
        "norm_fhd": norm_fhd,
        "cv_inside_p": cv_inside_p,
        "cv_ppi": cv_ppi,
        "cv_pad": cv_pad,
    }

    coords = {"z": bins}

    return (metrics, coords)


def qz_bin_metrics(
    z: npt.NDArray[np.floating],
    z_bin_count: float = 20,
    weights: npt.NDArray[np.floating] | None = None,
):
    if weights is None:
        weights = np.ones(len(z))

    total = weights.sum()

    z_bin_size = z.max() / z_bin_count

    # We use digitize over np.histogram because we want
    # the special case of the first bin being just 0
    labels = np.linspace(0, 100, z_bin_count + 1)
    bins = np.linspace(0, z.max(), z_bin_count + 1)
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
    ppi = exits / entries

    # Not in some literature k is used
    # however, its just a scalar so it can be applied in post if we want
    # similarly dz - though i think dz actually messes with the calculation
    # of pai and should not be used
    pad = -np.log(ppi) / z_bin_size

    # Set ground to have 0 pulse penetration and 0 pad
    ppi[0] = 0
    pad[0] = 0

    pai = np.nansum(pad) * z_bin_size

    fhd = entropy(inside_p, nan_policy="omit")

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_fhd = fhd / np.log(((~np.isnan(inside_p)) & (inside_p > 0)).sum())

    cv_inside_p = cv(inside_p)
    cv_ppi = cv(ppi)
    cv_pad = cv(pad)

    # Tuple metrics mean they are along
    # dimension z (i.e. 1D metrics - an array)
    metrics = {
        "qz_inside_pct": ("qz", inside_p * 100),
        "qz_ppi": ("qz", ppi),
        "qz_pad": ("qz", pad),
        "qz_pai": pai,
        "qz_fhd": fhd,
        "qz_norm_fhd": norm_fhd,
        "qz_cv_inside_p": cv_inside_p,
        "qz_cv_ppi": cv_ppi,
        "qz_cv_pad": cv_pad,
    }

    coords = {"qz": labels}

    return (metrics, coords)


def z_percentile_metrics(z: npt.NDArray[np.floating], percentiles: npt.ArrayLike):
    percentile_metrics = {}
    percentile_values = np.percentile(z, percentiles)

    for q, val in zip(percentiles, percentile_values):
        if np.isnan(val):
            raise ValueError("Val is nan q{q}: {val}")

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
