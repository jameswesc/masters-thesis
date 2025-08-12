import numpy as np
import numpy.typing as npt

import xarray as xr

from typing import Literal, Optional, Tuple, List

Op = Literal["at", "above", "below", "above_inc", "below_inc", "inside", "inside_inc"]
PercentageConfig = Tuple[Op, int, Optional[int]]
Percentages = List[PercentageConfig]


def z_metrics_xr(
    z: npt.NDArray[np.floating],
    include_basic_metrics=True,
    percentiles: npt.NDArray[np.integer] = None,
    percentages: Percentages | None = None,
    zbin_size: float | None = None,
    k: float = 1,
    weights: npt.NDArray[np.number] = None,
    suffix: str | None = None,
):
    metrics, coords = z_metrics(
        z,
        include_basic_metrics=include_basic_metrics,
        percentiles=percentiles,
        percentages=percentages,
        weights=weights,
        zbin_size=zbin_size,
        k=k,
    )

    return xr.Dataset(
        data_vars=metrics,
        coords=coords,
        attrs={
            "suffix": suffix,
            "zbin_size": zbin_size,
            "weighted": (weights is not None),
            "k": k,
        },
    )


def z_metrics(
    z: npt.NDArray[np.floating],
    include_basic_metrics=True,
    percentiles: npt.NDArray[np.integer] = None,
    percentages: Percentages | None = None,
    weights: npt.NDArray[np.number] = None,
    zbin_size: float | None = None,
    k: float = 1,
) -> dict[str, float]:
    metrics = {}
    coords = {}

    if include_basic_metrics:
        metrics |= basic_metrics(z)

    if percentiles is not None:
        metrics |= percentile_metrics(z, percentiles)

    if percentages is not None:
        metrics |= percentage_metrics(z, percentages, weights=weights)

    if zbin_size is not None:
        m, c = z_bin_metrics(z, zbin_size, k=k, weights=weights)
        metrics |= m
        coords |= c

    return (metrics, coords)


def basic_metrics(z: npt.NDArray[np.floating]):
    max = z.max()
    min = z.min()
    range = max - min
    mean = z.mean()
    median = np.median(z)
    sd = z.std()
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


def percentile_metrics(z: npt.NDArray[np.floating], percentiles: npt.ArrayLike):
    percentile_metrics = {}
    percentile_values = np.percentile(z, percentiles)

    for q, val in zip(percentiles, percentile_values):
        percentile_metrics[f"q{q}_h"] = val

    return percentile_metrics


def percentage_metrics(
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


def z_bin_metrics(
    z: npt.NDArray[np.floating],
    zbin_size: float,
    k=1,
    weights: npt.NDArray[np.floating] | None = None,
):

    if weights is None:
        weights = np.ones(len(z))

    total = weights.sum()

    bins = np.arange(0, z.max() + zbin_size, zbin_size)
    bin_indices = np.digitize(z, bins, right=True)

    inside = np.bincount(bin_indices, weights=weights, minlength=len(bins)).astype(
        float
    )
    inside[inside == 0] = np.nan
    inside_p = inside / total * 100
    entries = inside.cumsum()
    # No values exit the ground
    # Use nan to avoid infinities
    exits = np.concat(([np.nan], entries[:-1]))
    ppi = exits / entries

    vad = -np.log(ppi) * (1 / k) * (1 / zbin_size)

    vai = np.nansum(vad)

    attrs = {"zbin_size": zbin_size}

    metrics = {
        "zbin_inside": ("z", inside, attrs),
        "zbin_inside%": ("z", inside_p, attrs),
        "zbin_entries": ("z", entries, attrs),
        "zbin_exits": ("z", exits, attrs),
        "zbin_ppi": ("z", ppi, attrs),
        "zbin_vad": ("z", vad, {"zbin_size": zbin_size, "k": k}),
        "vai": vai,
    }

    coords = {"z": bins}

    return (metrics, coords)
