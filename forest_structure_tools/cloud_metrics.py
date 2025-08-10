import numpy as np
from .utils import with_suffix

# from scipy.stats import kurtosis, skew


# Cloud metrics assumes that all points with height 0 are ground points
@with_suffix
def height_metrics(z: np.ndarray):
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

    height_metrics = {
        "max_h": max,
        "min_h": min,
        "range_h": range,
        "mean_h": mean,
        "median_h": median,
        "sd_h": sd,
        "var_h": var,
        "cv_h": cv,
        # "skew": skew,
        # "kurt": kurt,
    }

    return height_metrics


@with_suffix
def percentile_metrics(z: np.ndarray, percentiles=np.arange(10, 100, 10)):
    percentile_metrics = {}

    percentile_values = np.percentile(z, percentiles).astype(np.float32)
    for p, val in zip(percentiles, percentile_values):
        percentile_metrics[f"p{p}_h"] = val

    return percentile_metrics


@with_suffix
def cover_metrics(z: np.ndarray, weights: np.ndarray | None = None, cutoffs=[2]):
    if weights is None:
        zw = np.ones(z.shape)
    else:
        zw = weights

    total = zw.sum()

    cover_metrics = {}

    # If ground poitns are included calculate proportion ground
    if z.min() == 0:
        cover_metrics["ground_%"] = zw[z == 0].sum() / total * 100

    for cutoff in cutoffs:
        cover_metrics[f"above_{cutoff}m_%"] = zw[z > cutoff].sum() / total * 100

    return cover_metrics


@with_suffix
def relative_height_profile_metrics(
    z: np.ndarray,
    bins=10,
    weights: np.ndarray | None = None,
):
    metrics = {}

    total = len(z)
    if weights is not None:
        total = weights.sum()

    (counts, _) = np.histogram(z, bins=bins, weights=weights)
    (densities, _) = np.histogram(z, bins=bins, weights=weights, density=True)

    bounds = np.linspace(0, 100, bins + 1).astype(int)
    labels = [f"{lower}-{upper}%" for lower, upper in zip(bounds, bounds[1:])]

    for label, count, density in zip(labels, counts, densities):
        metrics[f"prp_{label}"] = count / total
        metrics[f"dns_{label}"] = density

    return metrics
