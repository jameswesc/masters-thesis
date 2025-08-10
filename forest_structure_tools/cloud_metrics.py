import numpy as np
from .utils import with_suffix
# from scipy.stats import kurtosis, skew

# Cloud metrics assumes that all points with height 0 are ground points
@with_suffix
def height_metrics(z: np.ndarray):

    z = z.astype(np.float32)    # float32 is enough for heights

    max = z.max()
    min = z.min()
    range = max - min
    mean = z.mean()
    median = np.median(z)

    height_metrics = {
        "max": max,
        "min": min,
        "range": range,
        "mean": mean,
        "median": median,
    }

    return height_metrics

@with_suffix
def quantile_metrics(z: np.ndarray, quantiles: np.ndarray):
    quantile_metrics = {}
    
    quantile_values = np.quantile(z, quantiles).astype(np.float32)
    for (q, val) in zip(quantiles, quantile_values):
        q_int = (q * 100).astype(int)
        quantile_metrics[f"q{q_int}"] = val
    
    return quantile_metrics


@with_suffix
def complexity_metrics(z: np.ndarray):
    z = z.astype(np.float32)    # float32 is enough for heights
    
    mean = z.mean()
    sd = z.std()
    cv = sd / mean
    # skew = skew(veg)  - TODO check these do what you think they do
    # kurt = kurtosis(veg) - TODO check these do what you think they do
    var = z.var()

    # TODO - gini coeff
    # TODO - entropy / FHD etc

    complexity_metrics = {
        "sd": sd,
        "cv": cv,
        # "skew": skew,
        # "kurt": kurt,
        "var": var,
    }

    return complexity_metrics




@with_suffix
def cover_metrics(z: np.ndarray, weights: np.ndarray | None = None, cutoffs = [2], density_bins = 0):
    z = z.astype(np.float32)

    if weights is None:
        zw = np.ones(z.shape)
    else:
        zw = weights

    total = zw.sum()

    cover_metrics = {}

    # If ground poitns are included calculate proportion ground
    if z.min() == 0: 
        cover_metrics["p_ground"] = zw[z == 0].sum() / total * 100
    
    for cutoff in cutoffs:
        cover_metrics[f"p_above_{cutoff}"] = zw[z > cutoff].sum() / total * 100
    
    if density_bins > 0:
        vals, _  = np.histogram(z, density_bins, weights=weights)
        pbounds = np.linspace(0, 100, density_bins + 1).astype(int)
        labels = [f"{lower}-{upper}" for lower, upper in zip(pbounds, pbounds[1:])]

        for val, label in zip(vals, labels):
            cover_metrics[f"d_{label}"] = val / total * 100

    return cover_metrics