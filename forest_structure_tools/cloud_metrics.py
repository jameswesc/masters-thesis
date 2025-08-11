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
def percent_at(
    z: np.ndarray, threshold: float | list[float] = 2, weights: np.ndarray | None = None
):
    if weights is None:
        zw = np.ones(z.shape)
    else:
        zw = weights

    total = zw.sum()

    if not isinstance(threshold, list):
        threshold = [threshold]

    metrics = {}

    for t in threshold:
        metrics[f"%at_{t}m"] = zw[z == t].sum() / total * 100

    return metrics


@with_suffix
def percent_above(
    z: np.ndarray,
    threshold: float | list[float] = 2,
    weights: np.ndarray | None = None,
    inclusive=False,
):
    if weights is None:
        zw = np.ones(z.shape)
    else:
        zw = weights

    total = zw.sum()

    if not isinstance(threshold, list):
        threshold = [threshold]

    metrics = {}

    for t in threshold:
        label = f"gt{"e" if inclusive else ""}"
        mask = z >= t if inclusive else z > t
        metrics[f"%{label}_{t}m"] = zw[mask].sum() / total * 100

    return metrics


@with_suffix
def percent_below(
    z: np.ndarray,
    threshold: float | list[float] = 2,
    weights: np.ndarray | None = None,
    inclusive=False,
):
    if weights is None:
        zw = np.ones(z.shape)
    else:
        zw = weights

    total = zw.sum()

    if not isinstance(threshold, list):
        threshold = [threshold]

    metrics = {}

    for t in threshold:
        label = f"lt{"e" if inclusive else ""}"
        mask = z <= t if inclusive else z < t
        metrics[f"%{label}_{t}m"] = zw[mask].sum() / total * 100

    return metrics


@with_suffix
def percent_in(
    z: np.ndarray,
    threshold: tuple[float, float] | list[tuple[float, float]] = (0, 1),
    weights: np.ndarray | None = None,
    inclusive=(False, True),
):
    if weights is None:
        zw = np.ones(z.shape)
    else:
        zw = weights

    total = zw.sum()

    if not isinstance(threshold, list):
        threshold = [threshold]

    metrics = {}

    for lt, ut in threshold:

        llab = "[" if inclusive[0] else "("
        ulab = "]" if inclusive[1] else ")"

        lmask = z >= lt if inclusive[0] else z > lt
        umask = z <= ut if inclusive[1] else z < ut

        metrics[f"%inside_{llab}{lt},{ut}m{ulab}"] = (
            zw[lmask & umask].sum() / total * 100
        )

    return metrics


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
    labels = [f"[{lower},{upper}%)" for lower, upper in zip(bounds, bounds[1:])]
    labels[-1] = labels[-1].replace("%)", "%]")

    for label, count, density in zip(labels, counts, densities):
        metrics[f"%inside_{label}"] = count / total * 100

    return metrics
