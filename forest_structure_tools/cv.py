import numpy as np
import numpy.typing as npt


# help function for coefficient of variation
def cv(x: npt.NDArray) -> float:
    valid_x = x[~np.isnan(x)]
    if len(valid_x) >= 2 and valid_x.mean() != 0:
        return valid_x.std() / valid_x.mean()
    else:
        return np.nan
