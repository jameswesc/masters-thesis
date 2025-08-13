from .typing import Suffix

def add_suffix(obj: dict, suffix: Suffix | None = None):
    """Add suffix to all keys in a dictionary if suffix is provided."""
    if suffix is None:
        return obj
    elif isinstance(suffix, list):
        suffix = ",".join(suffix)

    obj_with_suffix = {}
    for k, v in obj.items():
        obj_with_suffix[f"{k}[{suffix}]"] = v

    return obj_with_suffix
