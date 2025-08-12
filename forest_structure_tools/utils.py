Suffix = str | list[str] | None


def add_suffix(obj: dict, suffix: Suffix = None):
    """Add suffix to all keys in a dictionary if suffix is provided."""
    if suffix is None:
        return obj
    elif isinstance(suffix, list):
        suffix = ",".join(suffix)

    obj_with_suffix = {}
    for k, v in obj.items():
        obj_with_suffix[f"{k}[{suffix}]"] = v

    return obj_with_suffix


def with_suffix(func):
    """Decorator that adds suffix parameter support to metrics functions."""

    def wrapper(*args, suffix: Suffix, **kwargs):
        result = func(*args, **kwargs)
        return add_suffix(result, suffix)

    return wrapper
