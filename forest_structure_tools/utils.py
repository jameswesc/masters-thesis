def add_suffix(obj: dict, suffix: str | None = None):
    """Add suffix to all keys in a dictionary if suffix is provided."""
    if suffix is None:
        return obj
    
    obj_with_suffix = {}
    for k, v in obj.items():
        obj_with_suffix[f"{k}_{suffix}"] = v
    
    return obj_with_suffix


def with_suffix(func):
    """Decorator that adds suffix parameter support to metrics functions."""
    def wrapper(*args, suffix: str | None = None, **kwargs):
        result = func(*args, **kwargs)
        return add_suffix(result, suffix)
    return wrapper