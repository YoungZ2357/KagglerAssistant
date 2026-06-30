from math import isnan, isinf


def safe_val(v):
    if v is None:
        return None
    if isinstance(v, float):
        if isnan(v) or isinf(v):
            return None
        return round(v, 6)
    if hasattr(v, "__int__") and not isinstance(v, (bool, str)):
        return int(v)
    return str(v)
