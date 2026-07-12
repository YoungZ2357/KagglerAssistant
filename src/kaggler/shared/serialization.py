from math import isnan, isinf

from kaggler.shared.limits import truncate_str


def safe_val(v):
    if v is None:
        return None
    if isinstance(v, float):
        if isnan(v) or isinf(v):
            return None
        return round(v, 6)
    if hasattr(v, "__int__") and not isinstance(v, (bool, str)):
        return int(v)
    # 字符串统一截断，避免长文本单元格（sample_values / preview 等）撑爆上下文。
    return truncate_str(str(v))
