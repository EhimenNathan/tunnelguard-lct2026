"""Mapping between flat ROS parameter names (e.g. 'gauge.h_min') and the nested DetectorConfig dataclasses."""
import dataclasses

from .core.detector import DetectorConfig


def flatten(cfg, prefix=''):
    out = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        name = prefix + f.name
        if dataclasses.is_dataclass(v):
            out.update(flatten(v, name + '.'))
        elif isinstance(v, tuple) and v and isinstance(v[0], tuple):
            out[name] = [float(x) for row in v for x in row]          # tuple of tuples -> flat list
        elif isinstance(v, tuple):
            out[name] = [float(x) for x in v]
        else:
            out[name] = v
    return out


def apply(cfg, name, value):
    parts = name.split('.')
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    cur = getattr(obj, parts[-1])
    if isinstance(cur, tuple) and cur and isinstance(cur[0], tuple):
        k = len(cur[0])
        vals = list(value)
        value = tuple(tuple(float(x) for x in vals[i:i + k]) for i in range(0, len(vals), k))
    elif isinstance(cur, tuple):
        value = tuple(value)
    elif isinstance(cur, bool):
        value = bool(value)
    elif isinstance(cur, int):
        value = int(value)
    elif isinstance(cur, float):
        value = float(value)
    setattr(obj, parts[-1], value)


def default_config():
    return DetectorConfig()
