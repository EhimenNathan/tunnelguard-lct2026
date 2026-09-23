"""Regenerate config/tunnel_guard.yaml from the dataclass defaults (single source of truth)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tunnel_guard import params as P  # noqa: E402

HEADER = """# TunnelGuard parameters (generated from the defaults in tunnel_guard/core/*.py by tools/gen_config.py).
# Every value can be overridden here or on the command line, e.g.
#   ros2 run tunnel_guard detector_node --ros-args -p gauge.h_min:=0.2
"""


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, str):
        return "'" + v + "'"
    if isinstance(v, list):
        return '[' + ', '.join(fmt(x) for x in v) + ']'
    return repr(v)


def main():
    flat = P.flatten(P.default_config())
    lines = [HEADER, 'tunnel_guard:', '  ros__parameters:', "    input_topic: 'auto'", '    publish_envelope_points: true',
             '    marker_step: 2.0']
    groups = {}
    for k, v in flat.items():
        head, _, tail = k.partition('.')
        groups.setdefault(head if tail else '', []).append((tail or head, v))
    for k, v in groups.pop('', []):
        lines.append(f'    {k}: {fmt(v)}')
    for g, items in groups.items():
        lines.append(f'    {g}:')
        for k, v in items:
            lines.append(f'      {k}: {fmt(v)}')
    out = os.path.join(os.path.dirname(__file__), '..', 'config', 'tunnel_guard.yaml')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(open(out).read())


if __name__ == '__main__':
    main()
