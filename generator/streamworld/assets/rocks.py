"""Procedural rocks: boulders, cliff blocks, pillars, standing stones, pebbles.

lod0 is a displaced, fractured icosphere; lod1 and lod2 are Blender quadric-collapse decimations of it, so every
level is a genuine simplification of the same shape.
"""

import numpy as np

from .. import atlas
from .. import blender as bl
from .. import noise as nz
from ..meshutil import Mesh, icosphere


def fbm3(p, seed, freq, octaves=4):
    """Cheap 3D fBm from three orthogonal 2D noise planes."""
    x, y, z = p[:, 0] * freq, p[:, 1] * freq, p[:, 2] * freq
    a = nz.fbm(x, y, seed, 1.0, octaves)
    b = nz.fbm(y + 31.7, z, seed + 1, 1.0, octaves)
    c = nz.fbm(z + 17.3, x, seed + 2, 1.0, octaves)
    return (a + b + c) / 1.7


SHAPES = {
    # name: (ellipsoid scale, displacement, fracture planes, flatten bottom, strata, subdiv)
    "round": ((1.0, 0.85, 0.7), 0.22, 3, 0.45, 0.0, 4),
    "angular": ((1.0, 0.9, 0.8), 0.15, 9, 0.5, 0.0, 4),
    "flat": ((1.5, 1.1, 0.42), 0.15, 5, 0.6, 0.0, 4),
    "cliff_a": ((6.0, 3.2, 5.5), 0.12, 8, 0.7, 0.07, 5),
    "cliff_b": ((8.0, 2.6, 4.0), 0.12, 7, 0.7, 0.08, 5),
    "cliff_c": ((4.5, 4.0, 7.0), 0.13, 9, 0.7, 0.06, 5),
    "pillar_a": ((1.6, 1.4, 9.0), 0.12, 6, 0.8, 0.05, 5),
    "pillar_b": ((2.2, 1.7, 13.0), 0.10, 7, 0.8, 0.06, 5),
    "menhir_a": ((0.55, 0.38, 1.9), 0.10, 5, 0.9, 0.0, 4),
    "menhir_b": ((0.65, 0.3, 2.3), 0.10, 6, 0.9, 0.0, 4),
    "menhir_c": ((0.5, 0.45, 1.6), 0.14, 4, 0.9, 0.0, 4),
    "altar": ((1.4, 0.9, 0.5), 0.06, 8, 0.6, 0.0, 4),
    "pebble": ((0.12, 0.1, 0.07), 0.2, 3, 0.4, 0.0, 0),
}


def rock_mesh(shape, seed):
    scale, disp, nplanes, flat, strata, subdiv = SHAPES[shape]
    rng = np.random.default_rng([seed, abs(hash_str(shape))])
    m = icosphere(subdiv)
    p = m.points.copy()
    scale = np.asarray(scale)
    q = p * scale
    size = scale.max()
    d = 1.0 + disp * fbm3(p, seed + 3, 1.3) + disp * 0.35 * fbm3(p, seed + 9, 4.5, 3)
    q = q * d[:, None]
    # Fracture planes: clamp everything beyond random planes onto them.
    for _ in range(nplanes):
        n = rng.normal(0, 1, 3)
        n[2] *= 0.6
        n /= np.linalg.norm(n)
        ext = np.abs(q @ n).max()
        off = rng.uniform(0.62, 0.88) * ext
        s = q @ n - off
        q -= np.maximum(s, 0)[:, None] * n[None, :] * 0.92
    if strata > 0:
        z = q[:, 2]
        band = scale[2] * 0.11
        ph = (z / band) % 1.0
        ridge = (np.abs(ph - 0.5) * 2) ** 3
        r = np.hypot(q[:, 0], q[:, 1])
        q[:, :2] *= (1.0 + strata * (ridge - 0.5))[:, None] / np.maximum(1, 1)
    # Flatten the bottom and sit the rock on z = 0 with a little below ground.
    zmin = q[:, 2].min()
    cut = zmin + (q[:, 2].max() - zmin) * (1 - flat) * 0.25
    q[:, 2] = np.maximum(q[:, 2], cut)
    q[:, 2] -= cut + 0.12 * scale[2]
    m.points = q
    m.normals = None
    m.compute_normals()
    rep = atlas.ROCK_REPEAT_M
    m.st = np.stack([q[:, 0] + 0.45 * q[:, 2], q[:, 1] - 0.3 * q[:, 2]], 1) / rep
    return m


def hash_str(s):
    import zlib
    return zlib.crc32(s.encode())


def build_levels(shape, seed, cache_dir, targets):
    """[{part: Mesh}] per level. targets: triangle counts for the coarser levels."""
    m0 = rock_mesh(shape, seed)
    levels = [{"rock": m0}]
    jobs = [bl.Job("decimate", mesh=m0, target_tris=t) for t in targets]
    bl.run(jobs, cache_dir)
    for j in jobs:
        r = j.result
        r.compute_normals()
        q = r.points
        r.st = np.stack([q[:, 0] + 0.45 * q[:, 2], q[:, 1] - 0.3 * q[:, 2]], 1) / atlas.ROCK_REPEAT_M
        levels.append({"rock": r})
    return levels
