"""Ground cover (point-instancer prototypes) and deadfall (fallen logs, stumps)."""

import numpy as np

from .. import atlas
from ..meshutil import Mesh, cards, merge, prism, tube, mat_trs
from . import rocks


def _tuft(rect, n, height, width, spread, rng, lean=0.25, up_bias=0.0):
    """n crossed vertical cards around the origin, bases at z=0, texture rect `rect`."""
    ang = rng.uniform(0, np.pi, n) + np.arange(n) * np.pi / n
    offs = rng.normal(0, spread, (n, 2))
    h = height * rng.uniform(0.8, 1.15, n)
    w = width * rng.uniform(0.85, 1.15, n)
    right = np.stack([np.cos(ang), np.sin(ang), np.zeros(n)], 1) * (w / 2)[:, None]
    tilt = np.stack([-np.sin(ang), np.cos(ang), np.zeros(n)], 1) * rng.uniform(-lean, lean, n)[:, None]
    up = (np.array([0, 0, 1.0]) + tilt) * (h / 2)[:, None]
    centers = np.c_[offs, np.zeros(n)] + up
    nrm = np.tile([0, 0, 1.0], (n, 1)) + tilt * 0.5   # grass lit like the ground it grows on
    return cards(centers, right, up, [rect] * n, nrm)


PROTOTYPES = {
    # name: (material, builder)
    "grass_a": "grass",
    "grass_b": "grass",
    "grass_dry": "grass",
    "flowers_yellow": "grass",
    "flowers_white": "grass",
    "flowers_purple": "grass",
    "flowers_red": "grass",
    "clover": "grass",
    "reeds": "grass",
    "fern": "fern",
    "pebbles": "rock",
}


def scatter_mesh(name, seed):
    rng = np.random.default_rng([seed, rocks.hash_str(name)])
    R = atlas.GRASS_RECTS
    if name == "grass_a":
        return _tuft(R["grass_a"], 3, 0.55, 0.75, 0.06, rng)
    if name == "grass_b":
        return _tuft(R["grass_b"], 3, 0.95, 0.8, 0.08, rng, lean=0.3)
    if name == "grass_dry":
        return _tuft(R["grass_dry"], 2, 0.35, 0.9, 0.05, rng)
    if name.startswith("flowers_"):
        col = name.split("_")[1]
        a = _tuft(R[f"flower_{col}"], 2, 0.55, 0.45, 0.05, rng, lean=0.15)
        b = _tuft(R["grass_a"], 2, 0.4, 0.6, 0.08, rng)
        return merge([a, b])
    if name == "clover":
        m = _tuft(R["clover"], 3, 0.2, 0.55, 0.12, rng, lean=0.9)
        return m
    if name == "reeds":
        return _tuft(R["reed"], 4, 1.5, 0.45, 0.1, rng, lean=0.12)
    if name == "fern":
        n = 5
        parts = []
        for i in range(n):
            a = i * 2 * np.pi / n + rng.normal(0, 0.2)
            el = np.radians(rng.uniform(30, 55))
            L = rng.uniform(0.75, 1.05)
            d = np.array([np.cos(a) * np.cos(el), np.sin(a) * np.cos(el), np.sin(el)])
            side = np.array([-np.sin(a), np.cos(a), 0.0])
            # Two segments per frond so it arches over.
            d2 = d * np.array([1, 1, 0.2]) - np.array([0, 0, 0.25])
            d2 /= np.linalg.norm(d2)
            c1 = d * L * 0.25
            c2 = d * L * 0.5 + d2 * L * 0.25
            nrm = np.cross(side, d) + np.array([0, 0, 0.6])
            m1 = cards([c1], [side * 0.22], [d * L * 0.25], [(0, 0, 1, 0.5)], [nrm])
            nrm2 = np.cross(side, d2) + np.array([0, 0, 0.6])
            m2 = cards([c2], [side * 0.22], [d2 * L * 0.25], [(0, 0.5, 1, 1)], [nrm2])
            parts += [m1, m2]
        return merge(parts)
    if name == "pebbles":
        ms = []
        for i in range(2):
            r = rocks.rock_mesh("pebble", seed + i)
            ms.append(r.transformed(mat_trs((rng.normal(0, 0.25), rng.normal(0, 0.25), 0), rng.uniform(0, 6),
                                            rng.uniform(0.6, 1.8))))
        m = merge(ms)
        return m
    raise KeyError(name)


# Per level (lod0..lod7): tube sides, rings along the log, branch stubs / roots kept (fraction), stub/root sides.
LOG_LEVELS = [(12, 14, 1.0, 5), (10, 10, 1.0, 4), (8, 7, 1.0, 3), (7, 5, 0.5, 3), (6, 4, 0.0, 3), (5, 3, 0.0, 3),
              (4, 3, 0.0, 3), (3, 2, 0.0, 3)]
STUMP_LEVELS = [(14, 6, 1.0, 6), (12, 6, 1.0, 5), (10, 4, 1.0, 4), (8, 3, 0.6, 3), (6, 2, 0.6, 3), (5, 2, 0.0, 3),
                (4, 2, 0.0, 3), (3, 2, 0.0, 3)]


def _resample(path, r, n):
    t = np.linspace(0, 1, len(path))
    tn = np.linspace(0, 1, n)
    return np.stack([np.interp(tn, t, path[:, k]) for k in range(3)], 1), np.interp(tn, t, r)


def log_levels(variant, seed, cache_dir):
    """Fallen log (log_a, log_b) or stump, 8 levels: the same path and radius profile, with fewer sides and rings
    per level; branch stubs / roots thin out and disappear by lod4 / lod5."""
    rng = np.random.default_rng([seed, rocks.hash_str(variant)])
    uvb = 1.0 / atlas.BARK_REPEAT_M
    levels = []
    if variant == "stump":
        n = 6
        t = np.linspace(0, 1, n)
        path = np.stack([np.zeros(n), np.zeros(n), t * 0.7 - 0.15], 1)
        r = 0.45 * (1 + 0.5 * (1 - t) ** 3)
        roots = []
        for i in range(5):
            a = i * 2 * np.pi / 5 + rng.normal(0, 0.3)
            roots.append(np.array([[0, 0, 0.05], [np.cos(a) * 0.7, np.sin(a) * 0.7, -0.05],
                                   [np.cos(a) * 1.3, np.sin(a) * 1.3, -0.25]]))
        for sides, rings, keep, rsides in STUMP_LEVELS:
            p, rr = _resample(path, r, rings)
            bark = [tube(p, rr, sides, v_per_m=uvb, u_repeat=2.0 if sides > 8 else 1.0)]
            for rp in roots[:int(round(len(roots) * keep))]:
                bark.append(tube(rp if rsides > 3 else rp[[0, 2]], [0.22, 0.13, 0.05] if rsides > 3 else [0.22, 0.05],
                                 rsides, v_per_m=uvb))
            top = prism(rr[-1] * 0.98, 0.01, sides, z0=p[-1, 2] - 0.01, cap=True, uv_scale=0.6)
            levels.append({"bark": merge(bark), "wood": top})
        return levels
    L = 7.5 if variant == "log_a" else 10.0
    R0 = 0.38 if variant == "log_a" else 0.48
    n = 14
    t = np.linspace(0, 1, n)
    path = np.stack([t * L - L / 2, 0.15 * np.sin(t * 3 + rng.uniform(0, 3)), R0 * 0.75 + 0.06 * np.sin(t * 5)], 1)
    r = R0 * (1.0 - 0.35 * t)
    stubs = []
    for i in range(rng.integers(3, 6)):
        s = rng.uniform(0.25, 0.95)
        p = path[int(s * (n - 1))]
        a = rng.uniform(0, 2 * np.pi)
        d = np.array([rng.uniform(-0.3, 0.6), np.cos(a), abs(np.sin(a)) * 0.8 + 0.2])
        d /= np.linalg.norm(d)
        stubs.append((np.array([p, p + d * rng.uniform(0.5, 1.4)]), r[int(s * (n - 1))] * 0.35))
    # Largest stubs are kept longest.
    stubs.sort(key=lambda sb: -sb[1])
    for sides, rings, keep, ssides in LOG_LEVELS:
        p, rr = _resample(path, r, rings)
        bark = [tube(p, rr, sides, v_per_m=uvb, u_repeat=2.0 if sides > 8 else 1.0)]
        for sp, sr in stubs[:int(round(len(stubs) * keep))]:
            bark.append(tube(sp, [sr, 0.03], ssides, v_per_m=uvb, cap_end=ssides > 3))
        ends = []
        for k, (pp, re) in enumerate(((p[0], rr[0]), (p[-1], rr[-1]))):
            e = prism(re * 0.97, 0.02, sides, z0=-0.01, cap=True, uv_scale=0.6)
            rot = mat_trs((0, 0, 0), 0, 1.0, 0, np.pi / 2 * (1 if k else -1))
            ends.append(e.transformed(rot @ mat_trs(pp)))
        levels.append({"bark": merge(bark), "wood": merge(ends)})
    return levels
