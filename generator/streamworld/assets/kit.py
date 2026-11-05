"""Building blocks for the hand-modelled assets (buildings, fantasy pieces): primitives with planar UVs and a level kit.

A `Kit` collects parts, each a mesh with a material part name and the range of LOD levels it appears in. Detail is
added as parts that end early (trims, frames, rubble) and simplified stand-ins that start late (one box for a wall
of blocks), so every level is a genuine simplification of the one before it. `Kit.levels()` returns the asset's 8
levels as [{part: merged mesh}], finest first.
"""

import numpy as np

from ..meshutil import Mesh, box, merge, mat_trs, tube

NLEV = 8
UV = 0.5   # 2 m per texture repeat


def quad(p0, p1, p2, p3, uv=UV):
    """A planar quad (counter-clockwise seen from the front); st in metres along its first and last edges."""
    q = np.array([p0, p1, p2, p3], float)
    n = np.cross(q[1] - q[0], q[3] - q[0])
    n /= max(np.linalg.norm(n), 1e-12)
    eu = q[1] - q[0]
    eu /= max(np.linalg.norm(eu), 1e-12)
    ev = np.cross(n, eu)
    st = np.stack([(q - q[0]) @ eu, (q - q[0]) @ ev], 1) * uv
    return Mesh(q, [4], [0, 1, 2, 3], np.tile(n, (4, 1)), st)


def tri(p0, p1, p2, uv=UV):
    q = np.array([p0, p1, p2], float)
    n = np.cross(q[1] - q[0], q[2] - q[0])
    n /= max(np.linalg.norm(n), 1e-12)
    eu = q[1] - q[0]
    eu /= max(np.linalg.norm(eu), 1e-12)
    ev = np.cross(n, eu)
    st = np.stack([(q - q[0]) @ eu, (q - q[0]) @ ev], 1) * uv
    return Mesh(q, [3], [0, 1, 2], np.tile(n, (3, 1)), st)


def slab(p0, p1, p2, p3, thick, uv=UV):
    """A quad given thickness along its back normal: front, back and four edges (a roof plane, a sail)."""
    q = np.array([p0, p1, p2, p3], float)
    n = np.cross(q[1] - q[0], q[3] - q[0])
    n /= np.linalg.norm(n)
    b = q - n * thick
    faces = [quad(*q, uv=uv), quad(b[1], b[0], b[3], b[2], uv=uv)]
    for i in range(4):
        j = (i + 1) % 4
        faces.append(quad(q[j], q[i], b[i], b[j], uv=uv))
    return merge(faces)


def beam(a, b, w, h=None, uv=UV):
    """A square-section beam (w x h) from point a to point b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    h = w if h is None else h
    d = b - a
    L = float(np.linalg.norm(d))
    m = box((0, -w / 2, -h / 2), (L, w / 2, h / 2), uv)
    yaw = np.arctan2(d[1], d[0])
    pitch = np.arctan2(d[2], np.hypot(d[0], d[1]))
    return m.transformed(mat_trs(a, yaw, 1.0, 0.0, -pitch))


def lathe(rs, zs, sides, uv=UV, cap_top=False):
    """Surface of revolution of the profile (r, z) about z (bottom to top)."""
    rs, zs = np.asarray(rs, float), np.asarray(zs, float)
    path = np.stack([np.zeros_like(zs), np.zeros_like(zs), zs], 1)
    m = tube(path, rs, sides, v_per_m=uv, u_repeat=max(1.0, round(2 * np.pi * float(rs.max()) * uv)))
    if cap_top and rs[-1] > 1e-6:
        a = np.linspace(0, 2 * np.pi, sides + 1)[:-1]
        top = np.stack([np.cos(a) * rs[-1], np.sin(a) * rs[-1], np.full(sides, zs[-1])], 1)
        m = merge([m, Mesh(top, [sides], np.arange(sides), np.tile([0, 0, 1.0], (sides, 1)), top[:, :2] * uv)])
    return m


def gable(L, W, z0, rise, overhang=0.0):
    """Points of a gable roof along x (ridge at y = 0): (left eave, right eave, ridge) end points."""
    x0, x1 = -L / 2 - overhang, L / 2 + overhang
    y = W / 2 + overhang
    zr = z0 + rise
    ze = z0 - rise * overhang / (W / 2)
    return x0, x1, y, ze, zr


def capped_tube(path, radii, sides, uv=UV):
    """tube() closed at both ends with a short cone (bones, horns): no open, back-facing ends."""
    path = np.asarray(path, float)
    radii = np.asarray(radii, float)
    d0 = path[1] - path[0]
    d1 = path[-1] - path[-2]
    d0 /= np.linalg.norm(d0)
    d1 /= np.linalg.norm(d1)
    p = np.vstack([path[0] - d0 * radii[0] * 0.6, path[0] - d0 * radii[0] * 0.05, path,
                   path[-1] + d1 * radii[-1] * 0.05, path[-1] + d1 * radii[-1] * 0.6])
    r = np.r_[0.0, radii[0] * 0.8, radii, radii[-1] * 0.8, 0.0]
    return tube(p, r, sides, v_per_m=uv)


class Kit:
    def __init__(self):
        self.parts = []   # (part, mesh, first level, last level)

    def add(self, part, mesh, lo=0, hi=NLEV - 1):
        if mesh is not None:
            self.parts.append((part, mesh, lo, hi))

    def add_all(self, part, meshes, lo=0, hi=NLEV - 1):
        for m in meshes:
            self.add(part, m, lo, hi)

    def levels(self):
        out = []
        for li in range(NLEV):
            by = {}
            for part, m, lo, hi in self.parts:
                if lo <= li <= hi:
                    by.setdefault(part, []).append(m)
            lv = {p: merge(ms) for p, ms in sorted(by.items())}
            assert lv, f"level {li} is empty"
            out.append(lv)
        return out
