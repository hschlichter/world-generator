"""Procedural trees and bushes.

A tree is a branching skeleton (trunk, primary and secondary branches, leaf-card attachment points) generated
once per species variant. All eight LOD levels are built from that skeleton, so the trunk, branch layout and
crown volume stay put while detail is removed and each switch changes the picture only a little:

  lod0  bent leaf cards (3 x 3 vertices), trunk and two branch orders as finely tessellated tubes
  lod1  the same cards flat, wood at medium tessellation
  lod2  cards merged 2:1, wood with fewer sides and rings
  lod3  cards merged 4:1, twigs as 3-sided tubes with long segments
  lod4  cards merged 8:1, twigs dropped, primaries at 3 sides
  lod5  cards merged 16:1, trunk and a few rings per primary
  lod6  crown as 2 x crown_clusters convex hulls (Blender) around lod4's cards, trunk prism, largest primaries
  lod7  crown as crown_clusters hulls of about 8 triangles, trunk as a 4-sided prism

Card merging is hierarchical: each level clusters the previous level's cards, so a merged card sits where its
members were and covers about the same projected area. Bushes have no trunk (their wood is branch orders 1 and 2);
dead trees have no foliage, so their levels only simplify the wood.
"""

import zlib
from dataclasses import dataclass, field

import numpy as np

from .. import atlas
from .. import blender as bl
from ..meshutil import Mesh, merge, prism, tube


@dataclass
class Species:
    name: str
    kind: str                 # "broadleaf", "conifer", "dead", "bush"
    height: tuple
    trunk_radius: float
    crown_start: float        # fraction of height where branches begin
    bark: str
    foliage: str | None
    card_size: tuple = (0.9, 1.3)
    n_primary: tuple = (9, 13)
    primary_len: float = 0.5  # max primary branch length as fraction of height
    elevation: tuple = (25.0, 55.0)
    gravitropism: float = 0.06
    leaf_spacing: float = 0.42
    secondary: tuple = (5, 8)
    whorl_step: tuple = (0.45, 0.7)
    lean: float = 0.03
    crown_clusters: int = 4


SPECIES = {
    "oak": Species("oak", "broadleaf", (13.0, 18.0), 0.45, 0.27, "bark_oak", "leaves_oak", (1.0, 1.45), (11, 15),
                   0.62, (6.0, 36.0), 0.05, 0.38, (6, 9), crown_clusters=5),
    "birch": Species("birch", "broadleaf", (14.0, 19.0), 0.2, 0.42, "bark_birch", "leaves_birch", (0.8, 1.1), (12, 16),
                     0.30, (40.0, 65.0), -0.05, 0.36, (4, 7), lean=0.05, crown_clusters=3),
    "pine": Species("pine", "conifer", (17.0, 24.0), 0.34, 0.55, "bark_pine", "needles_pine", (0.9, 1.2),
                    primary_len=0.24, elevation=(5.0, 30.0), whorl_step=(0.7, 1.1), crown_clusters=2),
    "spruce": Species("spruce", "conifer", (14.0, 21.0), 0.3, 0.06, "bark_pine", "needles_spruce", (0.8, 1.05),
                      primary_len=0.30, elevation=(-18.0, 4.0), whorl_step=(0.42, 0.6), crown_clusters=2),
    "deadwood": Species("deadwood", "dead", (8.0, 13.0), 0.3, 0.35, "bark_dead", None, n_primary=(5, 8),
                        primary_len=0.42, elevation=(15.0, 50.0), secondary=(2, 4)),
    "shrub": Species("shrub", "bush", (1.4, 2.4), 0.04, 0.0, "bark_oak", "leaves_bush", (0.5, 0.75), (6, 10),
                     1.0, (35.0, 75.0), 0.02, 0.22, (3, 5), crown_clusters=2),
    "juniper": Species("juniper", "bush", (1.0, 1.8), 0.05, 0.0, "bark_pine", "needles_spruce", (0.45, 0.6), (8, 12),
                       1.0, (25.0, 65.0), 0.05, 0.2, (3, 5), crown_clusters=1),
}


@dataclass
class Branch:
    path: np.ndarray
    radii: np.ndarray
    order: int


@dataclass
class Skeleton:
    branches: list
    leaves: list = field(default_factory=list)   # (attach point, up dir, size)
    height: float = 0.0
    crown_center: np.ndarray = None


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


def _grow(start, direction, length, n, rng, gravitropism, wobble=0.12):
    pts = [np.asarray(start, float)]
    d = _unit(np.asarray(direction, float))
    step = length / (n - 1)
    for _ in range(n - 1):
        d = _unit(d + np.array([0, 0, gravitropism]) + rng.normal(0, wobble, 3))
        pts.append(pts[-1] + d * step)
    return np.array(pts)


def _point_on(path, t):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    s = t * seg[-1]
    i = min(np.searchsorted(seg, s) - 1, len(path) - 2)
    i = max(i, 0)
    f = (s - seg[i]) / max(seg[i + 1] - seg[i], 1e-9)
    return path[i] * (1 - f) + path[i + 1] * f, _unit(path[i + 1] - path[i]), i, f


def _radius_on(br, t):
    x = np.linspace(0, 1, len(br.radii))
    return float(np.interp(t, x, br.radii))


def skeleton(sp, rng):
    H = rng.uniform(*sp.height)
    br = []
    leaves = []
    if sp.kind == "bush":
        return _bush_skeleton(sp, rng, H)
    # Trunk.
    n = 22
    t = np.linspace(0, 1, n)
    top = 1.0 if sp.kind == "conifer" else (0.82 if sp.kind == "broadleaf" else 0.7)
    lean_dir = rng.normal(0, 1, 2)
    lean_dir /= np.linalg.norm(lean_dir)
    wob = np.cumsum(rng.normal(0, 0.05 if sp.kind == "conifer" else 0.12, (n, 2)), axis=0) * (H / 20)
    xy = lean_dir * sp.lean * H * t[:, None] ** 1.5 + wob * t[:, None]
    trunk = np.c_[xy, t * H * top]
    r = sp.trunk_radius * (1.0 - 0.85 * t) ** 1.05 + 0.03
    r = r * (1.0 + 0.55 * np.exp(-t * H / 0.7))   # root flare
    if sp.kind == "dead":
        r[-1] = r[-2] * 0.8                      # broken top
    br.append(Branch(trunk, r, 0))

    if sp.kind == "conifer":
        z = sp.crown_start * H
        whorl = 0
        while z < H * 0.97:
            tz = z / H
            maxlen = sp.primary_len * H
            if sp.name == "pine":
                env = np.clip(np.sin(np.pi * np.clip((tz - sp.crown_start) / (1 - sp.crown_start), 0, 1)) ** 0.8, 0.15, 1)
                env *= 1.0 - 0.6 * max(0, tz - 0.85) / 0.15
            else:
                env = (1.0 - tz) ** 0.95
            k = rng.integers(4, 7)
            off = rng.uniform(0, 2 * np.pi)
            base, _, _, _ = _point_on(trunk, tz)
            for j in range(k):
                az = off + j * 2 * np.pi / k + rng.normal(0, 0.15)
                el = np.radians(rng.uniform(*sp.elevation))
                L = maxlen * env * rng.uniform(0.75, 1.1) + 0.3
                d = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
                grav = 0.07 if sp.name == "pine" else -0.05
                path = _grow(base, d, L, 5, rng, grav, 0.05)
                if sp.name == "spruce":   # tips turn up
                    path[-1, 2] += 0.15 * L
                rr = np.linspace(0.32 * float(np.interp(tz, t, r)) * min(1, L / 2.5) + 0.012, 0.012, 5)
                br.append(Branch(path, rr, 1))
                # Needle cards along the branch, two crossed planes per position.
                s = 0.18
                while s <= 1.0:
                    p, dd, _, _ = _point_on(path, s)
                    sz = rng.uniform(*sp.card_size) * (0.7 + 0.3 * env)
                    leaves.append((p - dd * sz * 0.35, _unit(dd + rng.normal(0, 0.15, 3)), sz))
                    s += sp.leaf_spacing * 1.2 / max(L, 0.5)
            z += rng.uniform(*sp.whorl_step)
            whorl += 1
        # Leader tuft.
        for _ in range(3):
            leaves.append((trunk[-1] - np.array([0, 0, 0.8]), _unit(np.array([0, 0, 1.0]) + rng.normal(0, 0.3, 3)),
                           sp.card_size[1]))
        cc = np.array([0, 0, H * (sp.crown_start + 1) / 2])
        return Skeleton(br, leaves, H, cc)

    # Broadleaf / dead: primary branches from crown_start to the trunk top, phyllotaxis azimuths.
    n1 = rng.integers(*sp.n_primary)
    golden = np.radians(137.5)
    az0 = rng.uniform(0, 2 * np.pi)
    for i in range(n1):
        tt = sp.crown_start + (1 - sp.crown_start) * (i + rng.uniform(0.2, 0.8)) / n1
        tt = min(tt, 0.98)
        base, tdir, _, _ = _point_on(trunk, tt)
        rel = (tt - sp.crown_start) / (1 - sp.crown_start)
        env = np.sin(np.pi * (0.25 + 0.75 * rel)) ** 0.8 if sp.name != "birch" else (1 - rel) ** 0.6 * 0.8 + 0.2
        L = sp.primary_len * H * env * rng.uniform(0.7, 1.05) + 0.5
        az = az0 + i * golden + rng.normal(0, 0.2)
        el = np.radians(rng.uniform(*sp.elevation) + 20 * rel)
        d = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        path = _grow(base, d, L, 8, rng, sp.gravitropism)
        r0 = float(np.interp(tt, t, r)) * rng.uniform(0.45, 0.6)
        rr = np.linspace(r0, 0.02, 8) ** 1.0
        b1 = Branch(path, rr, 1)
        br.append(b1)
        n2 = rng.integers(*sp.secondary)
        for k in range(n2):
            s = 0.25 + 0.72 * (k + rng.uniform(0, 1)) / n2
            p, dd, _, _ = _point_on(path, s)
            side = _unit(np.cross(dd, rng.normal(0, 1, 3)))
            d2 = _unit(dd * 0.6 + side * 0.8 + np.array([0, 0, 0.25]))
            L2 = L * rng.uniform(0.3, 0.5) * (1.1 - 0.5 * s)
            path2 = _grow(p, d2, L2, 5, rng, sp.gravitropism * 1.5, 0.15)
            rr2 = np.linspace(max(_radius_on(b1, s) * 0.6, 0.015), 0.012, 5)
            br.append(Branch(path2, rr2, 2))
            if sp.foliage:
                _leaves_along(path2, sp, rng, leaves, 0.25)
        if sp.foliage:
            _leaves_along(path, sp, rng, leaves, 0.6)
    if sp.foliage:
        _leaves_along(trunk, sp, rng, leaves, 0.88)
    if sp.kind == "dead":
        # A few broken stubs low on the trunk.
        for _ in range(rng.integers(2, 5)):
            tt = rng.uniform(0.15, 0.5)
            base, _, _, _ = _point_on(trunk, tt)
            az = rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(az), np.sin(az), rng.uniform(-0.1, 0.4)])
            path = _grow(base, d, rng.uniform(0.4, 1.2), 3, rng, 0.0)
            rr = np.linspace(float(np.interp(tt, t, r)) * 0.35, 0.03, 3)
            br.append(Branch(path, rr, 2))
    cc = np.array([0, 0, H * (0.5 + sp.crown_start / 2)])
    return Skeleton(br, leaves, H, cc)


def _leaves_along(path, sp, rng, leaves, start):
    s = start
    length = np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
    while s <= 1.0:
        p, dd, _, _ = _point_on(path, s)
        sz = rng.uniform(*sp.card_size)
        up = _unit(dd + np.array([0, 0, 0.35]) + rng.normal(0, 0.45, 3))
        leaves.append((p, up, sz))
        s += sp.leaf_spacing / max(length, 0.3) * rng.uniform(0.7, 1.3)


def _bush_skeleton(sp, rng, H):
    br, leaves = [], []
    n1 = rng.integers(*sp.n_primary)
    for i in range(n1):
        az = rng.uniform(0, 2 * np.pi)
        el = np.radians(rng.uniform(*sp.elevation))
        d = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
        L = H * rng.uniform(0.7, 1.1)
        base = np.array([rng.normal(0, 0.08), rng.normal(0, 0.08), -0.05])
        path = _grow(base, d, L, 5, rng, sp.gravitropism, 0.15)
        br.append(Branch(path, np.linspace(sp.trunk_radius, 0.008, 5), 1))
        n2 = rng.integers(*sp.secondary)
        for k in range(n2):
            s = 0.3 + 0.65 * (k + rng.uniform(0, 1)) / n2
            p, dd, _, _ = _point_on(path, s)
            d2 = _unit(dd + rng.normal(0, 0.6, 3) + np.array([0, 0, 0.2]))
            path2 = _grow(p, d2, L * rng.uniform(0.3, 0.5), 3, rng, 0.05, 0.2)
            br.append(Branch(path2, np.linspace(sp.trunk_radius * 0.5, 0.006, 3), 2))
            _leaves_along(path2, sp, rng, leaves, 0.1)
        _leaves_along(path, sp, rng, leaves, 0.35)
    return Skeleton(br, leaves, H, np.array([0, 0, H * 0.5]))


# ---------------------------------------------------------------------------------------------------------------------
# Meshing
# ---------------------------------------------------------------------------------------------------------------------

def _wood(skel, sides_by_order, max_order, ring_step, u_repeat):
    parts = []
    for b in skel.branches:
        if b.order > max_order:
            continue
        sides = sides_by_order[b.order]
        L = np.linalg.norm(np.diff(b.path, axis=0), axis=1).sum()
        n = max(2, int(np.ceil(L / ring_step[b.order])) + 1)
        tt = np.linspace(0, 1, len(b.path))
        tn = np.linspace(0, 1, n)
        path = np.stack([np.interp(tn, tt, b.path[:, k]) for k in range(3)], 1)
        radii = np.interp(tn, tt, b.radii)
        circ = 2 * np.pi * radii[0]
        parts.append(tube(path, radii, sides, v_per_m=1.0 / atlas.BARK_REPEAT_M,
                          u_repeat=max(1.0, round(circ / 1.2)) if b.order == 0 else 1.0, cap_end=b.order > 0))
    return merge(parts)



def _kmeans(x, k, rng, iters=12):
    k = min(k, len(x))
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(k):
            if np.any(lab == j):
                c[j] = x[lab == j].mean(0)
    return lab, c


def _spherical_st(points, center, repeat):
    d = points - center
    az = np.arctan2(d[:, 1], d[:, 0])
    # Avoid the azimuth seam by mapping with |az| (mirrored texture across the seam).
    u = np.abs(az) / np.pi * repeat
    v = (points[:, 2] - points[:, 2].min()) / 1.5
    return np.stack([u, v], 1)


# ---------------------------------------------------------------------------------------------------------------------
# Foliage cards
# ---------------------------------------------------------------------------------------------------------------------

def _card_set(skel, rng):
    """Per-card attributes, fixed once so lod0 and lod1 show the same cards: attach, up, size, side, atlas cell."""
    P = np.array([l[0] for l in skel.leaves])
    U = np.array([l[1] for l in skel.leaves])
    S = np.array([l[2] for l in skel.leaves])
    side = np.cross(U, rng.normal(0, 1, (len(P), 3)))
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    cell = rng.integers(0, 4, len(P))
    return dict(P=P, U=U, S=S, side=side, cell=cell)


def _merge_cards(cs, k, rng):
    """Cluster cards into k cards. A merged card keeps its members' total area (less overlap) within their spread."""
    centers = cs["P"] + cs["U"] * cs["S"][:, None] * 0.5
    lab, cen = _kmeans(centers, max(k, 1), rng)
    out = {n: [] for n in cs}
    for j in range(len(cen)):
        m = np.nonzero(lab == j)[0]
        if len(m) == 0:
            continue
        S = cs["S"][m]
        spread = np.linalg.norm(centers[m] - cen[j], axis=1).max() if len(m) > 1 else 0.0
        size = min(np.sqrt((S ** 2).sum()) * 0.9, 2.0 * spread + S.max())
        size = max(size, S.max())
        up = _unit(cs["U"][m].sum(0))
        big = m[np.argmax(S)]
        side = cs["side"][big] - np.dot(cs["side"][big], up) * up
        side = side / max(np.linalg.norm(side), 1e-9)
        out["P"].append(cen[j] - up * size * 0.5)
        out["U"].append(up)
        out["S"].append(size)
        out["side"].append(side)
        out["cell"].append(cs["cell"][big])
    return {n: np.array(v) for n, v in out.items()}


def _cards_mesh(cs, crown_center, grid=1, bend=0.0):
    """Leaf cards as (grid x grid)-quad patches; bend > 0 cups each card around its up axis and droops its tip.

    Normals are bent away from the crown centre (crowns shade as volumes, not as flat cards) and tilted outward
    per vertex for softer shading."""
    n = len(cs["P"])
    S = cs["S"]
    U = cs["U"]
    R = cs["side"]
    C = cs["P"] + U * S[:, None] * 0.5
    N = np.cross(R, U)
    out = C - crown_center
    out[:, 2] *= 0.6
    base_n = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9) + np.array([0, 0, 0.35])
    t = np.linspace(-1.0, 1.0, grid + 1)
    a, b = np.meshgrid(t, t)                         # a along side, b along up
    a, b = a.ravel(), b.ravel()
    nv = len(a)
    h = S[:, None, None] * 0.5
    offs = (a[None, :, None] * R[:, None] + b[None, :, None] * U[:, None]) * h
    offs += N[:, None] * (bend * (a ** 2 * 0.5 + 0.35 * np.clip(b, 0, None) ** 2))[None, :, None] * h
    pts = (C[:, None] + offs).reshape(-1, 3)
    corner = offs / np.maximum(np.linalg.norm(offs, axis=2, keepdims=True), 1e-9)
    nrm = base_n[:, None] + 0.35 * corner
    nrm = (nrm / np.linalg.norm(nrm, axis=2, keepdims=True)).reshape(-1, 3)
    rects = np.array([atlas.foliage_cell(int(c)) for c in cs["cell"]])
    fu = (a + 1) * 0.5
    fv = (b + 1) * 0.5
    st = np.stack([rects[:, None, 0] + fu * (rects[:, None, 2] - rects[:, None, 0]),
                   rects[:, None, 1] + fv * (rects[:, None, 3] - rects[:, None, 1])], 2).reshape(-1, 2)
    quads = []
    for i in range(grid):
        for j in range(grid):
            v0 = i * (grid + 1) + j
            quads.append([v0, v0 + 1, v0 + grid + 2, v0 + grid + 1])
    quads = np.array(quads).ravel()
    idx = (quads[None, :] + nv * np.arange(n)[:, None]).ravel()
    return Mesh(pts, np.full(n * grid * grid, 4), idx, nrm, st, double_sided=True)


def _crown_hulls(cs, k, target_tris, crown_center, rng, cache):
    """k convex hulls around the card extents (Blender), foliage texture mapped spherically."""
    ext = cs["S"][:, None]
    mid = cs["P"] + cs["U"] * ext * 0.5
    tips = np.concatenate([cs["P"], cs["P"] + cs["U"] * ext, mid + cs["side"] * ext * 0.5,
                           mid - cs["side"] * ext * 0.5])
    lab, cen = _kmeans(tips, k, rng)
    groups = [tips[lab == j] for j in range(len(cen)) if np.sum(lab == j) >= 6]
    if not groups:
        groups, cen = [tips], tips.mean(0, keepdims=True)
    jobs = []
    for pts in groups:
        c = pts.mean(0)
        # Pull in slightly: card corners overshoot the visible foliage (alpha), the hull should not grow the crown.
        jobs.append(bl.Job("hull", points=c + (pts - c) * 0.97, target_tris=target_tris))
    bl.run(jobs, cache)
    hulls = []
    for j in jobs:
        h = j.result
        h.st = _spherical_st(h.points, crown_center, 3.0)
        h.double_sided = True
        hulls.append(h)
    return merge(hulls)


# ---------------------------------------------------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------------------------------------------------

N_LEVELS = 8

# Wood per level for lod0..lod5: (sides by branch order, max order, ring step by order in metres).
WOOD_TREE = [
    ({0: 20, 1: 10, 2: 6}, 2, {0: 0.25, 1: 0.3, 2: 0.35}),
    ({0: 12, 1: 7, 2: 4}, 2, {0: 0.45, 1: 0.5, 2: 0.6}),
    ({0: 10, 1: 5, 2: 3}, 2, {0: 0.8, 1: 0.9, 2: 1.2}),
    ({0: 8, 1: 4, 2: 3}, 2, {0: 1.2, 1: 1.3, 2: 2.5}),
    ({0: 7, 1: 3}, 1, {0: 1.8, 1: 2.0}),
    ({0: 6, 1: 3}, 1, {0: 2.5, 1: 3.0}),
]
# Conifers: many whorl branches with foliage along them, so thinner tubes, and the smallest whorl branches are
# dropped at the coarse levels (the last value: fraction of primaries kept, largest first).
WOOD_CONIFER = [
    ({0: 20, 1: 6}, 1, {0: 0.25, 1: 0.35}, 1.0),
    ({0: 12, 1: 4}, 1, {0: 0.45, 1: 0.6}, 1.0),
    ({0: 10, 1: 3}, 1, {0: 0.8, 1: 1.0}, 1.0),
    ({0: 8, 1: 3}, 1, {0: 1.2, 1: 1.8}, 0.6),
    ({0: 7, 1: 3}, 1, {0: 1.8, 1: 3.0}, 0.35),
    ({0: 6, 1: 3}, 1, {0: 2.5, 1: 5.0}, 0.2),
]
WOOD_BUSH = [
    ({1: 6, 2: 4}, 2, {1: 0.15, 2: 0.15}),
    ({1: 4, 2: 3}, 2, {1: 0.3, 2: 0.3}),
    ({1: 3, 2: 3}, 2, {1: 0.5, 2: 0.6}),
    ({1: 3, 2: 3}, 2, {1: 0.8, 2: 2.0}),
    ({1: 3}, 1, {1: 1.0}),
    ({1: 3}, 1, {1: 3.0}),
]


def build_levels(species_name, variant_index, seed, cache_dir):
    """Returns ([{part: Mesh}, ...] for the 8 LOD levels, finest first, {part: material}) for one species variant."""
    sp = SPECIES[species_name]
    rng = np.random.default_rng([seed, zlib.crc32(species_name.encode()), variant_index])
    skel = skeleton(sp, rng)
    mrng = np.random.default_rng([seed, 7, variant_index])
    bush = sp.kind == "bush"
    cc = skel.crown_center
    bark_v = 1.0 / atlas.BARK_REPEAT_M
    if sp.kind == "conifer":
        prim_r = sorted((b.radii[0] for b in skel.branches if b.order == 1), reverse=True)
        levels = []
        for sides, mo, step, keep in WOOD_CONIFER:
            rmin = prim_r[max(0, int(len(prim_r) * keep) - 1)] if prim_r else 0.0
            sk = Skeleton([b for b in skel.branches if b.order != 1 or b.radii[0] >= rmin], skel.leaves,
                          skel.height, skel.crown_center)
            levels.append({"wood": _wood(sk, sides, mo, step, 1)})
    else:
        levels = [{"wood": _wood(skel, sides, mo, step, 1)} for sides, mo, step in (WOOD_BUSH if bush else WOOD_TREE)]
    leafy = bool(sp.foliage and skel.leaves)
    if leafy:
        cards0 = _card_set(skel, mrng)
        n0 = len(cards0["P"])
        cs = [cards0, cards0]
        for div in (2, 4, 8, 16):
            cs.append(_merge_cards(cs[-1], max(3, n0 // div), mrng))
        levels[0]["leaves"] = _cards_mesh(cards0, cc, grid=2, bend=0.45)
        for li in range(1, 6):
            levels[li]["leaves"] = _cards_mesh(cs[li], cc)
    lv6, lv7 = {}, {}
    if not bush:
        trunk = skel.branches[0]
        top = trunk.path[-1, 2] * (0.8 if leafy else 1.0)
        # lod6 keeps the largest primaries so the base of the crown (or a dead tree's silhouette) keeps its limbs.
        prim = sorted([b for b in skel.branches if b.order == 1], key=lambda b: -b.radii[0])
        w6 = [prism(trunk.radii[0] * 0.85, top, 6, z0=-0.1, top_scale=0.35, cap=False, uv_scale=bark_v)]
        for b in prim[:5 if leafy else 6]:
            w6.append(tube(b.path[[0, len(b.path) // 2, -1] if not leafy else [0, len(b.path) // 2]],
                           b.radii[[0, len(b.radii) // 2, -1] if not leafy else [0, len(b.radii) // 2]], 3,
                           v_per_m=bark_v))
        lv6["wood"] = merge(w6)
        w7 = [prism(trunk.radii[0] * 0.8, top * (0.85 if leafy else 1.0), 4, z0=-0.1, top_scale=0.4, cap=False,
                    uv_scale=bark_v)]
        if not leafy:
            for b in prim[:3]:
                w7.append(tube(b.path[[0, -1]], b.radii[[0, -1]], 3, v_per_m=bark_v))
        lv7["wood"] = merge(w7)
    if leafy:
        # Conifer crowns are tall cones: more, larger hulls stacked up the trunk keep the tiered outline.
        k6, t6, k7, t7 = (6, 24, 2, 14) if sp.kind == "conifer" else (2 * sp.crown_clusters, 12, sp.crown_clusters, 8)
        lv6["crown"] = _crown_hulls(cs[4], k6, t6, cc, mrng, cache_dir)
        lv7["crown"] = _crown_hulls(cs[4], k7, t7, cc, mrng, cache_dir)
    levels += [lv6, lv7]
    assert len(levels) == N_LEVELS
    materials = {"wood": sp.bark, "leaves": sp.foliage, "crown": sp.foliage}
    return levels, {k: v for k, v in materials.items() if v}
