"""Ruins and landmarks built from stone blocks, 8 LOD levels each.

lod0 places every block (individually jittered boxes) and all the rubble. The levels after it merge the masonry
hierarchically, the way the trees merge leaf cards: blocks into runs (walls) or vertical stacks (towers), then
the stepped solids that follow the same ruined profile at halving resolution, then the silhouette as one or two
prisms. Rubble thins out from the smallest pieces up. The rock arch is a decimation chain (Blender).
"""

import numpy as np

from .. import atlas
from .. import blender as bl
from ..meshutil import Mesh, box, merge, prism, tube, mat_trs
from . import rocks

UV = 1.0 / atlas.STONE_REPEAT_M


def _block(cx, cy, z0, lx, ly, lz, rz, rng, jitter=0.03):
    gap = 0.025
    b = box((-lx / 2 + gap, -ly / 2 + gap, 0), (lx / 2 - gap, ly / 2 - gap, lz - gap), UV)
    b = b.transformed(mat_trs((cx, cy, z0), rz + rng.normal(0, 0.02), 1.0, rng.normal(0, 0.01), rng.normal(0, 0.01)))
    # Re-derive world-ish planar UVs so neighbouring blocks don't repeat the same patch.
    b.st = b.st + rng.uniform(0, 1, 2)
    return b


def _profile(n, rng, state, base_h):
    """Top height of a ruined wall along its length (n columns)."""
    x = np.linspace(0, 1, n)
    if state == "intact":
        return np.full(n, base_h)
    k = rng.uniform(0, 10)
    wave = 0.5 + 0.5 * np.sin(x * rng.uniform(4, 9) + k) * np.cos(x * rng.uniform(2, 5) + k * 0.3)
    if state == "ruined":
        return base_h * (0.35 + 0.65 * wave)
    return base_h * (0.08 + 0.32 * wave)  # collapsed


def _wall_columns(prof, n, L, T, ch):
    """The wall as n boxes along its length, each at its stretch of the profile (snapped to courses)."""
    k = len(prof) // n
    w = L / n
    cols = []
    for i in range(n):
        h = np.floor(float(np.mean(prof[i * k:(i + 1) * k])) / ch) * ch
        if h > 0:
            cols.append(box((-L / 2 + i * w, -T / 2, 0), (-L / 2 + (i + 1) * w, T / 2, h), UV))
    return cols


def _by_size(meshes):
    """Meshes sorted largest first (by bounding-box volume), so thinning keeps the big pieces."""
    return sorted(meshes, key=lambda m: -float(np.prod(np.ptp(m.points, axis=0))))


def _keep(items, frac):
    return items[:int(round(len(items) * frac))]


def wall(state, seed):
    rng = np.random.default_rng([seed, 11, rocks.hash_str(state)])
    L, T, H, ch = 8.0, 1.1, 4.6, 0.45
    ncol = 16
    prof = _profile(ncol, rng, state, H)
    blocks = []
    runs = []          # (course, x0, x1) per block, for the merged levels
    courses = int(H / ch)
    for c in range(courses):
        z = c * ch
        x = -L / 2 + (0.4 if c % 2 else 0.0)
        while x < L / 2 - 0.05:
            lx = min(rng.uniform(0.7, 1.15), L / 2 - x)
            xc = x + lx / 2
            col = int(np.clip((xc + L / 2) / L * ncol, 0, ncol - 1))
            if z + ch <= prof[col] + rng.uniform(-0.2, 0.2):
                blocks.append(_block(xc, 0.0, z, lx, T, ch, 0.0, rng))
                runs.append((c, x, x + lx))
            x += lx
    rubble = []
    if state != "intact":
        n = 14 if state == "ruined" else 30
        for _ in range(n):
            xc = rng.uniform(-L / 2 - 1, L / 2 + 1)
            yc = rng.choice([-1, 1]) * rng.uniform(T / 2 + 0.2, 2.8)
            lx = rng.uniform(0.5, 1.0)
            b = _block(0, 0, 0, lx, rng.uniform(0.4, 0.9), ch, 0.0, rng)
            b = b.transformed(mat_trs((xc, yc, -0.05), rng.uniform(0, np.pi), 1.0, rng.normal(0, 0.15), rng.normal(0, 0.15)))
            rubble.append(b)
    rubble = _by_size(rubble)

    def merged_runs(m):
        """Contiguous blocks of a course merged m at a time into one box (no gaps, no jitter)."""
        out = []
        i = 0
        while i < len(runs):
            c, x0, x1 = runs[i]
            j = i + 1
            while j < len(runs) and j - i < m and runs[j][0] == c and abs(runs[j][1] - runs[j - 1][2]) < 1e-6:
                x1 = runs[j][2]
                j += 1
            out.append(box((x0, -T / 2, c * ch), (x1, T / 2, (c + 1) * ch), UV))
            i = j
        return out

    levels = [
        blocks + rubble,
        merged_runs(2) + _keep(rubble, 0.6),
        merged_runs(3) + _keep(rubble, 0.4),
        merged_runs(6) + _keep(rubble, 0.2),
        _wall_columns(prof, 16, L, T, ch) + [box(*r.extent(), UV) for r in _keep(rubble, 0.1)],
        _wall_columns(prof, 8, L, T, ch),
        _wall_columns(prof, 4, L, T, ch),
        [box((-L / 2, -T / 2, 0), (L / 2, T / 2, max(float(np.mean(prof)), 0.4)), UV)],
    ]
    return [{"stone": merge(lv)} for lv in levels]


def _tower_shell(R, T, top, ang, ch, n, inner=True):
    """n-sided shell whose wall segments follow the broken top (snapped to courses)."""
    parts = []
    a1 = np.linspace(0, 2 * np.pi, n + 1)
    for i in range(n):
        h = float(np.interp((a1[i] + a1[i + 1]) / 2, ang, top, period=2 * np.pi))
        h = np.floor(h / ch) * ch
        p0 = np.array([np.cos(a1[i]), np.sin(a1[i])])
        p1 = np.array([np.cos(a1[i + 1]), np.sin(a1[i + 1])])
        quads = [[np.r_[p0 * R, 0], np.r_[p1 * R, 0], np.r_[p1 * R, h], np.r_[p0 * R, h]],
                 [np.r_[p0 * R, h], np.r_[p1 * R, h], np.r_[p1 * (R - T), h], np.r_[p0 * (R - T), h]]]
        if inner:
            quads.append([np.r_[p1 * (R - T), 0], np.r_[p0 * (R - T), 0], np.r_[p0 * (R - T), h],
                          np.r_[p1 * (R - T), h]])
        for q in quads:
            q = np.array(q)
            nn = np.cross(q[1] - q[0], q[3] - q[0])
            nn /= np.linalg.norm(nn)
            st = np.stack([np.r_[0, 1, 1, 0] * np.linalg.norm(q[1] - q[0]), q[:, 2]], 1) * UV
            parts.append(Mesh(q, [4], [0, 1, 2, 3], np.tile(nn, (4, 1)), st))
    return parts


def tower(state, seed):
    rng = np.random.default_rng([seed, 12, rocks.hash_str(state)])
    R, T, H, ch = 3.4, 1.0, 15.0 if state == "intact" else 13.0, 0.5
    nseg = 22
    ang = np.linspace(0, 2 * np.pi, nseg + 1)[:-1]
    if state == "intact":
        top = np.full(nseg, H)
    else:
        k = rng.uniform(0, 6)
        top = H * (0.45 + 0.55 * (0.5 + 0.5 * np.sin(ang * 2 + k) * np.cos(ang + k * 0.7)))
    blocks = []
    courses = int(H / ch) + 2
    seg_len = 2 * np.pi * (R - T / 2) / nseg

    def opening(c, i):
        # Windows and a door.
        return (c in (2, 3, 4) and i == 0) or (c % 9 in (6, 7) and i % 7 == 3)

    for c in range(courses):
        z = c * ch
        off = (0.5 if c % 2 else 0.0) * 2 * np.pi / nseg
        for i in range(nseg):
            a = ang[i] + off
            ii = int(((a % (2 * np.pi)) / (2 * np.pi)) * nseg) % nseg
            if z + ch > top[ii] + rng.uniform(-0.3, 0.3):
                continue
            if opening(c, i):
                continue
            cx, cy = np.cos(a) * (R - T / 2), np.sin(a) * (R - T / 2)
            blocks.append(_block(cx, cy, z, seg_len * 1.04, T, ch, a + np.pi / 2, rng))
    crown = []
    rubble = []
    if state == "intact":
        # Crenellations.
        for i in range(0, nseg, 2):
            a = ang[i]
            cx, cy = np.cos(a) * (R - T / 2), np.sin(a) * (R - T / 2)
            crown.append(_block(cx, cy, courses * ch, seg_len, T, 0.7, a + np.pi / 2, rng))
    else:
        for _ in range(25):
            a = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(R + 0.3, R + 4.0)
            b = _block(0, 0, 0, rng.uniform(0.5, 1.0), rng.uniform(0.4, 0.9), ch, 0, rng)
            rubble.append(b.transformed(mat_trs((np.cos(a) * rr, np.sin(a) * rr, -0.05), rng.uniform(0, 3), 1.0,
                                                rng.normal(0, 0.2), rng.normal(0, 0.2))))
    rubble = _by_size(rubble)

    def stacks(k):
        """Masonry as vertical stacks of k courses per segment (no stagger, no gaps), openings kept."""
        out = []
        for i in range(nseg):
            a = ang[i]
            h_top = np.floor(top[i] / ch) * ch
            c = 0
            while c * ch < h_top:
                c1 = min(c + k, int(round(h_top / ch)))
                # Split the stack around openings rather than closing them.
                cc = c
                while cc < c1:
                    if opening(cc, i):
                        cc += 1
                        continue
                    ce = cc
                    while ce < c1 and not opening(ce, i):
                        ce += 1
                    out.append(_seg_box(a, R, T, seg_len * 1.04, cc * ch, ce * ch))
                    cc = ce
                c = c1
        return out

    levels = [
        blocks + crown + rubble,
        stacks(2) + crown + _keep(rubble, 0.5),
        stacks(4) + crown + _keep(rubble, 0.3),
        stacks(courses) + crown + _keep(rubble, 0.15),
        _tower_shell(R, T, top, ang, ch, 22) + crown[::2],
        _tower_shell(R, T, top, ang, ch, 12),
        _tower_shell(R, T, top, ang, ch, 8, inner=False),
        [prism(R, float(np.mean(top)), 6, cap=True, uv_scale=UV)],
    ]
    return [{"stone": merge(lv)} for lv in levels]


def _seg_box(a, R, T, length, z0, z1):
    b = box((-length / 2, -T / 2, z0), (length / 2, T / 2, z1), UV)
    return b.transformed(mat_trs((np.cos(a) * (R - T / 2), np.sin(a) * (R - T / 2), 0), a + np.pi / 2))


def column(state, seed):
    rng = np.random.default_rng([seed, 13, rocks.hash_str(state)])
    H = 6.0 if state == "intact" else rng.uniform(2.0, 3.8)
    r = 0.42
    base = [box((-0.65, -0.65, 0), (0.65, 0.65, 0.45), UV), box((-0.55, -0.55, 0.45), (0.55, 0.55, 0.6), UV)]
    base1 = [box((-0.65, -0.65, 0), (0.65, 0.65, 0.6), UV)]
    cap = [box((-0.6, -0.6, H), (0.6, 0.6, H + 0.4), UV)] if state == "intact" else []
    ndrum = max(1, int(H / 1.1))
    dz = (H - 0.6) / ndrum
    jit = [(rng.normal(0, 0.01), rng.normal(0, 0.01), rng.uniform(0, 0.2)) for _ in range(ndrum)]

    def drums(sides):
        out = []
        for i in range(ndrum):
            z0 = 0.6 + i * dz
            d = prism(r * (1 - 0.03 * i), dz - 0.02, sides, z0=z0, top_scale=0.985, cap=True, uv_scale=UV)
            out.append(d.transformed(mat_trs((jit[i][0], jit[i][1], 0), jit[i][2])))
        return out

    def shaft(sides, z0=0.6):
        return [prism(r, H - z0, sides, z0=z0, top_scale=1 - 0.03 * ndrum, cap=True, uv_scale=UV)]

    top = H + (0.4 if state == "intact" else 0.0)
    levels = [
        base + drums(16) + cap,
        base + drums(12) + cap,
        base + drums(8) + cap,
        base + shaft(10) + cap,
        base1 + shaft(8) + cap,
        base1 + shaft(6) + cap,
        [prism(0.55, top, 6, cap=True, uv_scale=UV)],
        [prism(0.6, top, 4, cap=True, uv_scale=UV)],
    ]
    return [{"stone": merge(lv)} for lv in levels]


def shrine(state, seed, cache_dir):
    rng = np.random.default_rng([seed, 14, rocks.hash_str(state)])
    platform = []
    # Stepped platform.
    for i, (s, h) in enumerate(((4.2, 0.3), (3.7, 0.3), (3.3, 0.3))):
        platform.append(box((-s, -s, i * 0.3), (s, s, i * 0.3 + h), UV))
    platform2 = [box((-4.2, -4.2, 0), (4.2, 4.2, 0.6), UV), box((-3.3, -3.3, 0.6), (3.3, 3.3, 0.9), UV)]
    platform1 = [box((-4.2, -4.2, 0), (4.2, 4.2, 0.9), UV)]
    cols = []
    for cx in (-2.6, 2.6):
        for cy in (-2.6, 2.6):
            st = "intact" if (state == "intact" or rng.uniform() < 0.6) else "broken"
            c = column(st, seed + int(cx * 10 + cy))
            cols.append([lv["stone"].transformed(mat_trs((cx, cy, 0.9), 0, 0.75)) for lv in c])
    roof = []
    if state == "intact":
        roof.append(box((-3.3, -3.3, 5.7), (3.3, 3.3, 6.1), UV))
        pyr = prism(4.4, 2.2, 4, z0=6.1, top_scale=0.05, cap=False, uv_scale=UV)
        roof.append(pyr.transformed(mat_trs((0, 0, 0), np.pi / 4)))
        roofs = [roof] * 6 + [[box((-3.3, -3.3, 5.7), (3.3, 3.3, 7.4), UV)]] * 2
    else:
        for _ in range(12):
            b = _block(0, 0, 0, rng.uniform(0.6, 1.4), rng.uniform(0.4, 0.8), 0.45, 0, rng)
            roof.append(b.transformed(mat_trs((rng.uniform(-4, 4), rng.uniform(-4, 4), 0.9), rng.uniform(0, 3), 1.0,
                                              rng.normal(0, 0.2), rng.normal(0, 0.2))))
        roof = _by_size(roof)
        roofs = [roof, roof, roof[:6], roof[:3], roof[:2], roof[:1], [], []]
    # The altar is a rock: a decimation chain down to a box.
    altar0 = rocks.rock_mesh("altar", seed + 5).transformed(mat_trs((0, 0, 0.9), 0.3, 0.8))
    jobs = [bl.Job("decimate", mesh=altar0, target_tris=t) for t in (1200, 400, 120)]
    bl.run(jobs, cache_dir)
    for j in jobs:
        j.result.compute_normals()
        q = j.result.points
        j.result.st = np.stack([q[:, 0] + 0.45 * q[:, 2], q[:, 1] - 0.3 * q[:, 2]], 1) / atlas.ROCK_REPEAT_M
    altar_box = box((-1.1, -0.7, 0.9), (1.1, 0.7, 1.3), UV)
    altars = [altar0] + [j.result for j in jobs] + [altar_box, altar_box, None, None]
    plats = [platform] * 5 + [platform2] + [platform1] * 2
    levels = []
    for li in range(8):
        parts = plats[li] + [c[li] for c in cols] + roofs[li] + ([altars[li]] if altars[li] is not None else [])
        m = merge(parts)
        m.normals = m.normals if m.normals is not None else m.compute_normals().normals
        levels.append({"stone": m})
    return levels


def arch(variant, seed, cache_dir):
    """Natural rock arch: a lumpy tube along a half ellipse."""
    rng = np.random.default_rng([seed, 15, rocks.hash_str(variant)])
    span, height = (22.0, 16.0) if variant == "a" else (30.0, 13.0)
    t = np.linspace(0, np.pi, 64)
    path = np.stack([np.cos(t) * span / 2, np.zeros_like(t), np.sin(t) * height - 3.0], 1)
    r = 3.2 + 1.4 * np.sin(t * 3 + rng.uniform(0, 3)) * 0.3 + 2.5 * (1 - np.sin(t)) ** 2
    m = tube(path, r, 28, v_per_m=1 / atlas.ROCK_REPEAT_M, u_repeat=3.0)
    d = 1.0 + 0.18 * rocks.fbm3(m.points / 6.0, seed + 2, 1.0)
    c = np.repeat(path, 29, axis=0)
    m.points = c + (m.points - c) * d[:, None]
    m.normals = None
    m.compute_normals()
    q = m.points
    m.st = np.stack([q[:, 0] + 0.45 * q[:, 2], q[:, 1] - 0.3 * q[:, 2]], 1) / atlas.ROCK_REPEAT_M
    levels = [{"rock": m}]
    jobs = [bl.Job("decimate", mesh=m, target_tris=t) for t in (1800, 1000, 560, 320, 180, 100, 60)]
    bl.run(jobs, cache_dir)
    for j in jobs:
        rr = j.result
        rr.compute_normals()
        q = rr.points
        rr.st = np.stack([q[:, 0] + 0.45 * q[:, 2], q[:, 1] - 0.3 * q[:, 2]], 1) / atlas.ROCK_REPEAT_M
        levels.append({"rock": rr})
    return levels
