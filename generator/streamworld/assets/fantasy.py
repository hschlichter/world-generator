"""The uncanny side of Eldermark: a wizard's spire, runestones, crystal clusters and the bones of a wyrm.

8 LOD levels each (see kit.py). Parts are named after their library material; `crystal` glows faintly (the
material has an emissive colour), so runes, crystal panes and clusters read from far away.
"""

import numpy as np

from ..meshutil import Mesh, box, merge, mat_trs, prism, tube
from .buildings import _rng
from .kit import UV, Kit, beam, capped_tube, lathe


def _pos_noise(p, seed):
    """A value in [-1, 1] per point that depends only on its position, so coincident vertices move together."""
    v = np.sin(p @ np.array([12.9898, 78.233, 37.719]) + seed * 0.618) * 43758.5453
    return (v - np.floor(v)) * 2 - 1


def _shard(rng, r, h, sides=6, jitter=0.25):
    """A rough rock lump / floating shard: a jittered double cone."""
    zs = np.array([-h * 0.45, -h * 0.15, h * 0.2, h * 0.5])
    rs = np.array([0.05, 1.0, 0.85, 0.05]) * r
    m = lathe(rs, zs, sides)
    m.points = m.points * (1 + jitter * _pos_noise(np.round(m.points, 5), rng.uniform(0, 100)))[:, None]
    m.normals = None
    m.compute_normals()
    return m


def _crystal(base, d, length, r, sides=6):
    """A hexagonal crystal from `base` along direction d: prism body and a pyramid tip."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    body = length * 0.78
    path = np.array([base - d * 0.3, base + d * body, base + d * length])
    m = tube(path, np.array([r, r * 0.92, 0.0]), sides, v_per_m=UV)
    return m


# ---------------------------------------------------------------------------------------------------------------------
# The Spire of Aldric
# ---------------------------------------------------------------------------------------------------------------------

def wizard_tower(variant, seed):
    rng = _rng("wizard_tower", variant, seed)
    k = Kit()
    H = 24.0
    lean = np.array([0.7, 0.25])
    zs = np.linspace(-0.5, H, 13)
    lean_at = lambda z: lean * (np.clip(z, 0, H) / H) ** 2  # noqa: E731
    path = np.array([[*lean_at(z), z] for z in zs])
    radii = 3.4 - 0.8 * (np.clip(zs, 0, H) / H) + 0.25 * np.sin(zs * 0.9) * 0
    for lo, hi, sides in ((0, 0, 32), (1, 1, 24), (2, 3, 16), (4, 5, 10), (6, 7, 6)):
        k.add("stone_blocks", tube(path, radii, sides, v_per_m=UV, u_repeat=8.0), lo, hi)
    # Corbelled rings and the balcony at two thirds of the height.
    for z, dr, lo, hi in ((0.0, 0.5, 0, 4), (8.0, 0.25, 0, 2), (16.0, 1.1, 0, 5), (H, 0.6, 0, 5)):
        c = lean_at(z)
        r = 3.4 - 0.8 * z / H
        k.add("stone_blocks", lathe([r, r + dr, r + dr, r], [z - 0.6, z, z + 0.35, z + 0.35], 24 if hi > 3 else 16)
              .transformed(mat_trs((c[0], c[1], 0))), lo, hi)
    zb = 16.35
    cb = lean_at(zb)
    rb = 3.4 - 0.8 * 16 / H + 1.0
    for i in range(16):
        a = 2 * np.pi * i / 16
        p = (cb[0] + np.cos(a) * rb, cb[1] + np.sin(a) * rb)
        k.add("timber", beam((p[0], p[1], zb), (p[0], p[1], zb + 1.0), 0.09), 0, 2)
    ring = np.array([[cb[0] + np.cos(a) * rb, cb[1] + np.sin(a) * rb, zb + 1.0] for a in np.linspace(0, 2 * np.pi, 33)])
    k.add("timber", tube(ring, np.full(33, 0.06), 5, v_per_m=UV), 0, 3)
    # The spire: a curved, flared cone of slate whose tip bends away from the lean.
    t = np.linspace(0, 1, 12)
    tip_off = np.array([1.6, 0.6])
    sp = np.array([[*(lean_at(H) + tip_off * t ** 2.2), H + 0.3 + 12.5 * t] for t in t])
    sr = (3.0 - 0.8 * 1) * (1 - t) ** 1.3 + 0.03
    sr[0] += 0.9
    for lo, hi, sides in ((0, 0, 28), (1, 1, 20), (2, 3, 14), (4, 5, 8), (6, 7, 5)):
        k.add("roof_slate", tube(sp, sr, sides, v_per_m=UV, u_repeat=6.0), lo, hi)
    # Dormers on the spire and windows climbing the tower, their panes glowing.
    for i in range(14):
        z = 2.5 + i * 1.5
        if z > H - 1.5:
            break
        a = i * 2.3 + 0.4
        c = lean_at(z)
        r = 3.4 - 0.8 * z / H
        pos = (c[0] + np.cos(a) * (r - 0.05), c[1] + np.sin(a) * (r - 0.05), 0)
        k.add("timber", box((-0.45, -0.12, z - 0.1), (0.45, 0.12, z + 1.3), UV).transformed(mat_trs(pos, a + np.pi / 2)), 0, 3)
        k.add("crystal", box((-0.32, -0.16, z), (0.32, 0.16, z + 1.15), UV).transformed(mat_trs(pos, a + np.pi / 2)), 0, 5)
    for a in (0.3, 2.4, 4.5):
        z = H + 2.2
        c = lean_at(H) + tip_off * 0.02
        r = 2.4
        pos = np.array([c[0] + np.cos(a) * r, c[1] + np.sin(a) * r, 0])
        k.add("roof_slate", lathe([0.85, 0.05], [z + 1.2, z + 2.2], 4).transformed(mat_trs(pos, a + np.pi / 4)), 0, 3)
        k.add("timber", box((-0.6, -0.4, z - 0.1), (0.6, 0.4, z + 1.2), UV).transformed(mat_trs(pos, a + np.pi / 2)), 0, 3)
        k.add("crystal", box((-0.4, -0.44, z), (0.4, 0.44, z + 1.0), UV).transformed(mat_trs(pos, a + np.pi / 2)), 0, 4)
    # The door, its stair, and a glowing lantern above it.
    k.add("wood", box((-0.8, -3.55, 0.3), (0.8, -3.0, 2.9), UV), 0, 6)
    for i in range(3):
        k.add("stone_blocks", box((-1.4 + i * 0.15, -4.6 + i * 0.35, -0.4), (1.4 - i * 0.15, -3.2, 0.1 + i * 0.1), UV), 0, 3)
    k.add("crystal", lathe([0.0, 0.25, 0.25, 0.0], [3.2, 3.4, 3.8, 4.0], 6).transformed(mat_trs((0, -3.75, 0))), 0, 2)
    # The floating heart: an octahedral crystal above the tip, and rock shards orbiting the upper tower.
    top = np.array([*(lean_at(H) + tip_off), H + 0.3 + 12.5 + 2.6])
    for lo, hi, sides in ((0, 3, 8), (4, 7, 4)):
        k.add("crystal", lathe([0.0, 1.0, 0.0], [-1.6, 0.0, 1.6], sides).transformed(mat_trs(top, 0.3)), lo, hi)
    for i in range(7):
        a = 2 * np.pi * i / 7 + rng.uniform(-0.3, 0.3)
        r = rng.uniform(5.5, 8.0)
        z = rng.uniform(15.0, 30.0)
        size = rng.uniform(0.5, 1.3)
        pos = (np.cos(a) * r + lean_at(z)[0], np.sin(a) * r + lean_at(z)[1], z)
        s = _shard(rng, size, size * 2.2, 7)
        tilt = rng.uniform(-0.6, 0.6, 2)
        lo_hi = (0, 5) if size > 0.8 else (0, 3)
        k.add("rock", s.transformed(mat_trs(pos, rng.uniform(0, 6), 1.0, *tilt)), *lo_hi)
        if size > 0.9:
            k.add("crystal", _crystal(np.zeros(3), (0.2, 0.1, 1.0), size * 1.4, size * 0.22)
                  .transformed(mat_trs(pos, 0.0, 1.0, *tilt)), 0, 3)
    return k.levels()


# ---------------------------------------------------------------------------------------------------------------------
# Runestones
# ---------------------------------------------------------------------------------------------------------------------

RUNESTONES = {   # width, depth, height, glyph rows, columns, top taper
    "a": (1.0, 0.38, 3.3, 7, 1, 0.75),
    "b": (1.5, 0.5, 2.0, 3, 2, 0.9),
    "way": (0.5, 0.32, 1.25, 2, 1, 0.85),
}


def _glyph(rng, cx, cz, size, y):
    """One rune: 2-4 short strokes (stave and twigs) in the plane y = const, slightly proud of the face."""
    strokes = [((0, -0.5), (0, 0.5))]
    for _ in range(rng.integers(1, 4)):
        z0 = rng.uniform(-0.4, 0.4)
        sgn = rng.choice([-1, 1])
        strokes.append(((0, z0), (sgn * 0.4, z0 + rng.choice([-0.35, 0.35]))))
    out = []
    for (x0, z0), (x1, z1) in strokes:
        out.append(beam((cx + x0 * size, y, cz + z0 * size), (cx + x1 * size + 1e-4, y, cz + z1 * size), 0.035, 0.06))
    return out


def runestone(variant, seed):
    rng = _rng("runestone", variant, seed)
    k = Kit()
    W, D, H, rows, cols, taper = RUNESTONES[variant]
    lean = (np.radians(rng.normal(0, 2.0)), np.radians(rng.normal(0, 2.0)))
    M = mat_trs((0, 0, 0), 0.0, 1.0, *lean)

    def hewn(sides_jitter, sides=4):
        m = prism(0.5 / np.cos(np.pi / sides), H + 0.5, sides, z0=-0.5, top_scale=taper, cap=True, uv_scale=UV)
        m = m.transformed(mat_trs((0, 0, 0), np.pi / sides)).transformed(mat_trs((0, 0, 0), 0.0, (W, D, 1.0)))
        if sides_jitter:
            sd = rng.uniform(0, 100)
            key = np.round(m.points, 5)
            m.points[:, :2] *= (1 + sides_jitter * _pos_noise(key, sd))[:, None]
            m.points[:, 2] += np.where(key[:, 2] > H - 0.1, 0.12 * _pos_noise(key, sd + 1), 0.0)
        return m.transformed(M)

    k.add("rock", hewn(0.06, 8), 0, 0)
    k.add("rock", hewn(0.06), 1, 5)
    k.add("rock", hewn(0.0), 6, 7)
    # A domed cap on the waystones.
    if variant == "way":
        k.add("rock", lathe([0.32, 0.3, 0.0], [H, H + 0.12, H + 0.22], 8).transformed(M), 0, 4)
    # Runes on both broad faces; each level keeps fewer glyph strokes, then one glowing band.
    for side in (-1, 1):
        yface = side * (D / 2 * (1 + taper) / 2 + 0.01)
        gl_all, gl_half = [], []
        for c in range(cols):
            for r in range(rows):
                cx = (c - (cols - 1) / 2) * W / cols * 0.9
                cz = 0.35 + (r + 0.5) * (H * 0.75) / rows
                size = min(W / cols * 0.55, H * 0.75 / rows * 0.7)
                g = _glyph(rng, cx, cz, size, yface)
                gl_all += g
                if r % 2 == 0:
                    gl_half.append(box((cx - size * 0.06, yface - 0.02, cz - size * 0.5), (cx + size * 0.06, yface + 0.02, cz + size * 0.5), UV))
        k.add_all("crystal", [g.transformed(M) for g in gl_all], 0, 1)
        k.add_all("crystal", [g.transformed(M) for g in gl_half], 2, 2)
        band = box((-W * 0.08, yface - 0.02, 0.35), (W * 0.08, yface + 0.02, 0.35 + H * 0.75), UV)
        k.add("crystal", band.transformed(M), 3, 6)
    if variant == "b":
        # A ring carved through the broad stone, glowing at its rim.
        ring = np.array([[np.cos(t) * W * 0.28, -D / 2 - 0.02, H * 0.62 + np.sin(t) * W * 0.28] for t in np.linspace(0, 2 * np.pi, 25)])
        k.add("crystal", tube(ring, np.full(25, 0.05), 6, v_per_m=UV).transformed(M), 0, 4)
    return k.levels()


# ---------------------------------------------------------------------------------------------------------------------
# Crystal clusters
# ---------------------------------------------------------------------------------------------------------------------

def crystal_cluster(variant, seed):
    rng = _rng("crystal", variant, seed)
    k = Kit()
    n, lmax, rmax, splay = (10, 4.5, 0.6, 0.55) if variant == "a" else (6, 2.2, 0.32, 0.8)
    xs = []
    for i in range(n):
        f = 1.0 if i == 0 else rng.uniform(0.25, 0.85)
        az = rng.uniform(0, 2 * np.pi)
        tilt = 0.0 if i == 0 else rng.uniform(0.15, splay)
        d = np.array([np.cos(az) * np.sin(tilt), np.sin(az) * np.sin(tilt), np.cos(tilt)])
        base = np.array([np.cos(az), np.sin(az), 0.0]) * rng.uniform(0.0, rmax * 1.5) * (i > 0)
        xs.append((f, base, d))
    xs.sort(key=lambda x: -x[0])
    # Levels drop the smallest crystals first and use fewer sides; the coarsest keeps a single three-sided spike.
    keep = [n, n - 1, n - 2, n - 3, max(3, n - 5), min(4, max(3, n - 5)), 2, 1]
    sides = [8, 6, 6, 5, 4, 4, 4, 3]
    for li in range(8):
        for f, base, d in xs[:keep[li]]:
            k.add("crystal", _crystal(base, d, lmax * f, rmax * (0.45 + 0.55 * f), sides[li]), li, li)
    # Rock lumps the crystals break out of.
    for i in range(3 if variant == "a" else 2):
        a = rng.uniform(0, 2 * np.pi)
        s = _shard(rng, rmax * 1.6, rmax * 2.2, 7, 0.2)
        k.add("rock", s.transformed(mat_trs((np.cos(a) * rmax, np.sin(a) * rmax, 0.0), a)), 0, 4)
    return k.levels()


# ---------------------------------------------------------------------------------------------------------------------
# Wyrm's Rest: the bones of a dragon, half sunk into the downs
# ---------------------------------------------------------------------------------------------------------------------

def dragon_bones(variant, seed):
    rng = _rng("dragon_bones", variant, seed)
    k = Kit()
    # The spine: an S-curve from the skull (x = -21) to the tail tip (x = +26), rising over the ribcage.
    u = np.linspace(0, 1, 60)
    sx = -21 + 47 * u
    sy = 3.0 * np.sin(u * 5.0 + 0.4) * (0.3 + u)
    sz = 0.25 + 1.6 * np.exp(-((u - 0.28) / 0.17) ** 2) - 0.2 * u
    spine = np.stack([sx, sy, sz], 1)
    sr = 0.55 * (1 - u) ** 0.8 + 0.06
    # Vertebrae with spinous processes (lod0-1), centra only (lod2-3), the spine as a tube (lod4-7).
    for i in range(1, 59):
        p, q = spine[i - 1], spine[i + 1]
        d = (q - p) / np.linalg.norm(q - p)
        L = np.linalg.norm(spine[i] - spine[i - 1]) * 0.42
        c = spine[i]
        k.add("bone", capped_tube(np.array([c - d * L, c, c + d * L]), np.array([sr[i] * 0.85, sr[i], sr[i] * 0.85]), 8), 0, 0)
        k.add("bone", capped_tube(np.array([c - d * L, c, c + d * L]), np.array([sr[i] * 0.85, sr[i], sr[i] * 0.85]), 6), 1, 1)
        k.add("bone", capped_tube(np.array([c - d * L, c + d * L]), np.array([sr[i], sr[i]]), 5), 2, 3)
        if u[i] < 0.75:
            k.add("bone", beam(c, c + np.array([0.0, 0.0, sr[i] * 2.2 + 0.4]) + d * 0.3, sr[i] * 0.35, sr[i] * 0.12), 0, 1)
    for lo, hi, sides, step in ((4, 4, 6, 2), (5, 5, 5, 3), (6, 6, 4, 4), (7, 7, 3, 8)):
        k.add("bone", capped_tube(spine[::step], sr[::step] * 0.9, sides), lo, hi)
    # Ribs: arcs out and up from the chest vertebrae, curving in and down to the ground (a broken cage).
    chest = [i for i in range(8, 30, 2)]
    for j, i in enumerate(chest):
        c = spine[i]
        L = 4.2 + 2.2 * np.sin(np.pi * j / len(chest))
        for side in (-1, 1):
            if rng.uniform() < 0.12:
                continue    # a missing rib
            t = np.linspace(0, 1, 14)
            ang = 0.15 + t * 2.4
            pts = np.stack([c[0] + 0.6 * t * (1 + 0.2 * side), c[1] + side * L * np.sin(ang) * 1.05,
                            c[2] + L * 1.25 * np.cos(ang) * (1 - 0.15 * t) + 0.3], 1)
            pts[:, 2] = np.maximum(pts[:, 2], -0.4)
            rr = 0.22 * (1 - 0.6 * t)
            k.add("bone", capped_tube(pts, rr, 8), 0, 0)
            k.add("bone", capped_tube(pts[::2], rr[::2], 7), 1, 1)
            k.add("bone", capped_tube(pts[::2], rr[::2], 6), 2, 3)
            k.add("bone", capped_tube(pts[::4], rr[::4], 4), 4, 4)
            k.add("bone", capped_tube(pts[::4], rr[::4], 3), 5, 5)
            if j % 2 == 0:
                k.add("bone", capped_tube(pts[::6], rr[::6], 3), 6, 6)
            if j % 4 == 0:
                k.add("bone", capped_tube(pts[::13], rr[::13], 3), 7, 7)
    # The skull: a long tapering snout, a heavy brow, a sunken jaw and two swept horns.
    head = spine[0] + np.array([-0.5, 0.0, 0.1])
    hx = np.linspace(0, -7.5, 10)
    hp = np.stack([head[0] + hx, head[1] + 0.25 * np.sin(np.linspace(0, 1, 10)), head[2] + 0.9 - 0.08 * np.linspace(0, 1, 10) ** 2 * 7], 1)
    hr = np.array([1.5, 1.7, 1.55, 1.2, 1.0, 0.85, 0.7, 0.6, 0.45, 0.12])
    for lo, hi, sides, step in ((0, 1, 12, 1), (2, 3, 8, 1), (4, 5, 6, 2), (6, 7, 4, 3)):
        sel = np.r_[np.arange(0, 9, step), 9]
        k.add("bone", capped_tube(hp[sel], hr[sel] * 0.85, sides), lo, hi)
    jaw = hp.copy()
    jaw[:, 2] -= 1.3
    jaw[:, 1] += 0.6
    k.add("bone", capped_tube(jaw[1:], hr[1:] * 0.35, 8), 0, 3)
    for side in (-1, 1):
        t = np.linspace(0, 1, 10)
        horn = np.stack([head[0] + 1.0 + 5.5 * t, head[1] + side * (1.0 + 2.2 * t), head[2] + 1.7 + 2.4 * np.sin(t * 2.2)], 1)
        hr2 = 0.45 * (1 - t) + 0.03
        k.add("bone", capped_tube(horn, hr2, 8), 0, 2)
        k.add("bone", capped_tube(horn[::3], hr2[::3], 4), 3, 6)
        # Teeth along the jaw line.
        for x in np.linspace(-1.5, -6.5, 8):
            p = np.array([head[0] + x, head[1] + side * 0.55 * (1 + x / 9), head[2] - 0.05])
            k.add("bone", capped_tube(np.array([p, p + (0, 0, -0.55)]), np.array([0.09, 0.0]), 4), 0, 1)
    # Wings: the arm bones splayed on the ground, with finger bones fanning out (the membrane long gone).
    for side in (-1, 1):
        sh = spine[14] + np.array([0.0, side * 1.0, 0.0])
        elbow = sh + np.array([-3.0, side * 7.5, -0.6])
        wrist = elbow + np.array([6.5, side * 6.0, -0.4])
        arm = np.array([sh, (sh + elbow) / 2 + (0, 0, 0.8), elbow, (elbow + wrist) / 2 + (0, 0, 0.3), wrist])
        k.add("bone", capped_tube(arm, np.array([0.35, 0.3, 0.32, 0.25, 0.22]), 8), 0, 3)
        k.add("bone", capped_tube(arm[::2], np.array([0.35, 0.32, 0.22]), 4), 4, 6)
        for f in range(4):
            a = np.radians(-30 + f * 28) * side
            d = np.array([np.cos(a), side * np.sin(a) + side * 0.4, 0.0])
            d /= np.linalg.norm(d)
            L = 9.0 - f * 1.2
            t = np.linspace(0, 1, 7)
            fp = wrist + np.outer(t, d * L) + np.outer(np.sin(t * np.pi), (0, 0, 0.6)) - np.outer(t, (0, 0, 0.5))
            k.add("bone", capped_tube(fp, 0.16 * (1 - 0.7 * t), 6), 0, 2)
            k.add("bone", capped_tube(fp[::3], 0.14 * (1 - 0.7 * t[::3]), 4), 3, 5)
    return k.levels()
