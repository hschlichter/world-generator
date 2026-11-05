"""Points of interest: each kind is a deterministic arrangement of structure / rock instances around its centre."""

import numpy as np

from .scatter import Placement
from .assets.rocks import hash_str


def _footprint_z(world, x, y, r):
    """Lowest terrain under a footprint of radius r, so structures never float on slopes."""
    a = np.linspace(0, 2 * np.pi, 9)[:-1]
    xs = np.r_[x, x + np.cos(a) * r]
    ys = np.r_[y, y + np.sin(a) * r]
    return float(world.height(xs, ys).min())


def layout(poi, world, seed):
    kind = poi["kind"]
    cx, cy = poi["position"]
    rng = np.random.default_rng([seed, hash_str(poi["name"])])
    items = []  # (asset, variant, lx, ly, rz_deg, scale, tilt(x,y), footprint radius, sink)

    def add(asset, variant, lx, ly, rz, s=1.0, tilt=(0.0, 0.0), fr=1.0, sink=0.2):
        items.append((asset, variant, lx, ly, rz, s, tilt, fr, sink))

    if kind == "ruin_keep":
        half = 22.0
        rot = rng.uniform(0, 90)
        for side in range(4):
            ang = np.radians(rot + side * 90)
            d = np.array([np.cos(ang), np.sin(ang)])
            nrm = np.array([-d[1], d[0]])
            for k in range(5):
                if side == 0 and k == 2:
                    continue  # the gate
                t = -half + 4.4 + k * 8.8
                p = nrm * half + d * t
                st = rng.choice(["ruined", "collapsed", "ruined", "intact"], p=[0.45, 0.3, 0.15, 0.1])
                add("wall", st, p[0], p[1], np.degrees(ang), 1.1, fr=4.5, sink=0.4)
            c = nrm * half + d * half
            add("tower", "ruined" if side % 2 else "intact", c[0], c[1], rng.uniform(0, 360), 1.0, fr=3.5, sink=0.5)
        add("tower", "ruined", 0.0, 0.0, rng.uniform(0, 360), 1.35, fr=4.5, sink=0.6)
        for i in range(6):
            a = np.radians(rot + 30 + i * 60)
            add("column", rng.choice(["intact", "broken"]), np.cos(a) * 12, np.sin(a) * 12, rng.uniform(0, 360), 1.0,
                fr=0.7)
        for i in range(10):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(4, 30)
            add("boulder", "angular", np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360), rng.uniform(0.3, 0.7),
                (rng.normal(0, 10), rng.normal(0, 10)), fr=0.5, sink=0.1)
    elif kind == "stone_circle":
        n = 9
        for i in range(n):
            a = 2 * np.pi * i / n + rng.normal(0, 0.03)
            r = 12.0 + rng.normal(0, 0.3)
            fallen = i == 4
            tilt = (rng.normal(0, 3), 78.0) if fallen else (rng.normal(0, 4), rng.normal(0, 4))
            add("standing_stone", ["a", "b", "c"][i % 3], np.cos(a) * r, np.sin(a) * r, np.degrees(a) + 90,
                rng.uniform(1.1, 1.5), tilt, fr=0.6, sink=0.35)
        add("boulder", "flat", 0.0, 0.0, rng.uniform(0, 360), 1.3, fr=1.5, sink=0.1)
    elif kind == "tower":
        add("tower", "ruined", 0.0, 0.0, rng.uniform(0, 360), 1.15, fr=4.0, sink=0.6)
        for i in range(3):
            a = rng.uniform(0, 2 * np.pi)
            add("wall", "collapsed", np.cos(a) * 9, np.sin(a) * 9, np.degrees(a) + 90, 0.9, fr=4.0, sink=0.4)
    elif kind == "rock_arch":
        add("rock_arch", "a", 0.0, 0.0, rng.uniform(0, 360), 1.0, fr=8.0, sink=2.0)
    elif kind == "shrine":
        add("shrine", "collapsed", 0.0, 0.0, rng.uniform(0, 360), 1.0, fr=4.5, sink=0.3)
        for i in range(3):
            a = rng.uniform(0, 2 * np.pi)
            add("column", "broken", np.cos(a) * 9, np.sin(a) * 9, rng.uniform(0, 360), 1.0, fr=0.7)
    elif kind == "colonnade":
        rot = rng.uniform(0, 180)
        d = np.array([np.cos(np.radians(rot)), np.sin(np.radians(rot))])
        nrm = np.array([-d[1], d[0]])
        for row in (-1, 1):
            for k in range(8):
                p = d * (-21 + k * 6) + nrm * row * 5
                st = "intact" if rng.uniform() < 0.55 else "broken"
                add("column", st, p[0], p[1], rng.uniform(0, 360), 1.1, fr=0.7)
        add("wall", "ruined", *(d * 26), rot + 90, 1.0, fr=4.0, sink=0.4)
    elif kind in ("village", "hamlet"):
        _settlement(poi, world, rng, add)
    elif kind == "castle":
        _castle(poi, world, rng, add)
    elif kind == "windmill":
        add("windmill", "a", 0.0, 0.0, rng.uniform(0, 360), 1.0, fr=3.6, sink=0.4)
        a = rng.uniform(0, 2 * np.pi)
        hx, hy = np.cos(a) * 15, np.sin(a) * 15
        add("house", "thatch", hx, hy, np.degrees(a) - 90 + rng.normal(0, 5), 1.0, fr=4.5, sink=0.45)
        for i in range(3):
            b = a + 2.2 + i * 0.5
            add("haystack", "a", np.cos(b) * 12, np.sin(b) * 12, rng.uniform(0, 360), rng.uniform(0.8, 1.1), fr=1.8, sink=0.2)
        _fence_ring(add, rng, 19.0, a + 0.6, a + 4.6, broken=0.2)
    elif kind == "graveyard":
        rot = rng.uniform(0, 360)
        R = np.radians(rot)
        d, nrm = np.array([np.cos(R), np.sin(R)]), np.array([-np.sin(R), np.cos(R)])
        add("house", "chapel", *(nrm * 9.0), rot, 1.0, fr=6.0, sink=0.5)
        for row in range(5):
            for col in range(7):
                if rng.uniform() < 0.18:
                    continue
                p = d * (-9 + col * 3.0 + rng.normal(0, 0.25)) + nrm * (-4 - row * 2.6 + rng.normal(0, 0.2))
                add("gravestone", rng.choice(["cross", "slab", "round"], p=[0.3, 0.4, 0.3]), p[0], p[1],
                    rot + rng.normal(0, 6), rng.uniform(0.85, 1.15), fr=0.5, sink=0.05)
        for k in range(14):   # a low, tumbled churchyard wall around it
            a = 2 * np.pi * k / 14 + R
            p = np.array([np.cos(a) * 21, np.sin(a) * 21])
            if k == 10:
                continue      # the lych gate gap
            add("wall", "collapsed", p[0], p[1], np.degrees(a) + 90, 1.0, fr=4.0, sink=0.4)
        add("wayside_shrine", "a", *(d * 12 - nrm * 14), rot, 1.1, fr=0.5, sink=0.15)
    elif kind == "wizard_tower":
        add("wizard_tower", "a", 0.0, 0.0, rng.uniform(0, 360), 1.0, fr=4.0, sink=0.5)
        for i in range(5):
            a = 2 * np.pi * i / 5 + rng.uniform(-0.1, 0.1)
            add("runestone", "a" if i % 2 else "b", np.cos(a) * 15, np.sin(a) * 15, np.degrees(a) + 90,
                rng.uniform(0.9, 1.2), (rng.normal(0, 2), rng.normal(0, 2)), fr=0.8, sink=0.4)
        for i in range(4):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(7, 24)
            add("crystal", "b", np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360), rng.uniform(0.7, 1.2), fr=1.0, sink=0.3)
        for i in range(6):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(18, 28)
            add("boulder", "angular", np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360), rng.uniform(0.5, 1.2),
                (rng.normal(0, 10), rng.normal(0, 10)), fr=0.6, sink=0.15)
    elif kind == "dragon_bones":
        add("dragon_bones", "a", 0.0, 0.0, rng.uniform(0, 360), 1.0, fr=1.0, sink=0.6)
        for i in range(9):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(15, 42)
            add("boulder", rng.choice(["round", "flat"]), np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360),
                rng.uniform(0.5, 1.5), (rng.normal(0, 8), rng.normal(0, 8)), fr=0.6, sink=0.2)
        for i in range(3):
            a = rng.uniform(0, 2 * np.pi)
            add("deadwood", "a" if i % 2 else "b", np.cos(a) * 30, np.sin(a) * 30, rng.uniform(0, 360), 0.9, fr=0.5, sink=0.1)
        add("runestone", "a", 26.0, -8.0, rng.uniform(0, 360), 1.3, fr=0.8, sink=0.4)
    elif kind == "crystal_hollow":
        for i in range(12):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, 34) ** 0.9
            add("crystal", "a" if i < 6 else "b", np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360),
                rng.uniform(0.7, 1.6) if i < 6 else rng.uniform(0.8, 1.3),
                (rng.normal(0, 12), rng.normal(0, 12)), fr=1.0, sink=0.5)
        for i in range(10):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(10, 40)
            add("boulder", "angular", np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360), rng.uniform(0.8, 2.2),
                (rng.normal(0, 15), rng.normal(0, 15)), fr=0.8, sink=0.3)
    elif kind == "runestones":
        n = 7
        for i in range(n):
            a = 2 * np.pi * i / n + rng.normal(0, 0.04)
            add("runestone", "a" if i % 3 else "b", np.cos(a) * 14, np.sin(a) * 14, np.degrees(a) + 90,
                rng.uniform(1.0, 1.3), (rng.normal(0, 3), rng.normal(0, 3)), fr=0.8, sink=0.4)
        add("runestone", "b", 0.0, 0.0, rng.uniform(0, 360), 1.5, fr=1.0, sink=0.4)
        add("crystal", "b", 3.0, 2.0, rng.uniform(0, 360), 0.8, fr=0.6, sink=0.2)
    elif kind == "rock_pillars":
        for i in range(8):
            a = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0, 34)
            add("pillar", rng.choice(["a", "b"]), np.cos(a) * r, np.sin(a) * r, rng.uniform(0, 360),
                rng.uniform(0.7, 1.4), (rng.normal(0, 3), rng.normal(0, 3)), fr=2.0, sink=1.0)
    else:
        raise ValueError(f"unknown POI kind {kind}")

    asset, variant, pos, rz, sc, tilt = [], [], [], [], [], []
    for a, v, lx, ly, r, s, t, fr, sink in items:
        x, y = cx + lx, cy + ly
        z = _footprint_z(world, x, y, fr * s) - sink * s
        asset.append(a)
        variant.append(v)
        pos.append((x, y, z))
        rz.append(r)
        sc.append(s)
        tilt.append(t)
    return Placement(np.array(asset, object), np.array(variant, object), np.array(pos, float), np.array(rz, float),
                     np.array(sc, float), np.array(tilt, float))


def _fence_ring(add, rng, r, a0, a1, broken=0.15, cx=0.0, cy=0.0):
    """Fence panels (6 m) along an arc of radius r from angle a0 to a1."""
    n = max(1, int((a1 - a0) * r / 6.0))
    for i in range(n):
        a = a0 + (i + 0.5) * (a1 - a0) / n
        add("fence", "broken" if rng.uniform() < broken else "intact", cx + np.cos(a) * r, cy + np.sin(a) * r,
            np.degrees(a) + 90, 1.0, fr=0.4, sink=0.1)


def _settlement(poi, world, rng, add):
    """A village or hamlet around a green with a well: houses face the green (or the road through it), each with a
    fenced garden plot; haystacks on the outskirts; villages get a chapel and banners at the green."""
    R = float(poi["radius"])
    n = int(poi.get("houses", 8))
    village = poi["kind"] == "village"
    add("well", "a", 0.0, 0.0, rng.uniform(0, 360), 1.0, fr=1.5, sink=0.2)
    styles = ["thatch", "thatch", "long", "slate"] if village else ["thatch", "thatch", "long"]
    placed = []
    rings = [(0.42, 0.55), (0.62, 0.45), (0.82, 0.0)]
    k = 0
    if village:
        a = rng.uniform(0, 2 * np.pi)
        p = np.array([np.cos(a), np.sin(a)]) * R * 0.3
        add("house", "chapel", p[0], p[1], np.degrees(a) - 90, 1.0, fr=6.5, sink=0.5)
        placed.append((p, 9.0))
        for sx in (-1, 1):
            q = p * 0.55 + np.array([-np.sin(a), np.cos(a)]) * 3.5 * sx
            add("banner", "red" if sx < 0 else "blue", q[0], q[1], np.degrees(a), 1.0, fr=0.3, sink=0.3)
    tries = 0
    while k < n and tries < 400:
        tries += 1
        f = rings[min(k * len(rings) // max(n, 1), len(rings) - 1)][0]
        a = rng.uniform(0, 2 * np.pi)
        r = R * (f + rng.uniform(-0.08, 0.08))
        p = np.array([np.cos(a), np.sin(a)]) * r
        if any(np.linalg.norm(p - q) < rr + 8.5 for q, rr in placed):
            continue
        style = styles[int(rng.integers(len(styles)))]
        face = np.degrees(a) - 90 + rng.normal(0, 8)    # the door (-y) faces the green
        add("house", style, p[0], p[1], face, rng.uniform(0.92, 1.08), fr=5.0 if style != "long" else 7.5, sink=0.5)
        placed.append((p, 9.0 if style == "long" else 6.0))
        # A fenced plot behind the house.
        out = p / np.linalg.norm(p)
        side = np.array([-out[1], out[0]])
        g = p + out * 9.0
        for s0 in (-1, 1):
            q = g + side * s0 * 3.0
            add("fence", "intact" if rng.uniform() < 0.8 else "broken", q[0], q[1], np.degrees(np.arctan2(out[1], out[0])),
                1.0, fr=0.4, sink=0.1)
        q = g + out * 3.0
        add("fence", "intact", q[0], q[1], np.degrees(np.arctan2(side[1], side[0])), 1.0, fr=0.4, sink=0.1)
        if rng.uniform() < 0.45:
            q = g + out * 9.0 + side * rng.uniform(-6, 6)
            add("haystack", "a", q[0], q[1], rng.uniform(0, 360), rng.uniform(0.8, 1.15), fr=1.8, sink=0.2)
        k += 1
    if village:
        a = rng.uniform(0, 2 * np.pi)
        add("wayside_shrine", "a", np.cos(a) * R * 0.95, np.sin(a) * R * 0.95, rng.uniform(0, 360), 1.0, fr=0.5, sink=0.15)


def _castle(poi, world, rng, add):
    """A curtain wall (intact wall panels) around a bailey, round towers at the corners, a gatehouse facing the
    approach, the keep, banners on the walls and a well and two halls inside."""
    rot = rng.uniform(0, 360)
    R = np.radians(rot)
    d, nrm = np.array([np.cos(R), np.sin(R)]), np.array([-np.sin(R), np.cos(R)])
    hx, hy = 34.0, 26.0
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    loc = lambda u, v: d * u + nrm * v  # noqa: E731
    for i in range(4):
        (u0, v0), (u1, v1) = corners[i], corners[(i + 1) % 4]
        L = np.hypot(u1 - u0, v1 - v0)
        n = int(round((L - 6.0) / 8.0))
        ang = np.degrees(np.arctan2(*(loc(u1 - u0, v1 - v0)[::-1])))
        for j in range(n):
            t = (3.0 + (L - 6.0) / n * (j + 0.5)) / L
            u, v = u0 + (u1 - u0) * t, v0 + (v1 - v0) * t
            if i == 0 and abs(u) < 7:
                continue                                      # the gate
            p = loc(u, v)
            add("wall", "intact", p[0], p[1], ang, 1.0, fr=4.2, sink=0.5)
            if j % 2 == 1 and not (i == 0 and abs(u) < 12):
                # A banner standing on the wall walk (same footprint as the wall, raised to its top).
                add("banner", "red" if (i + j) % 4 else "blue", p[0], p[1], ang, 0.8, fr=4.2 / 0.8, sink=-3.9 / 0.8)
        p = loc(u0, v0)
        add("tower", "intact", p[0], p[1], rng.uniform(0, 360), 1.0, fr=3.6, sink=0.6)
    p = loc(0.0, -hy - 1.5)
    add("gatehouse", "a", p[0], p[1], rot, 1.0, fr=5.0, sink=0.5)
    p = loc(12.0, 8.0)
    add("keep", "a", p[0], p[1], rot + 180, 1.0, fr=7.0, sink=0.6)
    p = loc(-18.0, 14.0)
    add("house", "slate", p[0], p[1], rot + 180, 1.0, fr=5.0, sink=0.5)
    p = loc(-22.0, -10.0)
    add("house", "long", p[0], p[1], rot + 90, 1.0, fr=7.5, sink=0.5)
    p = loc(-4.0, -6.0)
    add("well", "a", p[0], p[1], rot, 1.0, fr=1.5, sink=0.2)
    p = loc(12.0, -18.0)
    add("haystack", "a", p[0], p[1], rng.uniform(0, 360), 0.9, fr=1.8, sink=0.2)
    for s0 in (-1, 1):
        p = loc(s0 * 9.0, -hy - 9.0)
        add("banner", "red" if s0 < 0 else "blue", p[0], p[1], rot, 1.0, fr=0.3, sink=0.3)


def path_markers(cfg):
    """Waystones and wayside shrines beside every path, every `markers.spacing` metres, alternating sides and kinds,
    skipping stretches inside points of interest. Pure function of the configuration: [(x, y, kind, rz deg)]."""
    m = cfg.c.get("markers")
    if not m:
        return []
    sp, off = float(m["spacing"]), float(m["offset"])
    pois = [(np.asarray(p["position"], float), float(p["radius"]) + 25.0) for p in cfg.c.pois]
    out = []
    k = 0
    for path in cfg.c.get("paths", []):
        pts = np.asarray(path["points"], float)
        seg = np.diff(pts, axis=0)
        L = np.hypot(*seg.T)
        cum = np.r_[0, np.cumsum(L)]
        for s in np.arange(sp * 0.5, cum[-1], sp):
            i = min(np.searchsorted(cum, s) - 1, len(seg) - 1)
            t = (s - cum[i]) / L[i]
            d = seg[i] / L[i]
            side = 1 if k % 2 else -1
            p = pts[i] + seg[i] * t + np.array([-d[1], d[0]]) * side * (float(path["width"]) / 2 + off)
            k += 1
            if any(np.linalg.norm(p - c) < r for c, r in pois):
                continue
            h = abs(cfg.world_size / 2) - 5
            if abs(p[0]) > h or abs(p[1]) > h:
                continue
            kind = "runestone" if (k // 2) % 2 else "wayside_shrine"
            out.append((float(p[0]), float(p[1]), kind, float(np.degrees(np.arctan2(d[1], d[0])))))
    return out


def _markers_in(cfg, world, x0, y0, x1, y1):
    sel = [m for m in path_markers(cfg) if x0 <= m[0] < x1 and y0 <= m[1] < y1]
    if not sel:
        return None
    asset = np.array([k for _, _, k, _ in sel], object)
    variant = np.array(["way" if k == "runestone" else "a" for _, _, k, _ in sel], object)
    pos = np.array([(x, y, _footprint_z(world, x, y, 0.5) - 0.3) for x, y, _, _ in sel], float)
    return Placement(asset, variant, pos, np.array([r for *_, r in sel], float), np.ones(len(sel)), np.zeros((len(sel), 2)))


def structures_in(cfg, world, x0, y0, x1, y1):
    out = []
    for p in cfg.c.pois:
        px, py = p["position"]
        r = float(p["radius"]) + 20
        if px + r < x0 or px - r > x1 or py + r < y0 or py - r > y1:
            continue
        pl = layout(p, world, cfg.seed)
        m = (pl.pos[:, 0] >= x0) & (pl.pos[:, 0] < x1) & (pl.pos[:, 1] >= y0) & (pl.pos[:, 1] < y1)
        out.append(Placement(pl.asset[m], pl.variant[m], pl.pos[m], pl.rz[m], pl.scale[m], pl.tilt[m]))
    out.append(_markers_in(cfg, world, x0, y0, x1, y1))
    return Placement.concat(out)
