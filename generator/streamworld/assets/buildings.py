"""Medieval buildings and village furniture, 8 LOD levels each (see kit.py for how levels are built).

Parts are named after their library material: stone_blocks, plaster, timber, wood, thatch, roof_slate, cloth_red,
cloth_blue, canvas. Local frame: z up from the ground, x along the building's length, the door on -y.
"""

import numpy as np

from ..meshutil import box, merge, mat_trs, prism, tube
from .kit import UV, Kit, beam, gable, lathe, quad, slab, tri
from .rocks import hash_str

# The material each part name binds (parts are named after their material).
PART_MATERIALS = {p: p for p in ("stone_blocks", "plaster", "timber", "wood", "thatch", "roof_slate", "cloth_red",
                                 "cloth_blue", "canvas", "crystal", "bone", "rock")}
PART_MATERIALS["rock"] = "rock_cliff"


def _rng(name, variant, seed):
    return np.random.default_rng([seed, hash_str(name), hash_str(variant)])


# ---------------------------------------------------------------------------------------------------------------------
# Houses
# ---------------------------------------------------------------------------------------------------------------------

HOUSES = {
    #          length, width, wall height, roof rise, roof, walls, frame spacing, chimney
    "thatch": (8.0, 5.4, 2.7, 3.6, "thatch", "plaster", 1.2, True),
    "slate": (9.0, 6.0, 5.0, 3.2, "roof_slate", "plaster", 1.1, True),
    "long": (15.0, 6.0, 2.6, 3.8, "thatch", "plaster", 1.3, True),
    "chapel": (13.0, 6.5, 5.4, 4.4, "roof_slate", "stone_blocks", 0.0, False),
}


def _walls_box(L, W, H):
    return box((-L / 2, -W / 2, 0), (L / 2, W / 2, H), UV)


def _gable_ends(L, W, H, rise, inset=0.0):
    """Triangular wall infill at both gable ends."""
    out = []
    for sx in (-1, 1):
        x = sx * (L / 2 - inset)
        a, b, c = (x, -W / 2, H), (x, W / 2, H), (x, 0.0, H + rise)
        out.append(tri(a, b, c) if sx > 0 else tri(b, a, c))
    return out


def _roof_planes(L, W, H, rise, over, thick):
    x0, x1, y, ze, zr = gable(L, W, H, rise, over)
    s = slab((x0, -y, ze), (x1, -y, ze), (x1, 0, zr), (x0, 0, zr), thick)
    n = slab((x1, y, ze), (x0, y, ze), (x0, 0, zr), (x1, 0, zr), thick)
    return [s, n], (x0, x1, y, ze, zr)


def _window(x, y_face, z, w, h, side, frame=True, shutters=True):
    """A window on a wall facing -y (side -1) or +y (side +1): dark recess, frame, sill, shutters."""
    d = 0.08 * side
    parts = {"timber": [box((x - w / 2, y_face - 0.02 * side, z), (x + w / 2, y_face + 0.02 * side, z + h), UV)]}
    if frame:
        t = 0.09
        parts["timber"] += [box((x - w / 2 - t, y_face, z - t), (x + w / 2 + t, y_face + d, z), UV),
                            box((x - w / 2 - t, y_face, z + h), (x + w / 2 + t, y_face + d, z + h + t), UV),
                            box((x - w / 2 - t, y_face, z), (x - w / 2, y_face + d, z + h), UV),
                            box((x + w / 2, y_face, z), (x + w / 2 + t, y_face + d, z + h), UV),
                            box((x - 0.02, y_face, z), (x + 0.02, y_face + d * 0.6, z + h), UV)]
    if shutters:
        for sx in (-1, 1):
            xs = x + sx * (w / 2 + 0.09 + w / 4)
            parts.setdefault("wood", []).append(box((xs - w / 4, y_face + d, z), (xs + w / 4, y_face + d + 0.04 * side, z + h), UV))
    return parts


def _sorted_y(lo, hi):
    return (min(lo, hi), max(lo, hi))


def house(variant, seed):
    rng = _rng("house", variant, seed)
    L, W, H, rise, roof, wallmat, fsp, chim = HOUSES[variant]
    k = Kit()
    over = 0.55 if roof == "thatch" else 0.35
    plinth = 0.45
    # Stone plinth: individual blocks (lod0-1), one band (lod2-5).
    blocks = []
    for side in range(4):
        along = L if side % 2 == 0 else W
        x = -along / 2
        while x < along / 2 - 0.05:
            lx = min(rng.uniform(0.45, 0.8), along / 2 - x)
            b = box((x + 0.015, -0.12, -0.3), (x + lx - 0.015, 0.12, plinth), UV)
            if side == 0:
                b = b.transformed(mat_trs((0, -W / 2, 0)))
            elif side == 2:
                b = b.transformed(mat_trs((0, W / 2, 0), np.pi))
            elif side == 1:
                b = b.transformed(mat_trs((L / 2, 0, 0), np.pi / 2))
            else:
                b = b.transformed(mat_trs((-L / 2, 0, 0), -np.pi / 2))
            blocks.append(b)
            x += lx
    if wallmat == "plaster":
        k.add_all("stone_blocks", blocks, 0, 0)
        k.add("stone_blocks", box((-L / 2 - 0.12, -W / 2 - 0.12, -0.3), (L / 2 + 0.12, W / 2 + 0.12, plinth), UV), 1, 5)
        k.add("plaster", box((-L / 2, -W / 2, plinth), (L / 2, W / 2, H), UV), 0, 6)
    else:
        k.add("stone_blocks", box((-L / 2, -W / 2, -0.3), (L / 2, W / 2, H), UV), 0, 6)
        # Buttresses along the nave.
        for x in np.linspace(-L / 2 + 1.5, L / 2 - 1.5, 4):
            for sy in (-1, 1):
                k.add("stone_blocks", box((x - 0.4, sy * W / 2 - (0.7 if sy < 0 else 0), -0.3),
                                          (x + 0.4, sy * W / 2 + (0.7 if sy > 0 else 0), H * 0.7), UV), 0, 4)
    gab = _gable_ends(L, W, H, rise)
    k.add_all(wallmat, gab, 0, 6)
    # Timber framing on plastered walls: posts, a mid rail, braces at the corners (lod0-2: all, lod3: posts only).
    if fsp > 0:
        t = 0.16
        for side, (y, s) in enumerate(((-W / 2, -1), (W / 2, 1))):
            yy = _sorted_y(y, y + s * 0.06)
            n = max(2, int(round(L / fsp)))
            for x in np.linspace(-L / 2 + t / 2, L / 2 - t / 2, n + 1):
                k.add("timber", box((x - t / 2, yy[0], plinth), (x + t / 2, yy[1], H), UV), 0, 3)
            for z in (plinth, H * 0.55 if H > 4 else H * 0.5, H - t):
                k.add("timber", box((-L / 2, yy[0], z), (L / 2, yy[1], z + t), UV), 0, 3)
            for x0 in (-L / 2 + t, L / 2 - t - fsp):
                for flip in (0, 1):
                    a = (x0 + (fsp if flip else 0), y + s * 0.03, plinth + 0.1)
                    b = (x0 + (0 if flip else fsp), y + s * 0.03, H * 0.5)
                    k.add("timber", beam(a, b, 0.05, 0.13), 0, 2)
            if H > 4:  # jetty beam of a two-storey house
                k.add("timber", box((-L / 2 - 0.1, yy[0] - 0.12 * (s < 0), H * 0.5), (L / 2 + 0.1, yy[1] + 0.12 * (s > 0), H * 0.5 + 0.22), UV), 0, 4)
        for sx in (-1, 1):
            x = sx * L / 2
            xx = _sorted_y(x, x + sx * 0.06)
            for yv in np.linspace(-W / 2, W / 2, 4):
                k.add("timber", box((xx[0], yv - t / 2, plinth), (xx[1], yv + t / 2, H), UV), 0, 3)
            k.add("timber", beam((x + sx * 0.03, -W / 2, H), (x + sx * 0.03, 0, H + rise * 0.98), 0.05, 0.14), 0, 3)
            k.add("timber", beam((x + sx * 0.03, W / 2, H), (x + sx * 0.03, 0, H + rise * 0.98), 0.05, 0.14), 0, 3)
    # Door and windows on the front (-y), windows on the back.
    dx = rng.uniform(-L * 0.2, L * 0.2) if variant != "chapel" else 0.0
    dw, dh = (1.1, 2.0) if variant != "chapel" else (1.6, 3.0)
    yd = -W / 2
    k.add("wood", box((dx - dw / 2, yd - 0.06, plinth * 0.4), (dx + dw / 2, yd + 0.02, plinth * 0.4 + dh), UV), 0, 4)
    k.add("timber", box((dx - dw / 2 - 0.15, yd - 0.1, plinth * 0.4), (dx - dw / 2, yd + 0.02, plinth * 0.4 + dh + 0.15), UV), 0, 2)
    k.add("timber", box((dx + dw / 2, yd - 0.1, plinth * 0.4), (dx + dw / 2 + 0.15, yd + 0.02, plinth * 0.4 + dh + 0.15), UV), 0, 2)
    k.add("timber", box((dx - dw / 2 - 0.15, yd - 0.1, plinth * 0.4 + dh), (dx + dw / 2 + 0.15, yd + 0.02, plinth * 0.4 + dh + 0.18), UV), 0, 2)
    k.add("stone_blocks", box((dx - dw / 2 - 0.3, yd - 0.7, -0.3), (dx + dw / 2 + 0.3, yd, plinth * 0.4), UV), 0, 3)
    nwin = max(1, int(L / 3.2))
    xs = [x for x in np.linspace(-L / 2 + 1.4, L / 2 - 1.4, nwin + 1) if abs(x - dx) > dw / 2 + 0.9]
    rows = [plinth + 0.9] + ([H * 0.5 + 0.7] if H > 4 else [])
    for z in rows:
        for side in (-1, 1):
            for x in xs:
                if variant == "chapel":
                    w, h = 0.5, 1.9
                    sh = False
                else:
                    w, h = 0.8, 0.85
                    sh = rng.uniform() < 0.7
                y = side * W / 2
                win = _window(x, y, z, w, h, side, frame=True, shutters=sh)
                k.add_all("timber", win["timber"][:1], 0, 4)
                k.add_all("timber", win["timber"][1:], 0, 0)
                k.add_all("wood", win.get("wood", []), 0, 2)
    # Roof.
    planes, (x0, x1, y, ze, zr) = _roof_planes(L, W, H, rise, over, 0.45 if roof == "thatch" else 0.16)
    k.add_all(roof, planes, 0, 3)
    thin, _ = _roof_planes(L, W, H, rise, over * 0.6, 0.05)
    k.add_all(roof, thin, 4, 6)
    if roof == "thatch":
        # A rolled ridge and a thicker, rounded eave course.
        ridge = np.stack([np.linspace(x0 - 0.05, x1 + 0.05, 9), np.zeros(9), np.full(9, zr + 0.05)], 1)
        k.add("thatch", tube(ridge, np.full(9, 0.32), 10, v_per_m=UV, u_repeat=1.0), 0, 0)
        k.add("thatch", tube(ridge[::2], np.full(5, 0.3), 6, v_per_m=UV, u_repeat=1.0), 1, 3)
        for sy in (-1, 1):
            eave = np.stack([np.linspace(x0, x1, 9), np.full(9, sy * (y - 0.15)), np.full(9, ze + 0.02)], 1)
            k.add("thatch", tube(eave, np.full(9, 0.2), 8, v_per_m=UV), 0, 0)
            k.add("thatch", tube(eave[::4], np.full(3, 0.2), 5, v_per_m=UV), 1, 2)
    else:
        k.add("roof_slate", beam((x0 - 0.05, 0, zr + 0.02), (x1 + 0.05, 0, zr + 0.02), 0.22, 0.18), 0, 3)
        # Barge boards along the gables.
        for sx, xe in ((-1, x0), (1, x1)):
            for sy in (-1, 1):
                k.add("timber", beam((xe, sy * y, ze), (xe, 0, zr), 0.06, 0.25), 0, 2)
    # Chimney (stone stack with a cap) or the chapel's bell gable.
    if chim:
        cx = L / 2 - 1.2 if variant != "long" else -L / 2 + 3.5
        k.add("stone_blocks", box((cx - 0.45, -0.45, H), (cx + 0.45, 0.45, zr + 1.1), UV), 0, 6)
        k.add("stone_blocks", box((cx - 0.55, -0.55, zr + 1.1), (cx + 0.55, 0.55, zr + 1.3), UV), 0, 3)
        k.add("stone_blocks", prism(0.14, 0.35, 6, z0=zr + 1.3, cap=True, uv_scale=UV).transformed(mat_trs((cx - 0.2, 0, 0))), 0, 1)
        k.add("stone_blocks", prism(0.14, 0.3, 6, z0=zr + 1.3, cap=True, uv_scale=UV).transformed(mat_trs((cx + 0.2, 0, 0))), 0, 1)
    else:
        xb = -L / 2
        bh = zr + 2.6
        k.add("stone_blocks", box((xb - 0.2, -1.1, H), (xb + 0.6, 1.1, bh), UV), 0, 6)
        k.add("stone_blocks", box((xb - 0.25, -1.2, bh), (xb + 0.65, 1.2, bh + 0.3), UV), 0, 3)
        k.add("timber", box((xb - 0.22, -0.45, zr + 0.6), (xb + 0.62, 0.45, zr + 1.8), UV), 0, 4)   # bell opening
        k.add("stone_blocks", tri((xb - 0.25, -1.2, bh + 0.3), (xb - 0.25, 1.2, bh + 0.3), (xb - 0.25, 0, bh + 1.4)), 0, 4)
        k.add("stone_blocks", tri((xb + 0.65, 1.2, bh + 0.3), (xb + 0.65, -1.2, bh + 0.3), (xb + 0.65, 0, bh + 1.4)), 0, 4)
        k.add("stone_blocks", beam((xb + 0.2, 0, bh + 1.3), (xb + 0.2, 0, bh + 2.5), 0.12), 0, 3)        # the cross
        k.add("stone_blocks", beam((xb + 0.2, -0.4, bh + 2.05), (xb + 0.2, 0.4, bh + 2.05), 0.1), 0, 3)
    # Coarsest: walls box and a gable prism (roof material), plus the chimney / bell gable above.
    k.add(wallmat, _walls_box(L, W, H), 7, 7)
    x0, x1, y, ze, zr = gable(L, W, H, rise, 0.0)
    roof7 = [quad((x0, -y, ze), (x1, -y, ze), (x1, 0, zr), (x0, 0, zr)),
             quad((x1, y, ze), (x0, y, ze), (x0, 0, zr), (x1, 0, zr))]
    k.add_all(roof, roof7, 7, 7)
    return k.levels()


# ---------------------------------------------------------------------------------------------------------------------
# Windmill (a stone tower mill with a turning cap and four lattice sails)
# ---------------------------------------------------------------------------------------------------------------------

def windmill(variant, seed):
    rng = _rng("windmill", variant, seed)
    k = Kit()
    R, H = 3.4, 11.0
    top = 0.68
    for li, sides in ((0, 32), (1, 20), (2, 14), (4, 10), (6, 6)):
        hi = {0: 0, 1: 1, 2: 3, 4: 5, 6: 7}[li]
        k.add("stone_blocks", prism(R, H, sides, top_scale=top, cap=True, uv_scale=UV), li, hi)
    # Plinth course and the stage (gallery) at the first floor.
    k.add("stone_blocks", prism(R + 0.25, 0.6, 24, cap=False, uv_scale=UV), 0, 3)
    zg = 4.2
    rg = R * (1 - (1 - top) * zg / H) + 1.3
    ring = [(np.cos(a) * rg, np.sin(a) * rg) for a in np.linspace(0, 2 * np.pi, 25)]
    k.add("wood", lathe([rg, rg, R * 0.95, R * 0.95], [zg, zg + 0.15, zg + 0.15, zg], 24), 0, 4)
    rail = np.array([[x, y, zg + 1.05] for x, y in ring])
    k.add("timber", tube(rail, np.full(len(rail), 0.05), 5, v_per_m=UV), 0, 2)
    for x, y in ring[:-1:2]:
        k.add("timber", beam((x, y, zg), (x, y, zg + 1.1), 0.08), 0, 2)
        k.add("timber", beam((x, y, zg), (x * 0.82, y * 0.82, zg - 1.3), 0.1), 0, 0)   # struts
    # Door and windows.
    k.add("wood", box((-0.6, -R - 0.05, 0.05), (0.6, -R + 0.3, 2.1), UV), 0, 5)
    for z, a in ((2.6, 1.2), (6.5, 2.4), (6.5, 4.0), (8.6, 0.4)):
        rr = R * (1 - (1 - top) * z / H)
        c = np.array([np.cos(a), np.sin(a)]) * (rr - 0.05)
        k.add("timber", box((-0.35, -0.12, z), (0.35, 0.12, z + 0.9), UV).transformed(mat_trs((c[0], c[1], 0), a + np.pi / 2)), 0, 3)
    # Cap: a boat-shaped timber cap (lathe), a ridge finial and the tail pole.
    rt = R * top
    k.add("roof_slate", lathe([rt + 0.35, rt + 0.3, rt * 0.75, rt * 0.3, 0.05], [H, H + 0.5, H + 2.0, H + 3.0, H + 3.25], 20), 0, 3)
    k.add("roof_slate", lathe([rt + 0.3, rt * 0.4, 0.05], [H, H + 2.5, H + 3.2], 8), 4, 7)
    k.add("timber", beam((0, rt * 0.6, H + 0.6), (0, R + 3.2, 0.6), 0.22), 0, 4)
    # Windshaft and four sails on the -y face, the sails tilted slightly back.
    hub = np.array([0.0, -rt - 0.9, H + 1.6])
    k.add("timber", beam(hub + (0, 1.4, 0), hub, 0.5), 0, 5)
    k.add("timber", box((-0.35, -0.35, -0.35), (0.35, 0.35, 0.35), UV).transformed(mat_trs(hub)), 0, 3)
    ang0 = rng.uniform(0, np.pi / 2)
    span, width = 9.5, 2.0
    for i in range(4):
        a = ang0 + i * np.pi / 2
        d = np.array([np.cos(a), 0.0, np.sin(a)])
        w = np.array([-np.sin(a), 0.0, np.cos(a)])
        tip = hub + d * span
        k.add("timber", beam(hub, tip, 0.22), 0, 6)                       # the stock (whip)
        root = hub + d * 1.8
        # Lattice: bars across the sail and two long rails (lod0-2), a framed canvas panel (lod0-4), a single panel (5-7).
        nb = 12
        for j, t in enumerate(np.linspace(0, 1, nb)):
            p = root + (tip - root) * t
            k.add("timber", beam(p - w * 0.05, p + w * width, 0.06), 0, 0 if j % 2 else 2)
        for off in (width * 0.5, width):
            k.add("timber", beam(root + w * off, tip + w * off, 0.07), 0, 3)
        panel = [root + w * 0.1 + (0, 0.05, 0), tip + w * 0.1 + (0, 0.05, 0), tip + w * width + (0, 0.05, 0),
                 root + w * width + (0, 0.05, 0)]
        k.add("canvas", slab(*panel, 0.03), 0, 4)
        k.add("canvas", quad(*panel), 5, 7)
        k.add("canvas", quad(panel[1], panel[0], panel[3], panel[2]), 5, 7)
    return k.levels()


# ---------------------------------------------------------------------------------------------------------------------
# Village furniture
# ---------------------------------------------------------------------------------------------------------------------

def well(variant, seed):
    rng = _rng("well", variant, seed)
    k = Kit()
    R = 1.1
    # Ring of stones (lod0-1), a ring wall (lod2-6), a plain prism (lod7).
    for c in range(3):
        n = 14
        off = rng.uniform(0, np.pi)
        for i in range(n):
            a = off + 2 * np.pi * i / n
            b = box((-0.24, -0.15, 0), (0.24, 0.15, 0.3), UV)
            k.add("stone_blocks", b.transformed(mat_trs((np.cos(a) * R, np.sin(a) * R, c * 0.3), a + np.pi / 2)), 0, 0)
    k.add("stone_blocks", lathe([R + 0.15, R + 0.15, R - 0.2, R - 0.2], [0, 0.9, 0.9, 0.2], 16), 1, 2)
    k.add("stone_blocks", lathe([R + 0.15, R + 0.15, R - 0.2, R - 0.2], [0, 0.9, 0.9, 0.2], 10), 3, 4)
    k.add("stone_blocks", prism(R + 0.15, 0.9, 8, cap=True, uv_scale=UV), 5, 7)
    k.add("timber", box((-0.95, -0.95, 0.15), (0.95, 0.95, 0.2), UV), 0, 3)   # dark water / boards
    for sx in (-1, 1):
        k.add("timber", beam((sx * (R + 0.05), 0, 0), (sx * (R + 0.05), 0, 2.6), 0.16), 0, 6)
    k.add("timber", beam((-R - 0.3, 0, 1.8), (R + 0.3, 0, 1.8), 0.18), 0, 4)       # winch roller
    k.add("wood", lathe([0.2, 0.24], [1.0, 1.35], 8, cap_top=True), 0, 1)  # bucket
    k.add("timber", beam((0, 0, 1.35), (0, 0, 1.8), 0.02), 0, 1)                    # rope
    L, W, rise = 2.9, 2.4, 1.0
    planes, _ = _roof_planes(L, W, 2.6, rise, 0.1, 0.08)
    planes = [p.transformed(mat_trs((0, 0, 0), np.pi / 2)) for p in planes]
    k.add_all("roof_slate", planes, 0, 7)
    return k.levels()


def fence(variant, seed):
    """A 6 m post-and-rail fence panel along x."""
    rng = _rng("fence", variant, seed)
    k = Kit()
    L = 6.0
    posts = np.linspace(-L / 2, L / 2, 4)
    for i, x in enumerate(posts):
        lean = rng.normal(0, 2.0) if variant == "broken" else rng.normal(0, 0.6)
        p = beam((x, 0, -0.2), (x + np.radians(lean) * 1.3, 0, 1.15), 0.14)
        k.add("timber", p, 0, 6)
    for j, z in enumerate((0.45, 0.85)):
        for i in range(3):
            if variant == "broken" and (i, j) in ((1, 1), (2, 0)):
                continue
            a, b = (posts[i], 0.09, z + rng.normal(0, 0.03)), (posts[i + 1], 0.09, z + rng.normal(0, 0.03))
            k.add("wood", beam(a, b, 0.05, 0.12), 0, 5)
    if variant == "broken":
        k.add("wood", beam((0.2, 0.6, 0.06), (2.2, 0.9, 0.06), 0.05, 0.12), 0, 3)   # a fallen rail
    k.add("wood", box((-L / 2, 0.06, 0.4), (L / 2, 0.12, 0.9), UV), 6, 7)
    k.add("timber", box((-L / 2 - 0.07, -0.07, 0), (-L / 2 + 0.07, 0.07, 1.1), UV), 7, 7)
    return k.levels()


def haystack(variant, seed):
    rng = _rng("haystack", variant, seed)
    k = Kit()
    zs = np.array([0.0, 0.8, 1.9, 2.9, 3.5, 3.8])
    rs = np.array([1.6, 1.85, 1.75, 1.2, 0.5, 0.08])
    for lo, hi, sides, sel in ((0, 0, 24, slice(None)), (1, 1, 16, slice(None)), (2, 3, 12, [0, 1, 3, 5]),
                               (4, 5, 8, [0, 1, 3, 5]), (6, 6, 6, [0, 2, 5]), (7, 7, 4, [0, 2, 5])):
        k.add("thatch", lathe(rs[sel], zs[sel], sides), lo, hi)
    k.add("timber", beam((0, 0, 3.5), (rng.normal(0, 0.05), rng.normal(0, 0.05), 4.6), 0.07), 0, 3)
    return k.levels()


def gravestone(variant, seed):
    rng = _rng("gravestone", variant, seed)
    k = Kit()
    lean = rng.normal(0, 3.5)
    M = mat_trs((0, 0, 0), rng.normal(0, 0.05), 1.0, np.radians(lean), np.radians(rng.normal(0, 2)))
    if variant == "cross":
        ring = np.array([[np.cos(t) * 0.27, 0.0, 0.9 + np.sin(t) * 0.27] for t in np.linspace(0, 2 * np.pi, 17)])
        parts0 = [box((-0.11, -0.08, -0.3), (0.11, 0.08, 1.35), UV), box((-0.45, -0.08, 0.8), (0.45, 0.08, 1.0), UV),
                  tube(ring, np.full(17, 0.04), 6, v_per_m=UV),       # the wheel of a celtic cross
                  box((-0.25, -0.18, -0.3), (0.25, 0.18, 0.12), UV)]
        k.add_all("stone_blocks", [p.transformed(M) for p in parts0], 0, 2)
        k.add_all("stone_blocks", [p.transformed(M) for p in (parts0[0], parts0[1], parts0[3])], 3, 5)
        k.add_all("stone_blocks", [p.transformed(M) for p in (parts0[0], parts0[1])], 6, 7)
    else:
        w, h = (0.6, 1.0) if variant == "slab" else (0.5, 0.75)
        # A slab with a rounded or pointed head.
        n = 9
        head = [(np.cos(t) * w / 2, h + np.sin(t) * (w / 2 if variant == "round" else w * 0.25)) for t in np.linspace(0, np.pi, n)]
        outline = [(w / 2, -0.3)] + head + [(-w / 2, -0.3)]
        pts = np.array(outline)
        front = [np.array([x, -0.07, z]) for x, z in pts]
        back = [np.array([x, 0.07, z]) for x, z in pts]
        faces = []
        c = np.mean(pts, 0)
        for i in range(len(pts)):
            j = (i + 1) % len(pts)
            faces.append(quad(front[j], front[i], back[i], back[j]))
            faces.append(tri(front[i], front[j], np.array([c[0], -0.07, c[1]])))
            faces.append(tri(back[j], back[i], np.array([c[0], 0.07, c[1]])))
        k.add("stone_blocks", merge(faces).transformed(M), 0, 4)
        k.add("stone_blocks", box((-w / 2, -0.07, -0.3), (w / 2, 0.07, h + 0.1), UV).transformed(M), 5, 7)
        k.add("stone_blocks", box((-w / 2 - 0.08, -0.6, -0.1), (w / 2 + 0.08, 0.6, 0.06), UV), 0, 3)   # ledger
    return k.levels()


def banner(variant, seed):
    """A tall pole with a hanging heraldic banner (variant = colour)."""
    rng = _rng("banner", variant, seed)
    k = Kit()
    cl = f"cloth_{variant}"
    H = 7.5
    k.add("timber", lathe([0.09, 0.07], [0, H], 8), 0, 3)
    k.add("timber", lathe([0.09, 0.07], [0, H], 4), 4, 7)
    k.add("timber", beam((0, -0.05, H - 0.4), (1.5, -0.05, H - 0.4), 0.07), 0, 6)
    k.add("timber", lathe([0.0, 0.14, 0.0], [H, H + 0.15, H + 0.45], 6), 0, 2)        # finial
    # The cloth hangs from the crossbar and ripples (more columns and rows at finer levels).
    ph = rng.uniform(0, 6)
    for lo, hi, nu, nv in ((0, 0, 8, 10), (1, 1, 6, 8), (2, 3, 4, 6), (4, 4, 2, 3), (5, 5, 1, 2), (6, 7, 1, 1)):
        u = np.linspace(0.05, 1.45, nu + 1)
        v = np.linspace(H - 0.45, H - 3.8, nv + 1)
        U, V = np.meshgrid(u, v)
        Yw = 0.12 * np.sin(U * 4 + ph) * (H - 0.45 - V) / 3.35 + 0.08 * np.sin(V * 2.1 + ph) * 0
        # The tail of the banner ends in a swallow-tail at the bottom centre.
        Vv = V + np.where(V == v[-1], 0.35 * (1 - np.abs((U - 0.75) / 0.7)), 0.0)
        pts = np.stack([U.ravel(), Yw.ravel() - 0.08, Vv.ravel()], 1)
        st = np.stack([(U.ravel() - 0.05) / 1.4, (Vv.ravel() - (H - 4.15)) / 3.7], 1)
        quads = []
        for j in range(nv):
            for i in range(nu):
                a = j * (nu + 1) + i
                quads += [a, a + nu + 1, a + nu + 2, a + 1]
        from ..meshutil import Mesh
        m = Mesh(pts, np.full(nu * nv, 4), quads, None, st, double_sided=True)
        m.compute_normals()
        k.add(cl, m, lo, hi)
    return k.levels()


def wayside_shrine(variant, seed):
    """A carved post with a little roofed niche, standing by the road."""
    k = Kit()
    k.add("timber", box((-0.12, -0.12, -0.3), (0.12, 0.12, 1.5), UV), 0, 7)
    k.add("timber", box((-0.32, -0.25, 1.5), (0.32, 0.25, 2.15), UV), 0, 5)
    k.add("stone_blocks", box((-0.12, -0.27, 1.6), (0.12, -0.2, 2.0), UV), 0, 2)          # the carved saint
    k.add("stone_blocks", lathe([0.0, 0.07, 0.0], [2.0, 2.06, 2.12], 6).transformed(mat_trs((0, -0.23, 0))), 0, 1)
    planes, _ = _roof_planes(0.8, 0.7, 2.15, 0.35, 0.12, 0.05)
    k.add_all("roof_slate", [p.transformed(mat_trs((0, 0, 0), np.pi / 2)) for p in planes], 0, 7)
    k.add("timber", beam((0, -0.12, 1.0), (0, -0.4, 1.45), 0.06), 0, 3)                  # brackets
    k.add("timber", beam((0, 0.12, 1.0), (0, 0.4, 1.45), 0.06), 0, 3)
    k.add("stone_blocks", box((-0.4, -0.4, -0.2), (0.4, 0.4, 0.15), UV), 0, 4)           # footing stone
    return k.levels()


# ---------------------------------------------------------------------------------------------------------------------
# Castle pieces
# ---------------------------------------------------------------------------------------------------------------------

def _crenels(x0, x1, y0, y1, z, n_per_m=0.55, h=0.9, t=0.6, both=True):
    """Merlons along the edges of a rectangle top."""
    out = []
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        L = np.hypot(bx - ax, by - ay)
        n = max(2, int(L * n_per_m))
        for i in range(0, n, 2):
            t0, t1 = i / n, (i + 1) / n
            px0, py0 = ax + (bx - ax) * t0, ay + (by - ay) * t0
            px1, py1 = ax + (bx - ax) * t1, ay + (by - ay) * t1
            lo = (min(px0, px1) - (t / 2 if ax == bx else 0), min(py0, py1) - (t / 2 if ay == by else 0), z)
            hi = (max(px0, px1) + (t / 2 if ax == bx else 0), max(py0, py1) + (t / 2 if ay == by else 0), z + h)
            out.append(box(lo, hi, UV))
    return out


def keep(variant, seed):
    """A square donjon with corner turrets, crenellations and a slate-capped stair turret."""
    rng = _rng("keep", variant, seed)
    k = Kit()
    S, H = 12.0, 19.0
    h2 = S / 2
    k.add("stone_blocks", box((-h2, -h2, -0.5), (h2, h2, H), UV), 0, 7)
    k.add("stone_blocks", box((-h2 - 0.6, -h2 - 0.6, -0.5), (h2 + 0.6, h2 + 0.6, 1.8), UV), 0, 4)        # battered base
    k.add("stone_blocks", box((-h2 - 0.3, -h2 - 0.3, H), (h2 + 0.3, h2 + 0.3, H + 0.5), UV), 0, 5)       # parapet walk
    k.add_all("stone_blocks", _crenels(-h2 - 0.3, h2 + 0.3, -h2 - 0.3, h2 + 0.3, H + 0.5), 0, 1)
    k.add("stone_blocks", box((-h2 - 0.3, -h2 - 0.3, H + 0.5), (h2 + 0.3, h2 + 0.3, H + 1.1), UV), 2, 6)
    # Corner turrets with conical slate caps.
    for sx in (-1, 1):
        for sy in (-1, 1):
            c = (sx * h2, sy * h2, 0)
            for lo, hi, sides in ((0, 0, 20), (1, 2, 12), (3, 5, 8), (6, 7, 6)):
                k.add("stone_blocks", prism(1.7, H + 3.0, sides, z0=0, cap=True, uv_scale=UV).transformed(mat_trs(c)), lo, hi)
                k.add("roof_slate", lathe([2.05, 0.05], [H + 3.0, H + 6.5], sides).transformed(mat_trs(c)), lo, hi)
            k.add("timber", beam((c[0], c[1], H + 6.4), (c[0], c[1], H + 7.4), 0.05), 0, 2)
    # Pilaster buttresses and arrow slits.
    for face in range(4):
        rz = face * np.pi / 2
        for x in (-2.5, 2.5):
            k.add("stone_blocks", box((x - 0.5, -h2 - 0.45, 0), (x + 0.5, -h2, H - 1), UV).transformed(mat_trs((0, 0, 0), rz)), 0, 3)
        for z in (5.0, 9.5, 14.0):
            for x in (-4.3, 0.0, 4.3):
                if face == 0 and z == 5.0 and x == 0.0:
                    continue
                w = 0.25 if z < 14 else 0.7
                k.add("timber", box((x - w / 2, -h2 - 0.05, z), (x + w / 2, -h2 + 0.1, z + 1.6), UV)
                      .transformed(mat_trs((0, 0, 0), rz)), 0, 2 if w < 0.5 else 4)
    # Door at first-floor height with a timber stair.
    k.add("wood", box((-0.9, -h2 - 0.1, 4.0), (0.9, -h2 + 0.1, 6.8), UV), 0, 6)
    for i in range(12):
        z = 4.0 * i / 12
        k.add("timber", box((-0.9, -h2 - 1.2 - (12 - i) * 0.45, z), (0.9, -h2 - 0.75 - (12 - i) * 0.45, z + 0.33), UV), 0, 0)
    k.add("timber", beam((0, -h2 - 6.6, 0), (0, -h2 - 0.4, 4.0), 1.8, 0.4), 1, 5)
    _ = rng
    return k.levels()


def gatehouse(variant, seed):
    """Twin round towers flanking an arched gate passage with a portcullis; along x, the gate opening along y."""
    k = Kit()
    W, D, H = 7.0, 8.0, 10.5
    Rt, Ht = 3.2, 13.0
    # The gate block around the passage: two side piers and a lintel mass above the arch.
    pw, ph = 3.6, 5.6
    for sx in (-1, 1):
        k.add("stone_blocks", box((sx * pw / 2, -D / 2, 0) if sx > 0 else (-W / 2, -D / 2, 0),
                                  (W / 2, D / 2, H) if sx > 0 else (-pw / 2, D / 2, H), UV), 0, 7)
    k.add("stone_blocks", box((-pw / 2, -D / 2, ph), (pw / 2, D / 2, H), UV), 0, 7)
    # The arch: voussoirs (lod0-2) or a half-disc infill (lod3-5).
    n = 11
    for i in range(n):
        a0, a1 = np.pi * i / n, np.pi * (i + 1) / n
        r0, r1 = pw / 2, pw / 2 + 0.6
        for yf in (-D / 2 - 0.05, D / 2 + 0.05):
            p = [(np.cos(a0) * r0, yf, ph - pw / 2 + np.sin(a0) * r0 * 0.9), (np.cos(a1) * r0, yf, ph - pw / 2 + np.sin(a1) * r0 * 0.9),
                 (np.cos(a1) * r1, yf, ph - pw / 2 + np.sin(a1) * r1 * 0.9), (np.cos(a0) * r1, yf, ph - pw / 2 + np.sin(a0) * r1 * 0.9)]
            if yf < 0:
                p = p[::-1]
            k.add("stone_blocks", quad(*p), 0, 0)
        # Arch fill above the round head (the passage below stays open).
        for yf in (-D / 2, D / 2):
            p = [(np.cos(a0) * r0, yf, ph - pw / 2 + np.sin(a0) * r0 * 0.9), (np.cos(a1) * r0, yf, ph - pw / 2 + np.sin(a1) * r0 * 0.9),
                 (np.cos(a1) * r0, yf, ph), (np.cos(a0) * r0, yf, ph)]
            k.add("stone_blocks", quad(*(p[::-1] if yf < 0 else p)), 0, 4)
    # The portcullis, half raised: a grid of iron-shod timbers.
    for x in np.linspace(-pw / 2 + 0.3, pw / 2 - 0.3, 7):
        k.add("timber", box((x - 0.07, -D / 2 + 0.8, 2.2), (x + 0.07, -D / 2 + 0.95, ph), UV), 0, 1)
    for z in np.linspace(2.4, ph - 0.3, 5):
        k.add("timber", box((-pw / 2, -D / 2 + 0.78, z), (pw / 2, -D / 2 + 0.97, z + 0.12), UV), 0, 1)
    k.add("timber", box((-pw / 2, -D / 2 + 0.8, 2.2), (pw / 2, -D / 2 + 0.95, ph), UV), 2, 4)
    # Twin towers.
    for sx in (-1, 1):
        c = (sx * (W / 2 + Rt * 0.55), -D / 2 + Rt * 0.9, 0)
        for lo, hi, sides in ((0, 0, 24), (1, 2, 16), (3, 5, 10), (6, 7, 6)):
            k.add("stone_blocks", prism(Rt, Ht, sides, cap=True, uv_scale=UV).transformed(mat_trs(c)), lo, hi)
        k.add("stone_blocks", prism(Rt + 0.5, 1.5, 20, cap=False, uv_scale=UV).transformed(mat_trs(c)), 0, 3)
        for i in range(10):
            a = 2 * np.pi * i / 10
            k.add("stone_blocks", box((-0.5, -0.35, Ht), (0.5, 0.35, Ht + 1.0), UV)
                  .transformed(mat_trs((c[0] + np.cos(a) * (Rt - 0.35), c[1] + np.sin(a) * (Rt - 0.35), 0), a + np.pi / 2)), 0, 3)
        k.add("stone_blocks", prism(Rt, 0.8, 10, z0=Ht, cap=True, uv_scale=UV).transformed(mat_trs(c)), 4, 6)
        for z in (4.0, 8.0):
            for a in (-np.pi / 2 - 0.5, -np.pi / 2 + 0.5, np.pi * sx * 0 + (0.0 if sx > 0 else np.pi)):
                k.add("timber", box((-0.13, -0.1, z), (0.13, 0.1, z + 1.4), UV)
                      .transformed(mat_trs((c[0] + np.cos(a) * (Rt + 0.02), c[1] + np.sin(a) * (Rt + 0.02), 0), a + np.pi / 2)), 0, 3)
    k.add_all("stone_blocks", _crenels(-W / 2, W / 2, -D / 2, D / 2, H), 0, 3)
    k.add("stone_blocks", box((-W / 2, -D / 2, H), (W / 2, D / 2, H + 0.6), UV), 4, 6)
    return k.levels()
