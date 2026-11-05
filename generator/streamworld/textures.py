"""Procedural textures: foliage and ground-cover atlases, tileable surfaces, terrain sources, sky, region bakes.

Everything is rasterised with numpy (analytic shapes evaluated per pixel inside each shape's bounding box) and
written with OpenImageIO. Generation is deterministic from the configuration seed. Layouts follow atlas.py.
"""

import os
import time
import zlib

import numpy as np

from . import atlas
from . import noise as nz
from .imgutil import linear_to_srgb, read_image, resize_bilinear, srgb_to_linear, write_image


def _log(msg):
    print(f"[textures] {msg}", flush=True)


# =====================================================================================================================
# Tileable noise / cellular helpers (rectangular images, rows top to bottom)
# =====================================================================================================================

def tnoise(h, w, seed, px, py, octaves=4, gain=0.5):
    """Tileable gradient fBm of shape (h, w), roughly [-1, 1]; px, py integer lattice periods at octave 0."""
    total = np.zeros((h, w))
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        X = np.arange(w)[None, :] * px / w + np.zeros((h, 1))
        Y = np.arange(h)[:, None] * py / h + np.zeros((1, w))
        xi = np.floor(X).astype(np.int64)
        yi = np.floor(Y).astype(np.int64)
        xf, yf = X - xi, Y - yi

        def g(ix, iy, dx, dy):
            a = nz.hash_float(ix % px, iy % py, seed + o * 131) * 2 * np.pi
            return np.cos(a) * dx + np.sin(a) * dy
        u, v = nz._fade(xf), nz._fade(yf)
        n00, n10 = g(xi, yi, xf, yf), g(xi + 1, yi, xf - 1, yf)
        n01, n11 = g(xi, yi + 1, xf, yf - 1), g(xi + 1, yi + 1, xf - 1, yf - 1)
        total += amp * 1.414 * ((n00 + u * (n10 - n00)) * (1 - v) + (n01 + u * (n11 - n01)) * v)
        norm += amp
        amp *= gain
        px *= 2
        py *= 2
    return total / norm


def voronoi(h, w, nx, ny, seed, jitter=0.85, metric=(1.0, 1.0), warp=0.0):
    """Tileable cellular noise: returns F1, F2 (in cell units) and the cell id of each pixel."""
    gx = (np.arange(w)[None, :] + 0.5) / w * nx + np.zeros((h, 1))
    gy = (np.arange(h)[:, None] + 0.5) / h * ny + np.zeros((1, w))
    if warp:
        gx = gx + warp * tnoise(h, w, seed + 77, 3, 3, 3)
        gy = gy + warp * tnoise(h, w, seed + 78, 3, 3, 3)
    cx = np.floor(gx).astype(np.int64)
    cy = np.floor(gy).astype(np.int64)
    f1 = np.full((h, w), 1e9)
    f2 = np.full((h, w), 1e9)
    cid = np.zeros((h, w), dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ncx, ncy = cx + dx, cy + dy
            wx, wy = ncx % nx, ncy % ny
            ppx = ncx + 0.5 + (nz.hash_float(wx, wy, seed) - 0.5) * jitter
            ppy = ncy + 0.5 + (nz.hash_float(wx, wy, seed + 1) - 0.5) * jitter
            d = np.hypot((gx - ppx) * metric[0], (gy - ppy) * metric[1])
            closer = d < f1
            f2 = np.where(closer, f1, np.minimum(f2, d))
            cid = np.where(closer, wy * nx + wx, cid)
            f1 = np.where(closer, d, f1)
    return f1, f2, cid


def cell_rand(cid, seed):
    return nz.hash_float(cid, seed)


def normal_from_height(hgt, strength):
    """Tileable tangent-space normal map (0..1 encoded, OpenGL +v up) from a height image (wrapping gradients)."""
    gx = (np.roll(hgt, -1, 1) - np.roll(hgt, 1, 1)) * 0.5 * strength
    grow = (np.roll(hgt, -1, 0) - np.roll(hgt, 1, 0)) * 0.5 * strength
    n = np.stack([-gx, grow, np.ones_like(hgt)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def blur_wrap(a, r):
    """Box blur with wrap-around (separable), radius r pixels."""
    if r <= 0:
        return a
    out = a.copy()
    for axis in (0, 1):
        acc = np.zeros_like(out)
        for k in range(-r, r + 1):
            acc += np.roll(out, k, axis)
        out = acc / (2 * r + 1)
    return out


def lerp(a, b, t):
    t = np.asarray(t)
    if t.ndim == 2 and np.ndim(a) >= 1 and np.shape(a)[-1:] == (3,):
        t = t[..., None]
    return np.asarray(a) * (1 - t) + np.asarray(b) * t


def col(hexstr):
    hexstr = hexstr.lstrip("#")
    return np.array([int(hexstr[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def ss(e0, e1, x):
    return nz.smoothstep(e0, e1, x)


# =====================================================================================================================
# Sprite canvas (foliage, grass)
# =====================================================================================================================

class Canvas:
    """RGB + alpha canvas, pixel coordinates (x right, y down)."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3))
        self.a = np.zeros((h, w))

    def _box(self, x0, y0, x1, y1):
        i0, i1 = max(int(np.floor(x0)), 0), min(int(np.ceil(x1)) + 1, self.w)
        j0, j1 = max(int(np.floor(y0)), 0), min(int(np.ceil(y1)) + 1, self.h)
        if i1 <= i0 or j1 <= j0:
            return None
        X, Y = np.meshgrid(np.arange(i0, i1) + 0.5, np.arange(j0, j1) + 0.5)
        return (slice(j0, j1), slice(i0, i1)), X, Y

    def paint(self, sl, mask, rgb):
        if rgb.ndim == 1:
            rgb = np.broadcast_to(rgb, mask.shape + (3,))
        self.rgb[sl][mask] = rgb[mask]
        self.a[sl][mask] = 1.0

    def stroke(self, pts, widths, color_fn):
        """Polyline with per-point half-width; color_fn(t_along, across[-1..1]) -> (n,3)."""
        pts = np.asarray(pts, float)
        widths = np.asarray(widths, float)
        wmax = widths.max() + 1
        b = self._box(pts[:, 0].min() - wmax, pts[:, 1].min() - wmax, pts[:, 0].max() + wmax, pts[:, 1].max() + wmax)
        if b is None:
            return
        sl, X, Y = b
        nseg = len(pts) - 1
        best = np.full(X.shape, np.inf)
        tt = np.zeros(X.shape)
        across = np.zeros(X.shape)
        for i in range(nseg):
            a, c = pts[i], pts[i + 1]
            d = c - a
            L2 = max(d @ d, 1e-9)
            t = np.clip(((X - a[0]) * d[0] + (Y - a[1]) * d[1]) / L2, 0, 1)
            qx, qy = a[0] + t * d[0], a[1] + t * d[1]
            dist = np.hypot(X - qx, Y - qy)
            wi = widths[i] + (widths[i + 1] - widths[i]) * t
            ratio = dist / np.maximum(wi, 1e-6)
            m = ratio < best
            best = np.where(m, ratio, best)
            tt = np.where(m, (i + t) / nseg, tt)
            side = np.sign((X - qx) * -d[1] + (Y - qy) * d[0])
            across = np.where(m, ratio * side, across)
        mask = best < 1.0
        if mask.any():
            self.paint(sl, mask, color_fn(tt, across))

    def leaf(self, base, angle, length, width, shape, color_fn):
        """Analytic leaf: base point, direction angle (radians, 0 = up), shape(t) -> half-width factor."""
        dx, dy = np.sin(angle), -np.cos(angle)
        tip = np.array(base) + np.array([dx, dy]) * length
        r = max(length, width) * 1.1
        b = self._box(min(base[0], tip[0]) - r * 0.6, min(base[1], tip[1]) - r * 0.6,
                      max(base[0], tip[0]) + r * 0.6, max(base[1], tip[1]) + r * 0.6)
        if b is None:
            return
        sl, X, Y = b
        lx = (X - base[0]) * dx + (Y - base[1]) * dy
        ly = (X - base[0]) * -dy + (Y - base[1]) * dx
        t = lx / length
        hw = shape(np.clip(t, 0, 1)) * width
        inside = (t >= 0) & (t <= 1) & (np.abs(ly) < hw)
        if inside.any():
            self.paint(sl, inside, color_fn(t, ly / np.maximum(hw, 1e-6)))

    def disk(self, c, rx, ry, angle, color_fn):
        r = max(rx, ry) + 1
        b = self._box(c[0] - r, c[1] - r, c[0] + r, c[1] + r)
        if b is None:
            return
        sl, X, Y = b
        ca, sa = np.cos(angle), np.sin(angle)
        u = ((X - c[0]) * ca + (Y - c[1]) * sa) / rx
        v = (-(X - c[0]) * sa + (Y - c[1]) * ca) / ry
        rr = u * u + v * v
        m = rr < 1
        if m.any():
            self.paint(sl, m, color_fn(np.sqrt(rr), u))


def bleed(rgb, a, levels=10):
    """Push-pull: fill transparent texels with nearby opaque colour so mip-mapping does not darken edges."""
    pyr = [(rgb * a[..., None], a.copy())]
    c, w = pyr[0]
    for _ in range(levels):
        h2, w2 = c.shape[0] // 2, c.shape[1] // 2
        if h2 < 1 or w2 < 1:
            break
        c = c[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2, 3).sum((1, 3))
        w = w[:h2 * 2, :w2 * 2].reshape(h2, 2, w2, 2).sum((1, 3))
        pyr.append((c, w))
    fill = pyr[-1][0] / np.maximum(pyr[-1][1][..., None], 1e-9)
    for c, w in reversed(pyr[:-1]):
        up = np.repeat(np.repeat(fill, 2, 0), 2, 1)
        up = np.pad(up, ((0, c.shape[0] - up.shape[0]), (0, c.shape[1] - up.shape[1]), (0, 0)), mode="edge")
        own = c / np.maximum(w[..., None], 1e-9)
        wt = np.clip(w, 0, 1)[..., None]
        fill = own * wt + up * (1 - wt)
    out = rgb.copy()
    m = a < 0.5
    out[m] = fill[m]
    return out


def rgba(canvas):
    rgb = bleed(canvas.rgb, canvas.a)
    return np.concatenate([rgb, canvas.a[..., None]], -1)


# =====================================================================================================================
# Foliage atlases
# =====================================================================================================================

def _leaf_color(rng, base, var=0.12, light=0.18):
    c = np.clip(base * (1 + rng.uniform(-var, var, 3)) * rng.uniform(0.82, 1.12), 0, 1)

    def fn(t, across):
        shade = 0.84 + light * across + 0.10 * t
        rib = np.where(np.abs(across) < 0.07, 0.82, 1.0)
        vein = 1.0 - 0.06 * (np.abs(np.sin(t * 22 + np.abs(across) * 6)) > 0.93)
        out = c[None, :] * (shade * rib * vein)[:, None] if np.ndim(t) == 1 else c * (shade * rib * vein)[..., None]
        return np.clip(out, 0, 1)
    return fn


def _flat(c):
    def fn(t, across):
        return np.broadcast_to(c, np.shape(t) + (3,)) * (0.9 + 0.12 * (1 - np.abs(across)))[..., None]
    return fn


def _grad(c0, c1):
    c0, c1 = np.asarray(c0), np.asarray(c1)

    def fn(t, across):
        return (c0 * (1 - t[..., None]) + c1 * t[..., None]) * (0.88 + 0.14 * (1 - np.abs(across)))[..., None]
    return fn


def _branch_path(rng, base, angle, length, bend, n=8):
    pts = [np.array(base, float)]
    a = angle
    step = length / n
    for _ in range(n):
        a += bend / n + rng.normal(0, 0.04)
        pts.append(pts[-1] + np.array([np.sin(a), -np.cos(a)]) * step)
    return np.array(pts)


def _twig(cv, rng, kind):
    S = cv.w
    stem_col = {"oak": col("5a4632"), "birch": col("6b5040"), "bush": col("4b3a2a"),
                "pine": col("6a4a32"), "spruce": col("5c4430")}[kind]
    leaf_base = {"oak": col("4f6e2a"), "birch": col("6e8f34"), "bush": col("3f5f2c")}.get(kind)
    main = _branch_path(rng, (S * 0.5 + rng.normal(0, 6), S - 2), rng.normal(0, 0.15), S * 0.86, rng.normal(0, 0.5))
    shapes = {
        "oak": lambda t: np.sin(np.pi * t) ** 0.7 * (0.62 + 0.38 * np.abs(np.sin(t * np.pi * 4.5))) * (t > 0.08),
        "birch": lambda t: (np.sin(np.pi * t ** 0.85) ** 0.6) * (1 - 0.25 * t) * (0.92 + 0.08 * np.abs(np.sin(t * 70)))
        * (t > 0.1),
        "bush": lambda t: np.sin(np.pi * t) ** 0.55 * (t > 0.06),
    }
    if kind in ("oak", "birch", "bush"):
        L = {"oak": S * 0.2, "birch": S * 0.12, "bush": S * 0.1}[kind]
        Wd = {"oak": L * 0.38, "birch": L * 0.4, "bush": L * 0.44}[kind]
        nside = {"oak": 9, "birch": 12, "bush": 14}[kind]
        branches = [main]
        for i in range(nside):
            k = int(len(main) * (0.18 + 0.75 * i / nside))
            k = min(k, len(main) - 2)
            side = 1 if i % 2 else -1
            ang = np.arctan2(main[k + 1][0] - main[k][0], -(main[k + 1][1] - main[k][1]))
            br = _branch_path(rng, main[k], ang + side * rng.uniform(0.7, 1.25), S * rng.uniform(0.24, 0.4),
                              -side * 0.5, n=6)
            branches.append(br)
        leaves = []
        for br in branches:
            for j in range(1, len(br)):
                for _ in range(3 if kind != "oak" else 2):
                    side = rng.choice([-1, 1])
                    d = br[j] - br[j - 1]
                    ang = np.arctan2(d[0], -d[1]) + side * rng.uniform(0.4, 1.2)
                    leaves.append((br[j], ang, L * rng.uniform(0.75, 1.15), Wd * rng.uniform(0.85, 1.15)))
            d = br[-1] - br[-2]
            leaves.append((br[-1], np.arctan2(d[0], -d[1]) + rng.normal(0, 0.2), L, Wd))
        for br in branches:
            wd = np.linspace(S * 0.012, S * 0.004, len(br)) if br is main else np.linspace(S * 0.006, S * 0.003, len(br))
            cv.stroke(br, wd, _flat(stem_col))
        order = rng.permutation(len(leaves))
        for i in order:
            p, a, l, w = leaves[i]
            cv.leaf(p, a, l, w, shapes[kind], _leaf_color(rng, leaf_base))
    elif kind == "pine":
        cv.stroke(main, np.linspace(S * 0.014, S * 0.006, len(main)), _flat(stem_col))
        nodes = [main[-1]] + [main[int(len(main) * f)] for f in (0.35, 0.6, 0.8)]
        needle_c0, needle_c1 = col("2f4a26"), col("6b8a3c")
        for p in nodes:
            n = 60
            for _ in range(n):
                a = rng.normal(0, 0.9)
                L = S * rng.uniform(0.18, 0.28)
                tip = p + np.array([np.sin(a), -np.cos(a)]) * L
                mid = (p + tip) / 2 + rng.normal(0, 3, 2)
                c0 = needle_c0 * rng.uniform(0.85, 1.15)
                cv.stroke([p, mid, tip], [S * 0.0045, S * 0.004, S * 0.0015], _grad(c0, needle_c1 * rng.uniform(0.85, 1.1)))
    elif kind == "spruce":
        branches = [main]
        for i in range(7):
            k = min(int(len(main) * (0.15 + 0.8 * i / 7)), len(main) - 2)
            side = 1 if i % 2 else -1
            br = _branch_path(rng, main[k], side * rng.uniform(0.7, 1.1), S * rng.uniform(0.2, 0.3), -side * 0.3, n=6)
            branches.append(br)
        for br in branches:
            cv.stroke(br, np.linspace(S * 0.009, S * 0.004, len(br)), _flat(stem_col))
        nc0, nc1 = col("223b24"), col("4a6a3a")
        for br in branches:
            for j in range(len(br) - 1):
                for t in np.linspace(0, 1, 6, endpoint=False):
                    p = br[j] + (br[j + 1] - br[j]) * t
                    d = br[j + 1] - br[j]
                    base_a = np.arctan2(d[0], -d[1])
                    for side in (-1, 1):
                        a = base_a + side * rng.uniform(0.5, 1.2)
                        L = S * rng.uniform(0.035, 0.055)
                        tip = p + np.array([np.sin(a), -np.cos(a)]) * L
                        cv.stroke([p, tip], [S * 0.004, S * 0.0018], _grad(nc0 * rng.uniform(0.85, 1.15), nc1))
            d = br[-1] - br[-2]
            a0 = np.arctan2(d[0], -d[1])
            for a in np.linspace(-0.6, 0.6, 5):
                tip = br[-1] + np.array([np.sin(a0 + a), -np.cos(a0 + a)]) * S * 0.045
                cv.stroke([br[-1], tip], [S * 0.004, S * 0.0018], _grad(nc0, nc1 * 1.1))


def foliage_atlas(name, seed):
    kind = {"leaves_oak": "oak", "leaves_birch": "birch", "needles_pine": "pine", "needles_spruce": "spruce",
            "leaves_bush": "bush"}[name]
    full = np.zeros((1024, 1024, 4))
    for i in range(4):
        rng = np.random.default_rng(seed * 100 + i)
        cv = Canvas(512, 512)
        _twig(cv, rng, kind)
        img = rgba(cv)
        r0 = (1 - i // 2) * 512
        c0 = (i % 2) * 512
        full[r0:r0 + 512, c0:c0 + 512] = img
    return full


# =====================================================================================================================
# Ground cover
# =====================================================================================================================

def _tuft(cv, rng, n, h_range, lean, c_base, c_tip, width=(0.012, 0.025), curl=0.5, spread=0.35):
    W, H = cv.w, cv.h
    for _ in range(n):
        bx = W * (0.5 + rng.normal(0, spread * 0.5))
        bx = np.clip(bx, W * 0.08, W * 0.92)
        L = H * rng.uniform(*h_range)
        a = rng.normal(0, lean) + (bx / W - 0.5) * 0.9
        bend = rng.normal(0, curl)
        pts = []
        p = np.array([bx, H - 1.0])
        for k in range(7):
            pts.append(p.copy())
            aa = a + bend * (k / 6) ** 2
            p = p + np.array([np.sin(aa), -np.cos(aa)]) * L / 6
        pts.append(p)
        w0 = W * rng.uniform(*width)
        wd = np.linspace(w0, w0 * 0.12, len(pts))
        c0 = np.clip(c_base * rng.uniform(0.8, 1.15), 0, 1)
        c1 = np.clip(c_tip * rng.uniform(0.85, 1.15), 0, 1)
        cv.stroke(pts, wd, _grad(c0, c1))


def _flowers(cv, rng, petal, center, n=6):
    W, H = cv.w, cv.h
    _tuft(cv, rng, 10, (0.25, 0.5), 0.25, col("3d5a24"), col("6f8f3a"), width=(0.012, 0.022))
    for _ in range(n):
        bx = W * rng.uniform(0.2, 0.8)
        L = H * rng.uniform(0.55, 0.9)
        a = rng.normal(0, 0.15)
        tip = np.array([bx + np.sin(a) * L, H - 1 - np.cos(a) * L])
        mid = (np.array([bx, H - 1]) + tip) / 2 + rng.normal(0, 3, 2)
        cv.stroke([[bx, H - 1], mid, tip], [W * 0.012, W * 0.01, W * 0.008], _flat(col("4a6b2c")))
        r = W * rng.uniform(0.06, 0.09)
        npet = rng.integers(5, 8)
        pc = np.clip(petal * rng.uniform(0.88, 1.08), 0, 1)
        for k in range(npet):
            ang = k / npet * 2 * np.pi + rng.uniform(0, 0.3)
            c = tip + np.array([np.cos(ang), np.sin(ang) * 0.7]) * r * 0.6
            cv.disk(c, r * 0.55, r * 0.28, ang, lambda rr, u, pc=pc: pc * (0.8 + 0.25 * (1 - rr))[..., None])
        cv.disk(tip, r * 0.32, r * 0.28, 0, lambda rr, u: center * (0.8 + 0.3 * (1 - rr))[..., None])


def grass_atlas(seed):
    full = np.zeros((1024, 1024, 4))
    for name, (u0, v0, u1, v1) in atlas.GRASS_RECTS.items():
        rng = np.random.default_rng(seed + zlib.crc32(name.encode()) % 1000)
        w, h = int(round((u1 - u0) * 1024)), int(round((v1 - v0) * 1024))
        cv = Canvas(w, h)
        if name == "grass_a":
            _tuft(cv, rng, 140, (0.45, 0.95), 0.18, col("3b5a22"), col("86a24a"))
        elif name == "grass_b":
            _tuft(cv, rng, 70, (0.6, 0.98), 0.22, col("4a6328"), col("a5a85a"), width=(0.01, 0.018), curl=0.7)
            for _ in range(10):
                bx = w * rng.uniform(0.25, 0.75)
                L = h * rng.uniform(0.75, 0.97)
                a = rng.normal(0, 0.12)
                tip = np.array([bx + np.sin(a) * L, h - 1 - np.cos(a) * L])
                cv.stroke([[bx, h - 1], tip], [w * 0.006, w * 0.004], _flat(col("8c8a50")))
                cv.disk(tip + [0, h * 0.05], w * 0.012, h * 0.06, 0.0, lambda rr, u: col("b8a96a") * (0.85 + 0.2 * (1 - rr))[..., None])
        elif name == "grass_dry":
            _tuft(cv, rng, 120, (0.4, 0.95), 0.3, col("7a6f3c"), col("c7b277"), width=(0.006, 0.012), spread=0.5)
        elif name == "clover":
            _tuft(cv, rng, 12, (0.2, 0.45), 0.4, col("3c5a24"), col("5f7f34"))
            for _ in range(14):
                c = np.array([w * rng.uniform(0.15, 0.85), h * rng.uniform(0.25, 0.85)])
                cv.stroke([[c[0], h - 1], c], [w * 0.01, w * 0.008], _flat(col("4d6a2e")))
                for k in range(3):
                    ang = k * 2.09 + rng.uniform(0, 0.4)
                    cc = c + np.array([np.cos(ang), np.sin(ang)]) * w * 0.035
                    lc = col("446b2c") * rng.uniform(0.85, 1.15)
                    cv.disk(cc, w * 0.04, w * 0.034, ang, lambda rr, u, lc=lc: lc * (0.85 + 0.2 * u)[..., None])
        elif name == "reed":
            _tuft(cv, rng, 30, (0.75, 0.99), 0.08, col("4f6230"), col("8a9658"), width=(0.012, 0.02), curl=0.2,
                  spread=0.3)
            for _ in range(4):
                bx = w * rng.uniform(0.35, 0.65)
                top = h * rng.uniform(0.03, 0.12)
                cv.stroke([[bx, h - 1], [bx + rng.normal(0, 4), top]], [w * 0.01, w * 0.008], _flat(col("6b7040")))
                cv.disk([bx, top + h * 0.08], w * 0.03, h * 0.07, 0, lambda rr, u: col("5a3c22") * (0.8 + 0.3 * (1 - rr))[..., None])
        else:
            petal = {"flower_yellow": col("e8c43a"), "flower_white": col("eeeae0"), "flower_purple": col("8a5aa8"),
                     "flower_red": col("b8382c")}[name]
            centre = col("d9a23a") if name != "flower_yellow" else col("b0702a")
            _flowers(cv, rng, petal, centre)
        img = rgba(cv)
        r0 = int(round((1 - v1) * 1024))
        c0 = int(round(u0 * 1024))
        full[r0:r0 + h, c0:c0 + w] = img
    return full


def fern(seed):
    rng = np.random.default_rng(seed)
    W, H = 512, 1024
    cv = Canvas(W, H)
    rach = _branch_path(rng, (W * 0.5, H - 1), 0.05, H * 0.95, 0.35, n=24)
    c_dark, c_light = col("2f5122"), col("6f9a3e")
    n = len(rach)
    for i in range(2, n - 1):
        t = i / (n - 1)
        L = W * 0.42 * np.sin(np.pi * (0.15 + 0.85 * t)) ** 0.8 * (1 - 0.15 * t) + 6
        d = rach[i + 1] - rach[i]
        base_a = np.arctan2(d[0], -d[1])
        for side in (-1, 1):
            a = base_a + side * (1.25 - 0.4 * t)
            # Pinna: a lobed leaflet.
            shape = lambda tt: (np.sin(np.pi * tt) ** 0.6) * (0.7 + 0.3 * np.abs(np.sin(tt * np.pi * 7))) * (tt > 0.03)
            cv.leaf(rach[i], a, L, L * 0.14 + 3, shape, _leaf_color(rng, lerp(c_dark, c_light, 0.4 + 0.4 * t), 0.08, 0.25))
    cv.stroke(rach, np.linspace(5, 1.5, n), _flat(col("4a5a28")))
    return rgba(cv)


# =====================================================================================================================
# Tileable surfaces
# =====================================================================================================================

def bark(kind, seed):
    h, w = 1024, 512
    n1 = tnoise(h, w, seed, 4, 8, 5)
    if kind == "oak":
        f1, f2, cid = voronoi(h, w, 7, 5, seed + 1, metric=(1.0, 0.35))
        ridge = np.clip((f2 - f1) * 2.2, 0, 1)
        hgt = ridge ** 0.6 + 0.25 * n1 + 0.12 * tnoise(h, w, seed + 3, 16, 32, 3)
        base = lerp(col("3a2e24"), col("6e6152"), np.clip(hgt * 0.9, 0, 1))
        base = lerp(base, col("55603a"), np.clip(tnoise(h, w, seed + 5, 3, 3, 3) * 1.5 - 0.3, 0, 0.5))  # moss tint
        strength = 9.0
    elif kind == "pine":
        f1, f2, cid = voronoi(h, w, 6, 7, seed + 1, metric=(1.0, 0.45), warp=0.6)
        plate = np.clip((f2 - f1) * 3.0, 0, 1)
        r = cell_rand(cid, seed)
        flake = tnoise(h, w, seed + 6, 12, 24, 3)
        hgt = plate ** 0.4 * (0.7 + 0.3 * r) + 0.12 * n1 + 0.1 * flake
        top = lerp(col("6a4634"), col("8c6248"), r[..., None]) * (0.85 + 0.2 * flake[..., None])
        top = lerp(top, col("6e6660"), np.clip(r - 0.6, 0, 1)[..., None])
        base = lerp(col("2a1e18"), top, ss(0.04, 0.3, plate))
        strength = 7.0
    elif kind == "birch":
        hgt = 0.15 * n1
        base = np.broadcast_to(col("e4e0d6"), (h, w, 3)).copy()
        base = lerp(base, col("c9c3b5"), np.clip(tnoise(h, w, seed + 2, 4, 6, 4) * 0.8 + 0.3, 0, 1))
        rng = np.random.default_rng(seed)
        mark = np.zeros((h, w))
        Y, X = np.mgrid[0:h, 0:w]
        for _ in range(160):
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            lx, ly = rng.uniform(15, 70), rng.uniform(1.5, 4)
            dx = (X - cx + w / 2) % w - w / 2
            dy = (Y - cy + h / 2) % h - h / 2
            mark = np.maximum(mark, (np.abs(dx) / lx) ** 2 + (dy / ly) ** 2 < 1)
        for _ in range(10):  # dark patches
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
            dx = (X - cx + w / 2) % w - w / 2
            dy = (Y - cy + h / 2) % h - h / 2
            m = np.exp(-((dx / 40) ** 2 + (dy / 26) ** 2)) * (tnoise(h, w, seed + 9, 16, 32, 2) > -0.1)
            mark = np.maximum(mark, m * 0.85)
        base = lerp(base, col("2a2622"), mark)
        hgt = hgt - 0.6 * mark
        strength = 5.0
    else:  # dead
        f1, f2, cid = voronoi(h, w, 10, 3, seed + 1, metric=(1.0, 0.15))
        crack = 1 - ss(0.0, 0.12, f2 - f1)
        grain = tnoise(h, w, seed + 4, 24, 2, 3)
        hgt = 0.4 * grain - 0.8 * crack + 0.15 * n1
        base = lerp(col("8a8378"), col("b1aa9c"), np.clip(grain * 0.8 + 0.5, 0, 1))
        base = lerp(base, col("3a3631"), crack * 0.9)
        strength = 6.0
    color = np.clip(base * (0.92 + 0.12 * n1[..., None]), 0, 1)
    return color, normal_from_height(hgt, strength)


def rock(kind, seed):
    s = 1024
    n = tnoise(s, s, seed, 4, 4, 6)
    f1, f2, cid = voronoi(s, s, 7, 7, seed + 1, warp=0.8)
    facet = np.clip(f2 - f1, 0, 1)
    r = cell_rand(cid, seed + 2)
    if kind == "rock":
        hgt = 0.5 * n + 0.35 * facet ** 0.5 + 0.08 * tnoise(s, s, seed + 3, 32, 32, 2)
        base = lerp(col("6c6a64"), col("9c968a"), np.clip(0.5 + 0.6 * n + 0.2 * (r - 0.5), 0, 1))
        lich = np.clip(tnoise(s, s, seed + 4, 6, 6, 4) * 2 - 0.6, 0, 1) * 0.6
        base = lerp(base, col("8a8a4e"), lich)
        dark = (1 - ss(0.0, 0.05, facet)) * ss(-0.1, 0.4, tnoise(s, s, seed + 6, 5, 5, 3))
        base = lerp(base, col("3c3a36"), dark * 0.45)
        strength = 10.0
    else:  # cliff with strata
        Y = np.arange(s)[:, None] / s + np.zeros((1, s))
        warp = 0.03 * tnoise(s, s, seed + 5, 3, 3, 3)
        layer = (Y + warp) * 12
        lid = np.floor(layer)
        lf = layer - lid
        lr = nz.hash_float(lid.astype(np.int64) % 12, seed)
        hgt = 0.35 * ss(0.0, 0.15, lf) * (1 - ss(0.85, 1.0, lf)) + 0.4 * n + 0.15 * facet + 0.1 * lr
        base = lerp(col("6a6258"), col("9a8f7e"), lr[..., None])
        base = lerp(base, col("4e4840"), (1 - ss(0.0, 0.1, lf))[..., None] * 0.7)
        base = base * (0.85 + 0.25 * n[..., None])
        strength = 12.0
    return np.clip(base, 0, 1), normal_from_height(hgt, strength)


def stone_blocks(seed):
    s = 1024
    rows = 8
    rng = np.random.default_rng(seed)
    hgt = np.zeros((s, s))
    color = np.zeros((s, s, 3))
    Y, X = np.mgrid[0:s, 0:s]
    rh = s // rows
    n = tnoise(s, s, seed + 1, 8, 8, 5)
    for rI in range(rows):
        widths = []
        total = 0
        while total < s:
            wb = int(rng.uniform(0.18, 0.4) * s)
            widths.append(wb)
            total += wb
        widths[-1] -= total - s
        if widths[-1] < s * 0.08:
            widths[-2] += widths[-1]
            widths.pop()
        off = int(rng.uniform(0, s))
        x = off
        for wb in widths:
            tint = col("8d8576") * rng.uniform(0.8, 1.1) + rng.normal(0, 0.006, 3)
            y0 = rI * rh
            xs = (X - x) % s
            m = (xs < wb) & (Y >= y0) & (Y < y0 + rh)
            ex = np.minimum(xs, wb - xs)
            ey = np.minimum(Y - y0, y0 + rh - Y)
            e = np.minimum(ex, ey)
            bevel = ss(2, 14 + 6 * n, e)
            hgt[m] = (bevel * (0.7 + 0.3 * rng.uniform()))[m]
            color[m] = (tint * (0.85 + 0.25 * bevel)[..., None])[m]
            x += wb
    mortar = hgt < 0.15
    color[mortar] = (col("4a463e") * (0.9 + 0.2 * n[..., None]))[mortar]
    hgt += 0.12 * n
    moss = np.clip(tnoise(s, s, seed + 2, 4, 4, 4) * 2 - 0.5, 0, 1) * (1 - ss(0.2, 0.7, hgt))
    color = lerp(color, col("4e5e2e"), moss * 0.8)
    return np.clip(color * (0.9 + 0.15 * n[..., None]), 0, 1), normal_from_height(hgt, 8.0)


def wood(seed):
    s = 512
    grain = tnoise(s, s, seed, 1, 2, 3) + 0.15 * tnoise(s, s, seed + 7, 8, 2, 2)
    rings = np.sin((np.arange(s)[None, :] / s * 2 * np.pi * 9) + grain * 6)
    hgt = 0.3 * rings + 0.4 * tnoise(s, s, seed + 1, 3, 24, 3)
    crack = ss(0.82, 0.95, np.abs(tnoise(s, s, seed + 2, 6, 1, 3))) * ss(0.0, 0.3, tnoise(s, s, seed + 3, 2, 4, 2))
    hgt -= 0.6 * crack
    base = lerp(col("6b5a46"), col("9a8a72"), np.clip(rings * 0.3 + 0.5 + grain * 0.3, 0, 1))
    base = lerp(base, col("2e261e"), crack * 0.8)
    return np.clip(base, 0, 1), normal_from_height(hgt, 6.0)


# -- medieval and fantasy surfaces (2 m per repeat unless noted) ------------------------------------------------------

def plaster(seed):
    """Lime-washed daub: soft lumps, hairline cracks, weathering streaks and a grimy plinth band at the bottom."""
    s = 512
    n = tnoise(s, s, seed, 4, 4, 5)
    lump = tnoise(s, s, seed + 1, 16, 16, 3)
    f1, f2, _ = voronoi(s, s, 6, 6, seed + 2, warp=0.6)
    crack = 1 - ss(0.0, 0.012, f2 - f1)
    streak = np.clip(tnoise(s, s, seed + 3, 24, 1, 3), 0, 1) * ss(0.2, 0.9, np.linspace(0, 1, s))[:, None]
    hgt = 0.5 * lump + 0.25 * n - 0.5 * crack
    c = lerp(col("d8cfba"), col("efe7d4"), np.clip(0.5 + 0.5 * n + 0.2 * lump, 0, 1)[..., None])
    c = lerp(c, col("8f8270"), (crack * 0.6 + streak * 0.35)[..., None])
    return np.clip(c, 0, 1), normal_from_height(hgt, 3.0)


def thatch(seed):
    """Straw thatch in overlapping courses (v runs down the roof slope): fibres along v, darker weathered course ends."""
    s = 512
    rng = np.random.default_rng(seed)
    Y, X = np.mgrid[0:s, 0:s] / s
    fib = tnoise(s, s, seed, 96, 2, 3)
    courses = 8
    cv = (Y * courses + 0.15 * tnoise(s, s, seed + 1, 8, 1, 2)) % 1.0
    bund = tnoise(s, s, seed + 2, 40, 4, 2)
    hgt = 0.5 * fib + 0.6 * cv + 0.2 * bund
    c = lerp(col("9a7a3c"), col("c9a85e"), np.clip(0.5 + 0.6 * fib + 0.2 * bund, 0, 1)[..., None])
    c = lerp(c, col("5a4a2a"), (ss(0.75, 1.0, cv) * 0.55)[..., None])
    moss = np.clip(tnoise(s, s, seed + 3, 4, 4, 4) * 1.8 - 0.6, 0, 1)
    c = lerp(c, col("5d6430"), moss[..., None] * 0.5)
    _ = rng
    return np.clip(c, 0, 1), normal_from_height(hgt, 5.0)


def roof_slate(seed):
    """Overlapping slates in staggered courses (8 per repeat), each slate tinted, with dark gaps."""
    s = 512
    rng = np.random.default_rng(seed)
    rows, cols = 8, 6
    Y, X = np.mgrid[0:s, 0:s]
    ry = (Y * rows / s)
    r = np.floor(ry).astype(int)
    fy = ry - r
    xx = (X * cols / s + 0.5 * (r % 2)) % cols
    ci = np.floor(xx).astype(int)
    fx = xx - ci
    tint = nz.hash_float(r % rows, ci % cols, seed)
    n = tnoise(s, s, seed + 1, 8, 8, 4)
    edge = np.minimum(ss(0.0, 0.05, fx), ss(1.0, 0.95, fx))
    lap = ss(0.0, 0.1, fy)
    hgt = 0.6 * fy * edge + 0.15 * n
    c = lerp(col("3d4248"), col("5d6066"), np.clip(tint + 0.25 * n, 0, 1)[..., None])
    c = lerp(col("1e2023"), c, (edge * lap)[..., None])
    lich = np.clip(tnoise(s, s, seed + 2, 6, 6, 4) * 2.2 - 1.0, 0, 1) * (tint > 0.4)
    c = lerp(c, col("9a9a6a"), lich[..., None] * 0.45)
    _ = rng
    return np.clip(c, 0, 1), normal_from_height(hgt, 6.0)


def timber(seed):
    """Dark, adzed oak beams (grain along u): darker than the planks of `wood`."""
    s = 512
    grain = tnoise(s, s, seed, 2, 1, 3) + 0.2 * tnoise(s, s, seed + 7, 2, 16, 2)
    rings = np.sin((np.arange(s)[:, None] / s * 2 * np.pi * 7) + grain * 5)
    adze = tnoise(s, s, seed + 1, 10, 3, 2)
    hgt = 0.25 * rings + 0.5 * adze
    c = lerp(col("2c2016"), col("5a4430"), np.clip(rings * 0.3 + 0.5 + grain * 0.25 + adze * 0.2, 0, 1)[..., None])
    return np.clip(c, 0, 1), normal_from_height(hgt, 5.0)


def cloth(seed, field, charge):
    """A heraldic banner, one repeat per banner (u across, v up): field colour, a border and a charged lozenge."""
    s = 512
    Y, X = np.mgrid[0:s, 0:s] / s
    v = 1 - Y
    weave = 0.5 + 0.5 * np.sin(X * s * np.pi * 0.5) * np.sin(Y * s * np.pi * 0.5)
    n = tnoise(s, s, seed, 4, 4, 3)
    c = np.broadcast_to(col(field), (s, s, 3)).copy()
    border = (X < 0.07) | (X > 0.93) | (v < 0.07)
    c[border] = col(charge)
    loz = (np.abs(X - 0.5) / 0.28 + np.abs(v - 0.55) / 0.3) < 1
    c[loz] = col(charge)
    inner = (np.abs(X - 0.5) / 0.14 + np.abs(v - 0.55) / 0.15) < 1
    c[inner] = col(field)
    c = c * (0.85 + 0.12 * weave[..., None] + 0.08 * n[..., None])
    return np.clip(c, 0, 1), normal_from_height(0.3 * weave + 0.4 * n, 1.5)


def crystal(seed):
    """Glassy violet-blue crystal: banded facets with bright inclusions (the material also glows)."""
    s = 512
    n = tnoise(s, s, seed, 3, 6, 4)
    f1, f2, cid = voronoi(s, s, 5, 9, seed + 1, metric=(1.0, 2.5))
    r = cell_rand(cid, seed + 2)
    c = lerp(col("3b3a8a"), col("8fb6ff"), np.clip(0.4 + 0.5 * n + 0.3 * (r - 0.5), 0, 1)[..., None])
    c = lerp(c, col("e4ecff"), (1 - ss(0.0, 0.03, f2 - f1))[..., None] * 0.6)
    return np.clip(c, 0, 1), normal_from_height(0.6 * r + 0.2 * n, 2.0)


def bone(seed):
    """Old weathered bone: ivory with brown pores, lichen and long cracks along u."""
    s = 512
    n = tnoise(s, s, seed, 4, 4, 5)
    pores = ss(0.55, 0.8, tnoise(s, s, seed + 1, 48, 48, 2))
    crack = ss(0.85, 0.97, np.abs(tnoise(s, s, seed + 2, 2, 12, 3)))
    c = lerp(col("cfc3a4"), col("e8e0c8"), np.clip(0.5 + 0.5 * n, 0, 1)[..., None])
    c = lerp(c, col("7a6a4a"), (pores * 0.4 + crack * 0.6)[..., None])
    lich = np.clip(tnoise(s, s, seed + 3, 5, 5, 4) * 2 - 0.8, 0, 1)
    c = lerp(c, col("8a9050"), lich[..., None] * 0.5)
    return np.clip(c, 0, 1), normal_from_height(0.4 * n - 0.5 * crack - 0.2 * pores, 4.0)


# -- terrain sources ---------------------------------------------------------------------------------------------------

def terrain_source(name, seed):
    s = 512
    n = tnoise(s, s, seed, 4, 4, 5)
    fine = tnoise(s, s, seed + 1, 64, 64, 2)
    f1, f2, cid = voronoi(s, s, 24, 24, seed + 2)
    r = cell_rand(cid, seed + 3)
    if name == "grass":
        c = lerp(col("3f5a24"), col("6e8a3a"), np.clip(0.5 + 0.5 * n + 0.35 * fine, 0, 1))
        c = lerp(c, col("8a8a44"), np.clip(r[..., None] - 0.82, 0, 1) * 3)
    elif name == "dry_grass":
        c = lerp(col("75703c"), col("a99a5c"), np.clip(0.5 + 0.5 * n + 0.4 * fine, 0, 1))
    elif name == "soil":
        c = lerp(col("4a3a2a"), col("6e5640"), np.clip(0.5 + 0.6 * n + 0.2 * fine, 0, 1))
        peb = ss(0.35, 0.25, f1) * (r > 0.6)
        c = lerp(c, col("8a8070") * (0.8 + 0.3 * r[..., None]), peb)
    elif name == "forest_floor":
        c = lerp(col("33281c"), col("5a4630"), np.clip(0.5 + 0.6 * n + 0.3 * fine, 0, 1))
        f1b, _, cidb = voronoi(s, s, 40, 40, seed + 9, metric=(1.0, 2.2), warp=0.5)
        rb = cell_rand(cidb, seed + 10)
        leaf = ss(0.42, 0.3, f1b) * (rb > 0.5) * ss(-0.3, 0.3, n)
        lc = lerp(col("6a4422"), col("7f6a36"), rb[..., None])
        c = lerp(c, lc, leaf * 0.55)
        c = lerp(c, col("3a4a22"), np.clip(n - 0.25, 0, 1) * 1.2)  # moss patches
    elif name == "stone":
        c = lerp(col("6a665e"), col("8f897d"), np.clip(0.5 + 0.4 * n + 0.3 * (r - 0.5), 0, 1))
        c = lerp(c, col("4a4640"), (1 - ss(0.0, 0.08, f2 - f1)) * 0.35)
    elif name == "mountain_rock":
        c = lerp(col("55524e"), col("7d7770"), np.clip(0.5 + 0.6 * n + 0.2 * fine, 0, 1))
        c = lerp(c, col("6e6a50"), np.clip(tnoise(s, s, seed + 5, 6, 6, 3) - 0.3, 0, 1))
    elif name == "snow":
        c = lerp(col("cfd6de"), col("f2f4f6"), np.clip(0.6 + 0.4 * n + 0.15 * fine, 0, 1))
    elif name == "crops":
        c = lerp(col("a08a3e"), col("cdb460"), np.clip(0.5 + 0.5 * n + 0.4 * fine, 0, 1))
    elif name == "ploughed":
        c = lerp(col("3e2e20"), col("5e4732"), np.clip(0.5 + 0.6 * n + 0.3 * fine, 0, 1))
    else:  # path
        c = lerp(col("6e5a42"), col("94805e"), np.clip(0.5 + 0.5 * n + 0.3 * fine, 0, 1))
        peb = ss(0.3, 0.2, f1) * (r > 0.5)
        c = lerp(c, col("a49a88"), peb * 0.7)
    return np.clip(c, 0, 1)


def terrain_detail(seed):
    s = 1024
    n = tnoise(s, s, seed, 8, 8, 6)
    f1, f2, cid = voronoi(s, s, 40, 40, seed + 1)
    peb = ss(0.45, 0.0, f1) * (cell_rand(cid, seed + 2) > 0.55)
    hgt = 0.6 * n + 0.5 * peb + 0.15 * tnoise(s, s, seed + 3, 96, 96, 2)
    nrm = normal_from_height(hgt, 7.0)
    r = 512
    rough = np.clip(0.82 + 0.12 * tnoise(r, r, seed + 4, 8, 8, 4), 0, 1)
    return nrm, rough


SKY_TURBIDITY = 3.0
SUN_DISC_ENERGY = 0.01   # fraction of the DistantLight's irradiance carried by the sun disc in the sky texture


def sky_directions(h, w):
    """Unit directions (h, w, 3) of a Z-up lat-long dome texture (UsdLux DomeLight_1, poleAxis Z): row 0 is the
    zenith, column 0 is longitude +pi, longitude 0 points to -Y and longitude pi/2 to +X."""
    lat = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    lon = np.pi - (np.arange(w) + 0.5) / w * 2 * np.pi
    cl = np.cos(lat)[:, None]
    return np.stack([cl * np.sin(lon)[None, :], -cl * np.cos(lon)[None, :], np.sin(lat)[:, None] + 0 * lon], -1)


def _preetham(cos_t, cos_g, gamma, theta_s, T):
    """Preetham et al. 1999 sky: relative luminance Y and chromaticity x, y (Perez distribution)."""
    coef = {"Y": [(0.1787, -1.4630), (-0.3554, 0.4275), (-0.0227, 5.3251), (0.1206, -2.5771), (-0.0670, 0.3703)],
            "x": [(-0.0193, -0.2592), (-0.0665, 0.0008), (-0.0004, 0.2125), (-0.0641, -0.8989), (-0.0033, 0.0452)],
            "y": [(-0.0167, -0.2608), (-0.0950, 0.0092), (-0.0079, 0.2102), (-0.0441, -1.6537), (-0.0109, 0.0529)]}
    t, t2, t3 = theta_s, theta_s ** 2, theta_s ** 3
    chi = (4 / 9 - T / 120) * (np.pi - 2 * theta_s)
    zen = {"Y": (4.0453 * T - 4.9710) * np.tan(chi) - 0.2155 * T + 2.4192,
           "x": T * T * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t) + T * (-0.02903 * t3 + 0.06377 * t2
                - 0.03202 * t + 0.00394) + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * t + 0.25886),
           "y": T * T * (0.00275 * t3 - 0.00610 * t2 + 0.00317 * t) + T * (-0.04214 * t3 + 0.08970 * t2
                - 0.04153 * t + 0.00516) + (0.15346 * t3 - 0.26756 * t2 + 0.06670 * t + 0.26688)}

    def perez(k, ct, cg, g):
        A, B, C, D, E = (a * T + b for a, b in coef[k])
        return (1 + A * np.exp(B / ct)) * (1 + C * np.exp(D * g) + E * cg * cg)
    out = {}
    for k in ("Y", "x", "y"):
        out[k] = zen[k] * perez(k, cos_t, cos_g, gamma) / perez(k, 1.0, np.cos(theta_s), theta_s)
    return out


def _xyY_to_linear_srgb(x, y, Y):
    X, Z = x * Y / y, (1 - x - y) * Y / y
    m = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    return np.einsum("ij,...j->...i", m, np.stack([X, Y, Z], -1))


def sky(seed, lighting, h=2048, w=4096, mean_radiance=0.425):
    """HDR lat-long sky (linear Rec.709 radiance, Z-up, see sky_directions), consistent with the [lighting] sun:
    Preetham sky for the sun's elevation, a sun disc of the DistantLight's angular size and colour at its
    direction (carrying SUN_DISC_ENERGY of its irradiance, so image-based lighting does not count the sun twice),
    clouds lit from the sun and thinned around it, and a ground hemisphere lit by sun and sky. Scaled so the sky
    hemisphere averages `mean_radiance` (before the dome's intensity), the level of the earlier 8-bit sky."""
    d = sky_directions(h, w)
    sun = np.asarray(lighting.sun_direction, float)
    sun /= np.linalg.norm(sun)
    theta_s = float(np.arccos(sun[2]))
    el = np.arcsin(np.clip(d[..., 2], -1, 1))
    cos_g = np.clip(d @ sun, -1, 1)
    gamma = np.arccos(cos_g)
    ct = np.maximum(d[..., 2], 0.02)
    p = _preetham(ct, cos_g, gamma, theta_s, SKY_TURBIDITY)
    c = np.clip(_xyY_to_linear_srgb(p["x"], p["y"], p["Y"]), 0, None)
    # Preetham's chromaticity turns mauve towards the horizon at midday; pull the lowest ~20 deg to a pale
    # blue-white haze of the same luminance.
    haze = (c @ np.array([0.2126, 0.7152, 0.0722]))[..., None] * np.array([0.86, 0.98, 1.2])
    c = lerp(c, haze, 0.85 * (1 - ss(0.0, 0.35, np.maximum(el, 0))))
    # Normalise on the upper hemisphere (solid-angle weighted) to the target mean radiance.
    wgt = np.cos(el) * (el > 0)
    c *= mean_radiance / ((c.mean(-1) * wgt).sum() / wgt.sum())
    lum = c @ np.array([0.2126, 0.7152, 0.0722])
    sky_mean = (lum * wgt).sum() / wgt.sum()
    # Clouds: tileable around the azimuth, band-limited to the sky, thinned within ~20 deg of the sun so the disc
    # (and the DistantLight's hard shadows) stay unobstructed. Lit with a forward-scattering phase (silver lining).
    cl = tnoise(h, w, seed, 8, 4, 5)
    cloud = ss(0.05, 0.45, cl) * ss(0.02, 0.15, el) * (1 - ss(0.8, 1.3, el)) * ss(0.15, 0.35, gamma)
    g = 0.6
    phase = (1 - g * g) / (1 + g * g - 2 * g * cos_g) ** 1.5
    cloud_c = col("f4f2ee")[None, None, :] * (sky_mean * (1.1 + 0.25 * phase))[..., None]
    c = lerp(c, cloud_c, cloud * 0.8)
    k = mean_radiance / ((c.mean(-1) * wgt).sum() / wgt.sum())
    c, sky_mean = c * k, sky_mean * k
    # Sun disc (limb-darkened) at the DistantLight's direction, angular diameter and colour.
    r = np.radians(float(lighting.sun_angle)) / 2
    omega = 2 * np.pi * (1 - np.cos(r))
    e_sun = float(lighting.sun_intensity) / float(lighting.sky_intensity)   # sun irradiance in dome units
    mu = np.sqrt(np.clip(1 - (gamma / r) ** 2, 0, 1))
    limb = (1 - 0.6 * (1 - mu)) / (1 - 0.6 / 3)          # mean 1 over the disc
    pix = ss(-0.5, 0.5, (r - gamma) / (np.pi / h))      # antialiased edge, about one texel
    disc = SUN_DISC_ENERGY * e_sun / omega * limb * pix
    c = c + disc[..., None] * np.asarray(lighting.sun_color, float)[None, None, :]
    # Ground: albedo lit by the sun and the sky's irradiance, fading into horizon haze just below the horizon.
    e_sky = np.pi * sky_mean
    ground = col("4a5a3c") * 0.35 * (e_sun * sun[2] + e_sky) / np.pi
    horizon = c[h // 2 - 1]
    below = el < 0
    hz = np.clip(-el * 10, 0, 1)[..., None]
    c = np.where(below[..., None], horizon[None, :, :] * (1 - hz) + ground * hz, c)
    return c.astype(np.float32)


# =====================================================================================================================
# Public entry points
# =====================================================================================================================

def _out(cfg, rel):
    return os.path.join(cfg.out, rel)


def build_shared(cfg, force=False):
    """Write every shared texture into <out>/textures/. Returns {name: path relative to cfg.out}."""
    t0 = time.time()
    seed = cfg.seed
    td = atlas.TEXTURE_DIR
    out = {}

    def job(name, rel, fn):
        out[name] = rel
        p = _out(cfg, rel)
        if os.path.exists(p) and not force:
            return
        write_image(p, fn())

    for i, (name, fname) in enumerate(atlas.FOLIAGE_TEXTURES.items()):
        job(name, f"{td}/{fname}", lambda name=name, i=i: foliage_atlas(name, seed + 1000 + i))
    job("grass_atlas", f"{td}/{atlas.GRASS_ATLAS}", lambda: grass_atlas(seed + 2000))
    job("fern", f"{td}/{atlas.FERN_TEXTURE}", lambda: fern(seed + 2100))

    surf = {
        "bark_oak": lambda s: bark("oak", s), "bark_pine": lambda s: bark("pine", s),
        "bark_birch": lambda s: bark("birch", s), "bark_dead": lambda s: bark("dead", s),
        "rock": lambda s: rock("rock", s), "rock_cliff": lambda s: rock("cliff", s),
        "stone_blocks": stone_blocks, "wood": wood,
        "plaster": plaster, "thatch": thatch, "roof_slate": roof_slate, "timber": timber,
        "cloth_red": lambda s: cloth(s, "8a1c1c", "d8b040"), "cloth_blue": lambda s: cloth(s, "1d3570", "e6e2d0"),
        "canvas": lambda s: cloth(s, "d9cfb6", "d9cfb6"),
        "crystal": crystal, "bone": bone,
    }
    for i, (name, (cfile, nfile)) in enumerate(atlas.SURFACE_TEXTURES.items()):
        cp, np_ = f"{td}/{cfile}", f"{td}/{nfile}"
        out[name + "_color"] = cp
        out[name + "_normal"] = np_
        if force or not (os.path.exists(_out(cfg, cp)) and os.path.exists(_out(cfg, np_))):
            c, n = surf[name](seed + 3000 + i * 17)
            write_image(_out(cfg, cp), c)
            write_image(_out(cfg, np_), n)
    for i, (name, fname) in enumerate(atlas.TERRAIN_SOURCES.items()):
        job("terrain_" + name, f"{td}/{fname}", lambda name=name, i=i: terrain_source(name, seed + 4000 + i * 13))
    dn, dr = f"{td}/{atlas.TERRAIN_DETAIL_NORMAL}", f"{td}/{atlas.TERRAIN_DETAIL_ROUGHNESS}"
    out["terrain_detail_normal"] = dn
    out["terrain_detail_roughness"] = dr
    if force or not (os.path.exists(_out(cfg, dn)) and os.path.exists(_out(cfg, dr))):
        n, r = terrain_detail(seed + 5000)
        write_image(_out(cfg, dn), n)
        write_image(_out(cfg, dr), r)
    job("sky", f"{td}/{atlas.SKY_TEXTURE}", lambda: sky(seed + 6000, cfg.c.lighting))
    _log(f"shared textures ready ({time.time() - t0:.1f}s)")
    return out


def _source_means(cfg, size=16):
    """Terrain source textures prefiltered to `size`^2 (linear colour), for sampling at bake resolution."""
    out = {}
    for name, fname in atlas.TERRAIN_SOURCES.items():
        img = read_image(_out(cfg, f"{atlas.TEXTURE_DIR}/{fname}"))[..., :3]
        k = img.shape[0] // size
        out[name] = srgb_to_linear(img.reshape(size, k, size, k, 3).mean((1, 3)))
    return out


def _sample_wrap(tex, x, y, repeat):
    """Bilinear wrap sample of a small (s, s, 3) texture at world (x, y) with `repeat` metres per tile."""
    s = tex.shape[0]
    fx = (x / repeat) * s - 0.5
    fy = (1.0 - (y / repeat) % 1.0) * s - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x0 %= s
    y0 %= s
    x1 = (x0 + 1) % s
    y1 = (y0 + 1) % s
    return (tex[y0, x0] * (1 - tx) + tex[y0, x1] * tx) * (1 - ty) + (tex[y1, x0] * (1 - tx) + tex[y1, x1] * tx) * ty


def bake_region(cfg, world, region_cell, force=False):
    """Bake the region's terrain colour (full and far) and the proxy's macro normal map. Returns relative paths."""
    paths = atlas.region_texture_paths(region_cell.name)
    if not force and all(os.path.exists(_out(cfg, p)) for p in paths.values()):
        return paths
    t0 = time.time()
    seed = cfg.seed + 7000
    S = region_cell.level.size
    x0, y0 = region_cell.x0, region_cell.y0
    N = int(cfg.c.terrain.region_texture_size)
    src = _source_means(cfg)
    rep = atlas.TERRAIN_REPEAT_M
    color = np.zeros((N, N, 3))
    px = S / N
    rows = 256
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        X = x0 + (np.arange(N)[None, :] + 0.5) * px + np.zeros((r1 - r0, 1))
        Y = y0 + S - (np.arange(r0, r1)[:, None] + 0.5) * px + np.zeros((1, N))
        color[r0:r1] = _bake_block(world, src, rep, X, Y, seed)
    color_s = linear_to_srgb(color)
    write_image(_out(cfg, paths["color"]), color_s, quality=90)

    coarse = region_cell.level.far_spec()   # the level drawn with the far material (the proxy)
    Nf = int(coarse.get("texture_size", 1024))
    k = N // Nf
    far = color[: Nf * k, : Nf * k].reshape(Nf, k, Nf, k, 3).mean((1, 3)) if k >= 1 else resize_bilinear(color, Nf, Nf)
    write_image(_out(cfg, paths["color_far"]), linear_to_srgb(far), quality=90)
    write_image(_out(cfg, paths["normal_far"]), _macro_normal(world, region_cell, Nf, int(coarse["terrain_quads"])))
    _log(f"baked {region_cell.name} ({time.time() - t0:.1f}s)")
    return paths


def _bake_block(world, src, rep, X, Y, seed, sampled=False):
    """Terrain colour at world (X, Y). sampled=False: each source contributes its mean colour (region bake, ~0.5 m
    texels, where a 4 m source repeat only aliases); sampled=True: the prefiltered sources are sampled in world space
    (the fine ground's per-tile bake, ~0.25 m texels), so the source texture's structure shows."""
    def S(name):
        return _sample_wrap(src[name], X, Y, rep) if sampled else src[name].reshape(-1, 3).mean(0)
    f = {n: world.field(n, X, Y) for n in ("forest", "conifer", "rock", "snow", "meadow", "moisture", "wet", "path",
                                           "alpine", "mount", "poi")}
    # Fine slope from the 1 m heightfield.
    d = 0.75
    hx = world.height(X + d, Y) - world.height(X - d, Y)
    hy = world.height(X, Y + d) - world.height(X, Y - d)
    slope = np.degrees(np.arctan(np.hypot(hx, hy) / (2 * d)))
    b1 = nz.fbm(X, Y, seed + 1, 9.0, 3)        # metre-scale breakup
    b2 = nz.fbm(X, Y, seed + 2, 38.0, 3)
    macro = nz.fbm(X, Y, seed + 3, 260.0, 3)
    rock_w = np.clip(np.maximum(f["rock"] * 0.8, ss(30.0, 44.0, slope + 6 * b1)) + 0.15 * b2, 0, 1)
    mtn = np.clip(f["alpine"] + ss(0.25, 0.6, f["mount"]), 0, 1)
    w = {}
    w["mountain_rock"] = rock_w * mtn
    w["stone"] = rock_w * (1 - mtn)
    w["snow"] = np.clip(f["snow"] * (1.2 + 0.6 * b1), 0, 1) * (1 - 0.6 * rock_w)
    soft = (1 - rock_w) * (1 - w["snow"])
    forest = np.clip(f["forest"] + 0.22 * b2 + 0.12 * b1, 0, 1)
    under = ss(0.2, 0.85, forest)
    # Under trees: leaf litter mixed with moss / sparse grass, denser litter under conifers.
    litter = under * np.clip(0.55 + 0.25 * f["conifer"] + 0.3 * b1, 0, 0.9)
    w["forest_floor"] = soft * litter
    open_ = soft * (1 - litter)
    dry = np.clip(ss(0.6, 0.2, f["moisture"] + 0.3 * b2) * 0.8 + 0.35 * f["meadow"] * ss(0.0, 0.5, b1)
                  + 0.2 * ss(0.2, 0.7, nz.fbm(X, Y, seed + 4, 18.0, 2)), 0, 1)
    soil = np.clip(ss(0.45, 0.85, b1 * 0.7 + b2 * 0.6) * 0.45 + ss(18.0, 30.0, slope) * 0.6 + 0.3 * f["alpine"], 0, 1)
    w["soil"] = open_ * soil * 0.8 + soft * ss(0.6, 1.0, f["wet"]) * 0.3
    w["grass"] = open_ * (1 - soil * 0.8) * (1 - dry)
    w["dry_grass"] = open_ * (1 - soil * 0.8) * dry
    path = 1.0 - ss(-0.6 + 0.5 * b1, 1.2 + 0.5 * b1, f["path"])
    farm = world.field("farm", X, Y) * (1 - rock_w) * ss(-2.0, 6.0, f["poi"])
    total = sum(w.values()) + 1e-6
    out = np.zeros(X.shape + (3,))
    # Metre-scale variation comes from world-space noise (continuous across region and tile borders).
    for name, wt in w.items():
        out += S(name) * (wt / total)[..., None]
    if farm.max() > 0:
        pc = _farm_parcels(world.cfg, S, X, Y, seed)
        out = out * (1 - farm[..., None]) + pc * farm[..., None]
    out = out * (1 - path[..., None]) + S("path") * path[..., None]
    fine = nz.fbm(X, Y, seed + 5, 2.2, 2)
    out *= (1.0 + 0.09 * fine)[..., None]
    # Macro tint variation, darker wet ground, forest shade, metre-scale clumps.
    out *= (1.0 + 0.10 * macro + 0.08 * b2 + 0.08 * b1)[..., None]
    # Desaturate slightly towards a common earthy tone so biomes share one palette.
    lum = out.mean(-1, keepdims=True)
    out = out * 0.85 + lum * 0.15
    out *= (1.0 - 0.18 * ss(0.5, 1.2, f["wet"]) * (1 - rock_w))[..., None]
    out *= (1.0 - 0.15 * f["forest"] * f["conifer"])[..., None]
    out *= (1.0 - 0.25 * ss(38.0, 60.0, slope) * rock_w + 0.08 * rock_w * b1)[..., None]
    out = out * (1 - 0.08 * f["meadow"][..., None]) + srgb_to_linear(col("8f8a40")) * 0.08 * f["meadow"][..., None]
    return np.clip(out, 0, 1)


def farm_strips(cfg, X, Y, seed):
    """Layout of the medieval open fields at world (X, Y), per farm area in its `angle`: furlongs of 160 m along v,
    staggered per 120 m block of u, and strips (selions) 9-18 m wide inside a furlong. Yields one dict per farm area
    that reaches (X, Y): weight `wt`, strip coordinate `u`, strip width `sw` and index `su`, distance `eu` to the
    nearest strip edge, distance `ev` from the furlong's middle, and per-strip hashes `kind` and `tint`."""
    for k, a in enumerate(a for a in cfg.c.get("biomes", {}).get("areas", []) if a["kind"] == "farm"):
        d = np.hypot(X - a["center"][0], Y - a["center"][1])
        wt = ss(float(a["radius"]) * 1.4, float(a["radius"]) * 0.6, d)
        if wt.max() <= 0:
            continue
        th = np.radians(float(a.get("angle", 0.0)))
        u = (X - a["center"][0]) * np.cos(th) + (Y - a["center"][1]) * np.sin(th)
        v = -(X - a["center"][0]) * np.sin(th) + (Y - a["center"][1]) * np.cos(th)
        fu = np.floor(u / 120.0)
        voff = v + 80.0 * nz.hash_float(fu.astype(np.int64), k, seed)
        fv = np.floor(voff / 160.0)
        sw = 9.0 + 9.0 * nz.hash_float(fu.astype(np.int64), fv.astype(np.int64), seed + 1)
        su = np.floor(u / sw)
        yield dict(wt=wt, u=u, sw=sw, su=su, eu=np.minimum(u - su * sw, (su + 1) * sw - u),
                   ev=np.abs((voff % 160.0) - 80.0),
                   kind=nz.hash_float(su.astype(np.int64), fv.astype(np.int64), seed + 2 + k),
                   tint=nz.hash_float(su.astype(np.int64), fv.astype(np.int64), seed + 7 + k))


def _farm_parcels(cfg, S, X, Y, seed):
    """Medieval open-field farming: each strip a crop, fallow or ploughed, with grass balks between them and furrows
    along each strip (layout: farm_strips). S(name) gives a terrain source's colour (mean or sampled)."""
    src = {k: S(k) for k in ("crops", "ploughed", "grass", "dry_grass")}
    out = np.zeros(X.shape + (3,))
    wsum = np.zeros(X.shape) + 1e-9
    for fs in farm_strips(cfg, X, Y, seed):
        kind, tint, u = fs["kind"], fs["tint"][..., None], fs["u"]
        crops = src["crops"] * (0.85 + 0.3 * tint)
        green = src["grass"] * (1.05 + 0.25 * tint)
        plough = src["ploughed"] * (0.85 + 0.3 * tint)
        fallow = src["dry_grass"] * (0.9 + 0.2 * tint)
        c = np.where((kind < 0.38)[..., None], crops, np.where((kind < 0.62)[..., None], green,
                                                               np.where((kind < 0.85)[..., None], plough, fallow)))
        # Furrows along the strip, a grass balk at each strip edge and a headland at each furlong end.
        fur = 0.88 + 0.12 * np.cos(2 * np.pi * u / 1.1)
        c = c * np.where((kind < 0.62) | (kind >= 0.85), 1.0, fur)[..., None]
        balk = np.maximum(ss(0.9, 0.3, fs["eu"]), ss(76.0, 79.0, fs["ev"]))
        c = c * (1 - balk[..., None]) + src["grass"] * 0.9 * balk[..., None]
        out += c * fs["wt"][..., None]
        wsum += fs["wt"]
    return out / wsum[..., None]


def _macro_normal(world, region_cell, N, quads):
    """Fine 1 m normal expressed in the tangent frame of the proxy's interpolated vertex normals.

    The proxy mesh samples world.grid() every S/quads metres; its vertex normals are taken as central differences
    of that coarse grid (np.gradient), interpolated bilinearly across each quad.
    """
    S = region_cell.level.size
    x0, y0 = region_cell.x0, region_cell.y0
    step = S / quads
    hc = world.grid(x0, y0, quads, int(round(step / world.spacing)))
    gy, gx = np.gradient(hc, step)
    nc = np.stack([-gx, -gy, np.ones_like(hc)], -1)
    nc /= np.linalg.norm(nc, axis=-1, keepdims=True)
    px = S / N
    X = x0 + (np.arange(N)[None, :] + 0.5) * px + np.zeros((N, 1))
    Y = y0 + S - (np.arange(N)[:, None] + 0.5) * px + np.zeros((1, N))
    # Interpolated coarse normal.
    fx = (X - x0) / step
    fy = (Y - y0) / step
    ix = np.clip(np.floor(fx).astype(int), 0, quads - 1)
    iy = np.clip(np.floor(fy).astype(int), 0, quads - 1)
    tx = (fx - ix)[..., None]
    ty = (fy - iy)[..., None]
    Nn = (nc[iy, ix] * (1 - tx) + nc[iy, ix + 1] * tx) * (1 - ty) + (nc[iy + 1, ix] * (1 - tx) + nc[iy + 1, ix + 1] * tx) * ty
    Nn /= np.linalg.norm(Nn, axis=-1, keepdims=True)
    # Fine normal from the heightfield.
    d = max(px, world.spacing)
    hx = world.height(X + d, Y) - world.height(X - d, Y)
    hy = world.height(X, Y + d) - world.height(X, Y - d)
    nf = np.stack([-hx, -hy, np.full_like(hx, 2 * d)], -1)
    nf /= np.linalg.norm(nf, axis=-1, keepdims=True)
    T = np.array([1.0, 0, 0]) - Nn[..., 0:1] * Nn
    T /= np.linalg.norm(T, axis=-1, keepdims=True)
    B = np.cross(Nn, T)
    tn = np.stack([(nf * T).sum(-1), (nf * B).sum(-1), (nf * Nn).sum(-1)], -1)
    tn /= np.linalg.norm(tn, axis=-1, keepdims=True)
    return tn * 0.5 + 0.5
