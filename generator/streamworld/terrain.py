"""World heightfield and biome fields.

The heightfield is one global grid at `terrain.spacing` (1 m by default). It is built once per configuration and
cached: a macro layout (mountain ranges, valley, rolling plains, strata) is evaluated on the erosion grid, eroded
with a vectorized particle (droplet) simulation, upsampled to the final spacing with cubic interpolation, given
fine detail and has its footpaths carved in. Every terrain mesh at every level samples this one grid, so shared
edges of neighbouring cells at the same level are bit-identical and coarser levels sample a subset of the same
points.

Biome fields (forest density, moisture, rock, snow, meadow, paths) are derived from the heightfield and noise on a
coarser grid and sampled bilinearly.
"""

import os
import time

import numpy as np

from . import noise as nz

FIELD_SPACING = 4.0
# How far each kind of point of interest levels the ground under it (0..1); kinds not listed keep the terrain.
LEVEL_GROUND = {"village": 0.6, "hamlet": 0.65, "castle": 0.9, "windmill": 0.85, "graveyard": 0.8,
                "wizard_tower": 0.7, "dragon_bones": 0.75}


def _log(msg):
    print(f"[terrain] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------------------------------

def polyline_distance(px, py, pts):
    """Distance from points to a polyline, and the arc-length parameter (0..1) of the nearest point."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = pts[1:] - pts[:-1]
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = cum[-1]
    best = np.full(np.shape(px), np.inf)
    best_s = np.zeros(np.shape(px))
    for i in range(len(seg)):
        ax, ay = pts[i]
        dx, dy = seg[i]
        t = ((px - ax) * dx + (py - ay) * dy) / (seglen[i] ** 2)
        t = np.clip(t, 0.0, 1.0)
        qx = ax + t * dx
        qy = ay + t * dy
        d = np.hypot(px - qx, py - qy)
        m = d < best
        best = np.where(m, d, best)
        best_s = np.where(m, (cum[i] + t * seglen[i]) / total, best_s)
    return best, best_s


def bilinear(grid, x0, y0, spacing, x, y):
    """Bilinear sample of grid[j, i] (row = y) whose sample (0, 0) is at world (x0, y0)."""
    ny, nx = grid.shape
    fx = (np.asarray(x, dtype=np.float64) - x0) / spacing
    fy = (np.asarray(y, dtype=np.float64) - y0) / spacing
    fx = np.clip(fx, 0.0, nx - 1.000001)
    fy = np.clip(fy, 0.0, ny - 1.000001)
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    tx = fx - ix
    ty = fy - iy
    g00 = grid[iy, ix]
    g10 = grid[iy, ix + 1]
    g01 = grid[iy + 1, ix]
    g11 = grid[iy + 1, ix + 1]
    return (g00 * (1 - tx) + g10 * tx) * (1 - ty) + (g01 * (1 - tx) + g11 * tx) * ty


def _upsample2(a):
    """Upsample a grid by 2 with 4-point cubic interpolation (separable). Shape n -> 2n-1."""
    def axis0(a):
        n = a.shape[0]
        out = np.empty((2 * n - 1,) + a.shape[1:], dtype=a.dtype)
        out[0::2] = a
        p = np.concatenate([a[:1], a, a[-1:]], axis=0)
        out[1::2] = (-p[:-3] + 9 * p[1:-2] + 9 * p[2:-1] - p[3:]) / 16.0
        return out
    return axis0(axis0(a).T).T


def _box_blur(a, r):
    """Separable box blur with radius r samples (edge clamped)."""
    def blur1(a):
        p = np.concatenate([np.repeat(a[:1], r, 0), a, np.repeat(a[-1:], r, 0)], axis=0)
        c = np.cumsum(p, axis=0, dtype=np.float64)
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), c], axis=0)
        return ((c[2 * r + 1:] - c[:-2 * r - 1]) / (2 * r + 1)).astype(a.dtype)
    return blur1(blur1(a).T).T


def _smin(a, b, k):
    """Smooth minimum (polynomial), avoids the crease where a == b."""
    hh = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * hh - k * hh * (1.0 - hh)


def slope_deg(h, spacing):
    gy, gx = np.gradient(h, spacing)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


# ---------------------------------------------------------------------------------------------------------------------
# Macro layout
# ---------------------------------------------------------------------------------------------------------------------

def macro_height(cfg, x, y):
    """Un-eroded height (and the normalised mountain mask) at world coordinates."""
    t = cfg.c.terrain
    s = cfg.seed
    xw, yw = nz.warp(x, y, s + 101, 900.0, 260.0)

    # Rolling plains: long gentle swells, mid-scale hills and smaller hummocks.
    h = float(t.base_height) + 24.0 * nz.fbm(xw, yw, s + 1, 1000.0, 3) + 13.0 * nz.fbm(xw, yw, s + 3, 340.0, 3) \
        + 4.0 * nz.fbm(x, y, s + 2, 110.0, 3)
    # Broad swells and basins at the scale of the whole march (a few km).
    h += float(t.get("swell_height", 0.0)) * nz.fbm(xw, yw, s + 4, 3200.0, 2)

    # Mountain ranges along ridge polylines.
    mount = np.zeros_like(h)
    for i, r in enumerate(t.ranges):
        d, along = polyline_distance(xw, yw, r["points"])
        w = float(r["width"])
        mask = np.exp(-((d / w) ** 2) * 2.2)
        # Height varies along the ridge so the range reads as separate peaks and saddles.
        pts = np.asarray(r["points"], dtype=np.float64)
        length = np.hypot(*(pts[1:] - pts[:-1]).T).sum()
        var = 0.62 + 0.55 * nz.smoothstep(-0.4, 0.6, nz.fbm(along * length, i * 1000.0, s + 40 + i, 450.0, 2))
        mount = np.maximum(mount, float(r["height"]) * mask * var)
    mnorm = np.clip(mount / 600.0, 0.0, 1.5)
    rid = nz.ridged(xw, yw, s + 7, 620.0, 6)
    rid2 = nz.ridged(x, y, s + 8, 190.0, 4)
    hm = mount * (0.30 + 0.70 * rid) + mount * 0.12 * rid2
    # Foothills: ridged bumps fading out from the ranges.
    foot = nz.smoothstep(0.03, 0.35, mnorm) * (1.0 - nz.smoothstep(0.5, 1.0, mnorm))
    hm += 55.0 * foot * nz.ridged(xw, yw, s + 9, 260.0, 4)

    # Strata: terrace bands in the mountains create cliff faces once eroded. The bands follow a tilted, warped
    # bedding plane rather than the height contours, so they do not read as contour lines.
    rock = nz.smoothstep(0.2, 0.6, mnorm) * nz.smoothstep(0.0, 0.5, nz.fbm(x, y, s + 11, 450.0, 2))
    band = 24.0
    bed = hm + 0.035 * x - 0.02 * y + 18.0 * nz.fbm(x, y, s + 12, 260.0, 2)
    tt = bed / band
    fl = np.floor(tt)
    fr = tt - fl
    step = (fl + nz.smoothstep(0.6, 0.9, fr) - tt) * band
    hm = hm + step * 0.45 * rock * nz.smoothstep(0.15, 0.6, rid)

    # The valley cuts through the plains and, as a gorge, through the mountains: the carve depth includes most of
    # the local mountain height, and the floor is kept falling along the valley so it drains.
    for v in t.valleys:
        d, along = polyline_distance(x, y, v["points"])
        dw, _ = polyline_distance(xw * 0.2 + x * 0.8, yw * 0.2 + y * 0.8, v["points"])
        d = 0.5 * (d + dw)
        width = float(v["width"]) * (1.0 - 0.6 * np.clip(mnorm, 0, 1)) * (0.8 + 0.4 * (nz.fbm(x, y, s + 14, 500.0, 2) + 0.5))
        fw = float(v["floor_width"])
        prof = 1.0 - nz.smoothstep(fw * 0.5, fw * 0.5 + width, d)
        prof = prof * prof * (3 - 2 * prof)
        hm = hm * (1.0 - 0.88 * prof)
        h = h - float(v["depth"]) * 0.32 * prof
        floor_h = float(t.base_height) - 10.0 + 120.0 * (1.0 - along) ** 2.0
        h = h + (_smin(h, floor_h, 10.0) - h) * prof
    return h + hm, mnorm


# ---------------------------------------------------------------------------------------------------------------------
# Erosion
# ---------------------------------------------------------------------------------------------------------------------

def droplet_erosion(h, spacing, n_droplets, steps, seed, batch=200000, spawn=None, quiet=False):
    """Vectorized particle hydraulic erosion. Returns (eroded height, water-flow count). Droplets spawn inside
    `spawn` = (i0, i1, j0, j1) (grid columns / rows, default the whole grid) and die when they leave the grid."""
    h = h.astype(np.float64).copy()
    ny, nx = h.shape
    si0, si1, sj0, sj1 = spawn if spawn is not None else (1, nx - 2, 1, ny - 2)
    flow = np.zeros(h.size, dtype=np.float64)
    rng = np.random.default_rng(seed)
    inertia, capacity_k, min_slope = 0.08, 3.0, 0.004
    erode_k, deposit_k, evap, gravity = 0.25, 0.25, 0.025, 9.0
    max_erode = 0.6
    gy0, gx0 = np.gradient(h, spacing)
    spawn_weight = np.clip(0.15 + np.hypot(gx0, gy0) * 1.5, 0.0, 1.0)
    done = 0
    t0 = time.time()
    while done < n_droplets:
        b = min(batch, n_droplets - done)
        done += b
        # Spawn droplets preferentially on high, steep ground, where erosion shapes the landscape most.
        cand = 4 * b
        cx = rng.uniform(max(si0, 1), min(si1, nx - 2), cand)
        cy = rng.uniform(max(sj0, 1), min(sj1, ny - 2), cand)
        ci = cy.astype(np.int64) * nx + cx.astype(np.int64)
        wgt = spawn_weight.ravel()[ci]
        keep = rng.uniform(0, 1, cand) < wgt
        px = cx[keep][:b]
        py = cy[keep][:b]
        b = px.size
        dx = np.zeros(b)
        dy = np.zeros(b)
        vel = np.ones(b)
        water = np.ones(b)
        sed = np.zeros(b)
        alive = np.ones(b, dtype=bool)
        for _ in range(steps):
            idx = np.nonzero(alive)[0]
            if idx.size == 0:
                break
            x = px[idx]
            y = py[idx]
            ix = x.astype(np.int64)
            iy = y.astype(np.int64)
            fx = x - ix
            fy = y - iy
            i00 = iy * nx + ix
            hf = h.ravel()
            h00 = hf[i00]
            h10 = hf[i00 + 1]
            h01 = hf[i00 + nx]
            h11 = hf[i00 + nx + 1]
            gx = (h10 - h00) * (1 - fy) + (h11 - h01) * fy
            gy = (h01 - h00) * (1 - fx) + (h11 - h10) * fx
            hcur = (h00 * (1 - fx) + h10 * fx) * (1 - fy) + (h01 * (1 - fx) + h11 * fx) * fy
            ndx = dx[idx] * inertia - gx * (1 - inertia)
            ndy = dy[idx] * inertia - gy * (1 - inertia)
            ln = np.hypot(ndx, ndy)
            ok = ln > 1e-9
            ndx = np.where(ok, ndx / np.maximum(ln, 1e-12), 0.0)
            ndy = np.where(ok, ndy / np.maximum(ln, 1e-12), 0.0)
            nxp = x + ndx
            nyp = y + ndy
            inside = ok & (nxp > 1) & (nxp < nx - 2) & (nyp > 1) & (nyp < ny - 2)
            nix = np.clip(nxp.astype(np.int64), 0, nx - 2)
            niy = np.clip(nyp.astype(np.int64), 0, ny - 2)
            nfx = nxp - nix
            nfy = nyp - niy
            j00 = niy * nx + nix
            hnew = (hf[j00] * (1 - nfx) + hf[j00 + 1] * nfx) * (1 - nfy) + \
                   (hf[j00 + nx] * (1 - nfx) + hf[j00 + nx + 1] * nfx) * nfy
            dh = np.where(inside, hnew - hcur, 0.0)
            v = vel[idx]
            w = water[idx]
            s = sed[idx]
            cap = np.maximum(-dh * v * w * capacity_k, min_slope)
            deposit_mask = (s > cap) | (dh > 0)
            dep = np.where(dh > 0, np.minimum(dh, s), (s - cap) * deposit_k)
            dep = np.where(deposit_mask, dep, 0.0)
            ero = np.where(deposit_mask, 0.0, np.minimum((cap - s) * erode_k, np.minimum(-dh, max_erode)))
            ero = np.where(inside, ero, 0.0)
            # Deposit at the four corners of the current cell, bilinearly weighted; erode with a wider brush so
            # channels form instead of pits. Many droplets move at once, so the net change per step is clamped.
            wts = ((1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy)
            offs = (0, 1, nx, nx + 1)
            dpos = np.zeros(h.size)
            dneg = np.zeros(h.size)
            for wt, of in zip(wts, offs):
                dpos += np.bincount(i00 + of, weights=dep * wt, minlength=h.size)
                dneg += np.bincount(i00 + of, weights=ero * wt, minlength=h.size)
            dneg = _box_blur(dneg.reshape(h.shape), 1).ravel()
            delta = np.clip(dpos - dneg, -0.2, 0.2)
            h += delta.reshape(h.shape)
            flow += np.bincount(i00, weights=w, minlength=h.size)
            sed[idx] = s + ero - dep
            vel[idx] = np.sqrt(np.maximum(v * v + dh * gravity / spacing, 0.0))
            water[idx] = w * (1 - evap)
            dx[idx] = ndx
            dy[idx] = ndy
            px[idx] = nxp
            py[idx] = nyp
            alive[idx] = inside
        if not quiet:
            _log(f"erosion {done}/{n_droplets} droplets ({time.time() - t0:.0f}s)")
    return h, flow.reshape(h.shape)


def droplet_erosion_tiled(h, spacing, n_droplets, steps, seed, window=512, margin=96, batch=100000):
    """droplet_erosion over a large grid, one window at a time.

    Each window's droplets spawn in its core and run on the core plus a margin wider than a droplet can travel
    (`steps` cells), and the window is written back before the next one starts: the result is a sequential erosion
    in window order, without seams, at a cost that scales with the window rather than the whole grid."""
    h = h.astype(np.float64).copy()
    ny, nx = h.shape
    flow = np.zeros_like(h)
    assert margin > steps
    t0 = time.time()
    wins = [(i0, j0) for j0 in range(0, ny, window) for i0 in range(0, nx, window)]
    for k, (i0, j0) in enumerate(wins):
        i1, j1 = min(i0 + window, nx), min(j0 + window, ny)
        a0, b0 = max(i0 - margin, 0), max(j0 - margin, 0)
        a1, b1 = min(i1 + margin, nx), min(j1 + margin, ny)
        n = int(round(n_droplets * (i1 - i0) * (j1 - j0) / (nx * ny)))
        sub, fl = droplet_erosion(h[b0:b1, a0:a1], spacing, n, steps, seed + k, batch,
                                  spawn=(i0 - a0, i1 - a0, j0 - b0, j1 - b0), quiet=True)
        h[b0:b1, a0:a1] = sub
        flow[b0:b1, a0:a1] += fl
        if k % 10 == 9 or k == len(wins) - 1:
            _log(f"erosion window {k + 1}/{len(wins)} ({time.time() - t0:.0f}s)")
    return h, flow


def thermal_erosion(h, spacing, talus_deg, iters, rate=0.25):
    """Moves material down slopes steeper than the talus angle (scree)."""
    h = h.copy()
    talus = np.tan(np.radians(talus_deg)) * spacing
    for _ in range(iters):
        delta = np.zeros_like(h)
        for sy, sx in ((0, 1), (1, 0)):
            a = h[: h.shape[0] - sy, : h.shape[1] - sx]
            b = h[sy:, sx:]
            d = a - b
            move = np.where(np.abs(d) > talus, (np.abs(d) - talus) * 0.5 * rate * np.sign(d), 0.0)
            delta[: h.shape[0] - sy, : h.shape[1] - sx] -= move
            delta[sy:, sx:] += move
        h += delta
    return h


# ---------------------------------------------------------------------------------------------------------------------
# The world
# ---------------------------------------------------------------------------------------------------------------------

class World:
    """The heightfield and biome fields for a configuration, built once and cached on disk."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.size = cfg.world_size
        self.x0 = -self.size / 2
        self.spacing = float(cfg.c.terrain.spacing)
        self.n = int(round(self.size / self.spacing))  # quads across the world
        # Two cache levels: the eroded macro grid depends on the terrain section only (slow to build); the final
        # heightfield adds fine detail and the paths (fast), and the biome fields add points of interest and biomes.
        self.erosion_dir = os.path.join(cfg.cache, f"erosion_{cfg.fingerprint('terrain')}")
        key = cfg.fingerprint("terrain", "paths", "pois")
        self.cache_dir = os.path.join(cfg.cache, f"terrain_{key}")
        os.makedirs(self.cache_dir, exist_ok=True)
        hpath = os.path.join(self.cache_dir, "height.npy")
        fpath = os.path.join(self.cache_dir, f"fields_{cfg.fingerprint('pois', 'paths', 'biomes', 'markers')}.npz")
        if not os.path.exists(hpath):
            self._build(hpath)
        self.H = np.load(hpath, mmap_mode="r")
        if not os.path.exists(fpath):
            self._build_fields(fpath)
        f = np.load(fpath)
        self.fields = {k: f[k] for k in f.files}

    # -- build -------------------------------------------------------------------------------------------------------

    def _eroded(self):
        """The macro layout on the erosion grid, eroded (cached). Returns (height, mountain mask)."""
        cfg = self.cfg
        t = cfg.c.terrain
        d = self.erosion_dir
        hp, mp = os.path.join(d, "eroded.npy"), os.path.join(d, "mountain_erosion_grid.npy")
        if os.path.exists(hp) and os.path.exists(mp):
            return np.load(hp), np.load(mp).astype(np.float64)
        os.makedirs(d, exist_ok=True)
        es = float(t.erosion_spacing)
        ne = int(round(self.size / es))
        _log(f"macro layout on {ne + 1}^2 grid at {es} m")
        t0 = time.time()
        xs = self.x0 + np.arange(ne + 1) * es
        rows = []
        mrows = []
        for j0 in range(0, ne + 1, 250):
            yy = self.x0 + np.arange(j0, min(j0 + 250, ne + 1)) * es
            X, Y = np.meshgrid(xs, yy)
            h, m = macro_height(cfg, X, Y)
            rows.append(h)
            mrows.append(m)
        h = np.concatenate(rows, axis=0)
        mnorm = np.concatenate(mrows, axis=0)
        n_drop = int(float(t.erosion_droplets_per_km2) * (self.size / 1000.0) ** 2)
        _log(f"macro done ({time.time() - t0:.0f}s); eroding with {n_drop} droplets")
        h0 = h
        h, flow = droplet_erosion_tiled(h, es, n_drop, int(t.erosion_steps), cfg.seed + 5)
        h = h0 + _box_blur(h - h0, 1)
        h = thermal_erosion(h, es, 42.0, 12)
        np.save(os.path.join(d, "flow_erosion_grid.npy"), flow.astype(np.float32))
        np.save(mp, mnorm.astype(np.float32))
        np.save(hp, h)
        _log(f"erosion done ({time.time() - t0:.0f}s)")
        return h, mnorm

    def _build(self, hpath):
        cfg = self.cfg
        t = cfg.c.terrain
        es = float(t.erosion_spacing)
        t0 = time.time()
        h, mnorm = self._eroded()
        # Upsample to the final spacing.
        factor = int(round(es / self.spacing))
        while factor > 1:
            h = _upsample2(h)
            factor //= 2
        assert h.shape[0] == self.n + 1, (h.shape, self.n)
        _log("fine detail and paths")
        out = np.empty(h.shape, dtype=np.float32)
        xs = self.x0 + np.arange(self.n + 1) * self.spacing
        mup = mnorm
        for _ in range(int(round(np.log2(es / self.spacing)))):
            mup = _upsample2(mup)
        for j0 in range(0, self.n + 1, 500):
            j1 = min(j0 + 500, self.n + 1)
            yy = self.x0 + np.arange(j0, j1) * self.spacing
            X, Y = np.meshgrid(xs, yy)
            rockiness = nz.smoothstep(0.2, 0.7, mup[j0:j1])
            detail = 0.35 * nz.fbm(X, Y, cfg.seed + 21, 22.0, 3) + 0.12 * nz.fbm(X, Y, cfg.seed + 22, 5.0, 2)
            detail += rockiness * 1.6 * (nz.ridged(X, Y, cfg.seed + 23, 14.0, 3) - 0.5)
            out[j0:j1] = (h[j0:j1] + detail).astype(np.float32)
        del h, mup
        h = out
        # Settlements and built sites level their ground: blend towards a percentile of the height under the site
        # (a pad that keeps some of the natural slope), fading out over half a radius beyond it.
        for p in cfg.c.get("pois", []):
            strength = LEVEL_GROUND.get(p["kind"])
            if not strength:
                continue
            r = float(p["radius"])
            cx, cy = p["position"]
            i0, j0 = [max(0, int((v - 1.6 * r - self.x0) / self.spacing)) for v in (cx, cy)]
            i1, j1 = [min(self.n, int((v + 1.6 * r - self.x0) / self.spacing) + 1) for v in (cx, cy)]
            X, Y = np.meshgrid(self.x0 + np.arange(i0, i1 + 1) * self.spacing,
                               self.x0 + np.arange(j0, j1 + 1) * self.spacing)
            xw, yw = nz.warp(X, Y, cfg.seed + 62, 40.0, 0.15 * r)
            d = np.hypot(xw - cx, yw - cy)
            sub = h[j0:j1 + 1, i0:i1 + 1]
            target = float(np.percentile(sub[d < r * 0.8], 60))
            wgt = nz.smoothstep(r * 1.5, r * 0.8, d) * strength
            h[j0:j1 + 1, i0:i1 + 1] = sub + (target - sub) * wgt
        # Paths: blend towards a smoothed height and sink slightly, within each segment's bounding box only.
        smooth = _box_blur(h, 7)
        for p in cfg.c.get("paths", []):
            pts = np.asarray(p["points"], dtype=np.float64)
            w = float(p["width"])
            for a, b in zip(pts[:-1], pts[1:]):
                lo = np.minimum(a, b) - (w * 3 + 4)
                hi = np.maximum(a, b) + (w * 3 + 4)
                i0, j0 = [max(0, int((v - self.x0) / self.spacing)) for v in lo]
                i1, j1 = [min(self.n, int((v - self.x0) / self.spacing) + 1) for v in hi]
                X, Y = np.meshgrid(self.x0 + np.arange(i0, i1 + 1) * self.spacing,
                                   self.x0 + np.arange(j0, j1 + 1) * self.spacing)
                xw, yw = nz.warp(X, Y, cfg.seed + 61, 60.0, 6.0)
                d, _ = polyline_distance(xw, yw, [a, b])
                wgt = 1.0 - nz.smoothstep(w * 0.5, w * 2.5, d)
                sink = 0.18 * (1.0 - nz.smoothstep(0.0, w * 0.6, d))
                sub = h[j0:j1 + 1, i0:i1 + 1]
                tgt = smooth[j0:j1 + 1, i0:i1 + 1] - sink
                h[j0:j1 + 1, i0:i1 + 1] = np.where(wgt > 0, sub + (tgt - sub) * wgt * 0.85, sub)
        np.save(hpath, h.astype(np.float32))
        _log(f"heightfield {h.shape} range {h.min():.1f}..{h.max():.1f} m ({time.time() - t0:.0f}s)")

    def _build_fields(self, fpath):
        cfg = self.cfg
        s = cfg.seed
        k = int(round(FIELD_SPACING / self.spacing))
        Hc = np.asarray(self.H[::k, ::k], dtype=np.float64)
        n = Hc.shape[0]
        xs = self.x0 + np.arange(n) * FIELD_SPACING
        X, Y = np.meshgrid(xs, xs)
        slope = slope_deg(_box_blur(Hc, 1), FIELD_SPACING)
        es = float(cfg.c.terrain.erosion_spacing)
        flow = np.load(os.path.join(self.erosion_dir, "flow_erosion_grid.npy"))
        mount = np.load(os.path.join(self.erosion_dir, "mountain_erosion_grid.npy"))
        ke = int(round(FIELD_SPACING / es))
        flow = flow[::ke, ::ke][:n, :n]
        mount = mount[::ke, ::ke][:n, :n].astype(np.float64)
        wet = np.log1p(_box_blur(flow.astype(np.float64), 2)) / 6.0
        wet = np.clip(wet, 0, 1.5)
        vd = np.full_like(X, 1e9)
        for v in cfg.c.terrain.valleys:
            d, _ = polyline_distance(X, Y, v["points"])
            vd = np.minimum(vd, d)
        moisture = 0.5 + 0.35 * nz.fbm(X, Y, s + 71, 800.0, 3) + 0.35 * (1 - nz.smoothstep(40, 500, vd)) + 0.25 * wet
        moisture = np.clip(moisture, 0, 1)
        height = Hc
        treeline = 520.0 + 60.0 * nz.fbm(X, Y, s + 72, 400.0, 2)
        alpine = nz.smoothstep(treeline - 60, treeline + 40, height)
        # Forest: large-scale woodland patches, broken into groves near their edges, with natural clearings.
        f = 0.55 * nz.fbm(X, Y, s + 73, 1300.0, 3) + 0.35 * nz.fbm(X, Y, s + 74, 420.0, 3) \
            + 0.18 * nz.fbm(X, Y, s + 75, 120.0, 2)
        f += 0.25 * (moisture - 0.5) + 0.18 * nz.smoothstep(0.05, 0.4, mount) * (1 - alpine)
        forest = nz.smoothstep(-0.05, 0.22, f)
        clearing = nz.smoothstep(0.42, 0.55, nz.fbm(X, Y, s + 76, 160.0, 2))
        forest *= 1.0 - 0.95 * clearing
        # Designed open land (meadows, grazing land), farmland and woodland from the layout.
        farm = np.zeros_like(X)
        for area in cfg.c.get("biomes", {}).get("areas", []):
            xw2, yw2 = nz.warp(X, Y, s + 81, 180.0, 0.35 * float(area["radius"]))
            d = np.hypot(xw2 - area["center"][0], yw2 - area["center"][1])
            w = nz.smoothstep(float(area["radius"]), float(area["radius"]) * 0.45, d) * float(area["strength"])
            if area["kind"] == "open":
                forest *= 1.0 - np.clip(w, 0, 1)
            elif area["kind"] == "farm":
                farm = np.maximum(farm, np.clip(w, 0, 1))
            else:
                forest = forest + (1.0 - forest) * np.clip(w, 0, 1) * (1 - alpine)
        forest *= 1.0 - nz.smoothstep(30.0, 42.0, slope)
        forest *= 1.0 - alpine
        # Fields stay on workable ground.
        farm *= (1.0 - nz.smoothstep(10.0, 18.0, slope)) * (1.0 - alpine)
        forest *= 1.0 - farm
        # Conifers take over with altitude and in the mountains.
        conifer = np.clip(nz.smoothstep(150.0, 380.0, height) + 0.35 * nz.fbm(X, Y, s + 77, 500.0, 2)
                          + 0.4 * nz.smoothstep(0.1, 0.5, mount), 0, 1)
        rock = np.clip(nz.smoothstep(28.0, 44.0, slope) + 0.8 * alpine * nz.smoothstep(15.0, 30.0, slope)
                       + 0.25 * nz.smoothstep(0.2, 0.7, mount) * nz.smoothstep(0.3, 0.6, nz.fbm(X, Y, s + 78, 90.0, 3)),
                       0, 1)
        snow = nz.smoothstep(640.0, 760.0, height + 40 * nz.fbm(X, Y, s + 79, 150.0, 3)) * \
            (1 - nz.smoothstep(38.0, 52.0, slope))
        meadow = nz.smoothstep(0.05, 0.45, nz.fbm(X, Y, s + 80, 140.0, 3)) * (1 - forest) * (1 - alpine) * (1 - farm)
        pathd = np.full_like(X, 1e9)
        for p in cfg.c.get("paths", []):
            d, _ = polyline_distance(X, Y, p["points"])
            pathd = np.minimum(pathd, d - float(p["width"]) * 0.5)
        poid = np.full_like(X, 1e9)
        for p in cfg.c.pois:
            poid = np.minimum(poid, np.hypot(X - p["position"][0], Y - p["position"][1]) - float(p["radius"]))
        from .pois import path_markers
        for mx, my, _, _ in path_markers(cfg):
            i0, j0 = [max(0, int((v - 8 - self.x0) / FIELD_SPACING)) for v in (mx, my)]
            sl = np.s_[j0:j0 + 5, i0:i0 + 5]
            poid[sl] = np.minimum(poid[sl], np.hypot(X[sl] - mx, Y[sl] - my) - 2.5)
        np.savez_compressed(fpath, slope=slope.astype(np.float32), moisture=moisture.astype(np.float32),
                            forest=forest.astype(np.float32), conifer=conifer.astype(np.float32),
                            rock=rock.astype(np.float32), snow=snow.astype(np.float32),
                            meadow=meadow.astype(np.float32), alpine=alpine.astype(np.float32),
                            wet=wet.astype(np.float32), mount=mount.astype(np.float32),
                            path=pathd.astype(np.float32), poi=poid.astype(np.float32), farm=farm.astype(np.float32),
                            valley=vd.astype(np.float32))
        _log("fields built")

    # -- queries -----------------------------------------------------------------------------------------------------

    def height(self, x, y):
        return bilinear(self.H, self.x0, self.x0, self.spacing, x, y)

    def field(self, name, x, y):
        return bilinear(self.fields[name], self.x0, self.x0, FIELD_SPACING, x, y)

    def grid(self, x0, y0, quads, step):
        """Height samples of a (quads+1)^2 grid starting at world (x0, y0) with `step` grid samples between points.
        Samples beyond the heightfield (the normals' one-step border at the world edge) repeat the edge sample."""
        i0 = int(round((x0 - self.x0) / self.spacing))
        j0 = int(round((y0 - self.x0) / self.spacing))
        offs = np.arange(quads + 1) * step
        ii = np.clip(i0 + offs, 0, self.H.shape[1] - 1)
        jj = np.clip(j0 + offs, 0, self.H.shape[0] - 1)
        return np.asarray(self.H[np.ix_(jj, ii)], dtype=np.float64)

    def normal(self, x, y, d=1.0):
        hx = self.height(x + d, y) - self.height(x - d, y)
        hy = self.height(x, y + d) - self.height(x, y - d)
        n = np.stack([-hx, -hy, np.full_like(hx, 2 * d)], axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)
