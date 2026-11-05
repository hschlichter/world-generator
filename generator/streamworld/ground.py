"""Ground LOD inside a tile's full level ([terrain.ground_lod]).

A tile's 1 m terrain is split into chunks; each chunk is a LOD root with two levels:
    fine   0.5 m mesh: Catmull-Rom interpolation of the 1 m heightfield plus synthesized micro-relief, coloured by a
           per-tile texture sampled from the terrain source textures (payload into the tile's ground file)
    base   the 1 m mesh (inline in the tile's full level)

Chunks are Cells of a pseudo level below the finest partition level, so ids, names, centres and bounds work as for
the partition cells; they are not partition cells (cfg.children of a tile stays empty).
"""

import os
import time

import numpy as np

from . import noise as nz
from . import textures
from .config import Cell, CellLevel
from .imgutil import linear_to_srgb, write_image

RELIEF_SEED = 7000   # same offset as the terrain bake, so ridge-and-furrow follows the painted farm strips


def ground_cfg(cfg):
    return cfg.c.terrain.ground_lod


def ground_level(cfg):
    """The chunk level: a CellLevel below the finest partition level, with the fine level as its full level."""
    g = ground_cfg(cfg)
    tile = cfg.levels[-1]
    size = float(g.chunk)
    n = tile.size / size
    if abs(n - round(n)) > 1e-6 or abs(size / float(cfg.c.terrain.spacing) - round(size / float(cfg.c.terrain.spacing))) > 1e-6:
        raise ValueError(f"terrain.ground_lod.chunk {size} must divide the tile size and be a multiple of the spacing")
    if float(g.streaming_unload_thresholds[-1]) > tile.streaming_distance_thresholds[0] or \
            float(g.distance_thresholds[-1]) >= tile.distance_thresholds[0]:
        raise ValueError("terrain.ground_lod thresholds must lie inside the tile's first boundary")
    lv = CellLevel(index=len(cfg.levels), name="ground", prefix="G", size=size, coarse=[{"name": "base"}],
                   distance_thresholds=list(g.distance_thresholds), distance_blend=list(g.get("distance_blend", [])),
                   screen_thresholds=[], screen_blend=[], physics_distance_thresholds=[],
                   streaming_distance_thresholds=list(g.streaming_distance_thresholds),
                   streaming_unload_thresholds=list(g.streaming_unload_thresholds), full_arc="payload",
                   file_format_full="usdc", subdivisions=0)
    lv.subdivisions_parent = int(round(n))
    return lv


GROUND_LOD_NAMES = ["fine", "base"]


def chunks(cfg, tile):
    lv = ground_level(cfg)
    n = lv.subdivisions_parent
    return [Cell(lv, tile.ix * n + i, tile.iy * n + j, tile, cfg.world_size) for j in range(n) for i in range(n)]


def ground_path(cfg, tile):
    """The tile's ground file (fine level of every chunk)."""
    from .cells import cell_level_path
    return cell_level_path(cfg, tile, "full").replace("_full.usdc", "_ground.usdc")


def fine_texture_path(tile):
    """Per-tile fine terrain colour, relative to world/."""
    from .atlas import TEXTURE_DIR
    return f"{TEXTURE_DIR}/regions/{tile.chain[0].name}/{tile.name}_fine.jpg"


# ---------------------------------------------------------------------------------------------------------------------
# Heights
# ---------------------------------------------------------------------------------------------------------------------

def catmull_rom(world, X, Y):
    """Catmull-Rom (bicubic) interpolation of the heightfield at world (X, Y); equals the samples on the grid."""
    H = world.H
    sp = world.spacing
    fx = (np.asarray(X, dtype=np.float64) - world.x0) / sp
    fy = (np.asarray(Y, dtype=np.float64) - world.x0) / sp
    ix = np.floor(fx).astype(np.int64)
    iy = np.floor(fy).astype(np.int64)
    tx = fx - ix
    ty = fy - iy

    def w(t):
        t2, t3 = t * t, t * t * t
        return [(-t3 + 2 * t2 - t) / 2, (3 * t3 - 5 * t2 + 2) / 2, (-3 * t3 + 4 * t2 + t) / 2, (t3 - t2) / 2]
    wx, wy = w(tx), w(ty)
    ny, nx = H.shape
    out = np.zeros(np.shape(fx))
    for b in range(4):
        jj = np.clip(iy + b - 1, 0, ny - 1)
        row = np.zeros(np.shape(fx))
        for a in range(4):
            row += wx[a] * H[jj, np.clip(ix + a - 1, 0, nx - 1)]
        out += wy[b] * row
    return out


def micro_relief(world, X, Y):
    """Synthesized relief below the heightfield's 1 m resolution (metres, roughly -0.3..0.4), from world-space noise
    and the biome fields, so it is continuous across chunks and tiles."""
    cfg = world.cfg
    seed = cfg.seed + RELIEF_SEED
    f = {n: world.field(n, X, Y) for n in ("forest", "rock", "path", "poi", "wet", "farm", "snow", "meadow")}
    d = 1.0
    hx = world.height(X + d, Y) - world.height(X - d, Y)
    hy = world.height(X, Y + d) - world.height(X, Y - d)
    slope = np.degrees(np.arctan(np.hypot(hx, hy) / (2 * d)))
    ss = nz.smoothstep
    # Clods and tussocks everywhere, larger lumps in meadows.
    soft = 0.045 * nz.fbm(X, Y, seed + 21, 2.4, 2) + 0.06 * nz.fbm(X, Y, seed + 22, 7.0, 2)
    soft += 0.05 * f["meadow"] * ss(0.15, 0.6, nz.fbm(X, Y, seed + 23, 1.6, 1))
    # Forest floor: hummocks and root mounds.
    soft += 0.13 * np.clip(f["forest"], 0, 1) * ss(0.05, 0.65, nz.fbm(X, Y, seed + 24, 4.5, 2))
    # Trodden paths and settlement ground are smoother; wet ground and snow are smoothed.
    path = 1.0 - ss(-0.5, 1.0, f["path"])
    soft *= (1 - 0.85 * path) * (0.35 + 0.65 * ss(0.0, 20.0, f["poi"])) * (1 - 0.5 * np.clip(f["wet"], 0, 1)) \
        * (1 - 0.7 * np.clip(f["snow"], 0, 1))
    soft -= 0.04 * path
    # Rock: sharp ridged detail on cliffs and rocky ground.
    rock_w = np.clip(np.maximum(f["rock"] * 0.8, ss(30.0, 44.0, slope)), 0, 1)
    rock = rock_w * 0.5 * (nz.ridged(X, Y, seed + 25, 3.2, 2) - 0.45)
    # Ridge-and-furrow: each ploughland strip is ploughed towards its middle, ending at the headland.
    farm = np.clip(f["farm"], 0, 1)
    ridge = np.zeros(np.shape(X))
    if farm.max() > 0:
        wsum = np.zeros(np.shape(X)) + 1e-9
        for fs in textures.farm_strips(cfg, X, Y, seed):
            t = (fs["u"] - fs["su"] * fs["sw"]) / fs["sw"]
            amp = np.where((fs["kind"] < 0.38) | ((fs["kind"] >= 0.62) & (fs["kind"] < 0.85)), 0.2, 0.1)
            ridge += amp * (0.5 - 0.5 * np.cos(2 * np.pi * t)) * ss(79.0, 72.0, fs["ev"]) * fs["wt"]
            wsum += fs["wt"]
        ridge = (ridge / wsum - 0.06) * farm * (1 - rock_w) * (1 - path) * ss(-2.0, 6.0, f["poi"])
    return float(ground_cfg(cfg).micro_relief) * (soft * (1 - rock_w) + rock + ridge)


def fine_height(world, X, Y):
    return catmull_rom(world, X, Y) + micro_relief(world, X, Y)


def fine_grid(world, x0, y0, n, step):
    """Fine heights of an (n+1)^2 grid from world (x0, y0); coordinates are exact multiples of `step`, so
    neighbouring chunks evaluate identical points on their shared edge."""
    xs = x0 + np.arange(n + 1) * step
    ys = y0 + np.arange(n + 1) * step
    X, Y = np.meshgrid(xs, ys)
    return fine_height(world, X, Y)


def fine_edge_gap(world, xa, ya, xb, yb, step, base_step):
    """Max |fine - base| along an axis-aligned edge, the base being linear between its own samples."""
    n = int(round(max(abs(xb - xa), abs(yb - ya)) / step))
    xs = np.linspace(xa, xb, n + 1)
    ys = np.linspace(ya, yb, n + 1)
    hf = fine_height(world, xs, ys)
    k = int(round(base_step / step))
    hb = world.height(xs[::k], ys[::k])
    t = np.arange(n + 1) / k
    return float(np.max(np.abs(np.interp(t, np.arange(len(hb)), hb) - hf)))


# ---------------------------------------------------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------------------------------------------------

def bake_tile_fine(cfg, world, tile, force=False):
    """The tile's fine terrain colour: the region bake's recipe with the source textures sampled in world space."""
    rel = fine_texture_path(tile)
    path = os.path.join(cfg.out, rel)
    if os.path.exists(path) and not force:
        return rel
    t0 = time.time()
    N = int(ground_cfg(cfg).fine_texture_size)
    S = tile.level.size
    px = S / N
    src = textures._source_means(cfg, size=16)
    out = np.zeros((N, N, 3))
    rows = 256
    for r0 in range(0, N, rows):
        r1 = min(r0 + rows, N)
        X = tile.x0 + (np.arange(N)[None, :] + 0.5) * px + np.zeros((r1 - r0, 1))
        Y = tile.y0 + S - (np.arange(r0, r1)[:, None] + 0.5) * px + np.zeros((1, N))
        out[r0:r1] = textures._bake_block(world, src, textures.atlas.TERRAIN_REPEAT_M, X, Y, cfg.seed + 7000,
                                          sampled=True)
    write_image(path, linear_to_srgb(out), quality=90)
    print(f"[ground] baked {tile.id} fine texture ({time.time() - t0:.1f}s)", flush=True)
    return rel
