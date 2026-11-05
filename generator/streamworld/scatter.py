"""Biome-driven placement of assets and ground cover.

Placement is a pure function of world position: every category uses a world-anchored jittered grid whose cells are
seeded by hashing their integer coordinates, accepted with a probability from the biome fields. A cell's content
therefore never depends on how the world is partitioned, any tile can be regenerated on its own, and neighbouring
tiles agree exactly along their shared edges.
"""

from dataclasses import dataclass

import numpy as np

from . import noise as nz
from .noise import hash_float

TREE_SPECIES = ["oak", "birch", "pine", "spruce", "deadwood"]
TREE_VARIANTS = {"oak": 3, "birch": 3, "pine": 3, "spruce": 3, "deadwood": 2}
_VARIANT_NAMES = {
    "oak": ["a", "b", "c"], "birch": ["a", "b", "c"], "pine": ["a", "b", "c"], "spruce": ["a", "b", "c"],
    "deadwood": ["a", "b"], "shrub": ["a", "b", "c"], "juniper": ["a", "b"],
    "boulder": ["round", "angular", "flat"], "cliff": ["a", "b", "c"], "pillar": ["a", "b"],
    "deadfall": ["log_a", "log_b", "stump"], "standing_stone": ["a", "b", "c"],
}


@dataclass
class Placement:
    """Native instances of one category: parallel arrays."""
    asset: np.ndarray      # asset names (object array)
    variant: np.ndarray    # variant names
    pos: np.ndarray        # (N, 3) world
    rz: np.ndarray         # degrees
    scale: np.ndarray
    tilt: np.ndarray       # (N, 2) degrees about x, y

    def __len__(self):
        return len(self.pos)

    @staticmethod
    def concat(ps):
        ps = [p for p in ps if p is not None and len(p)]
        if not ps:
            return Placement(np.array([], object), np.array([], object), np.zeros((0, 3)), np.zeros(0), np.zeros(0),
                             np.zeros((0, 2)))
        return Placement(*(np.concatenate([getattr(p, f) for p in ps]) for f in
                           ("asset", "variant", "pos", "rz", "scale", "tilt")))


@dataclass
class Scatter:
    """Point-instanced ground cover."""
    proto: np.ndarray      # prototype names (object array)
    pos: np.ndarray
    orient: np.ndarray     # (N, 4) quaternion (x, y, z, w)
    scale: np.ndarray      # (N,) uniform


GROUND_COVER_MAX_SCALE = 1.25 * 1.4   # largest ground-cover instance scale (ferns in full forest), for bounds

CAT_SEEDS = {"tree": 101, "sapling": 102, "bush": 103, "rock": 104, "mrock": 105, "log": 106, "cliff": 107,
             "scatter": 108}


def _grid_candidates(seed, cat, spacing, x0, y0, x1, y1, world_x0):
    """Jittered world-anchored grid points inside [x0, x1) x [y0, y1)."""
    i0 = int(np.floor((x0 - world_x0) / spacing)) - 1
    i1 = int(np.ceil((x1 - world_x0) / spacing)) + 1
    j0 = int(np.floor((y0 - world_x0) / spacing)) - 1
    j1 = int(np.ceil((y1 - world_x0) / spacing)) + 1
    I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
    I = I.ravel()
    J = J.ravel()
    k = CAT_SEEDS[cat]
    x = world_x0 + (I + hash_float(I, J, seed, k, 1)) * spacing
    y = world_x0 + (J + hash_float(I, J, seed, k, 2)) * spacing
    m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    I, J, x, y = I[m], J[m], x[m], y[m]
    rnd = lambda c: hash_float(I, J, seed, k, c)  # noqa: E731
    return x, y, rnd


def _pick(weights, r):
    """Choose a column index per row of `weights` (N, K) with uniforms r (N,)."""
    c = np.cumsum(weights, axis=1)
    c = c / np.maximum(c[:, -1:], 1e-12)
    return (r[:, None] > c).sum(axis=1).clip(0, weights.shape[1] - 1)


def _variants(asset, r):
    names = _VARIANT_NAMES[asset]
    return np.array(names, dtype=object)[np.minimum((r * len(names)).astype(int), len(names) - 1)]


class Placer:
    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world
        self.seed = cfg.seed
        d = cfg.c.density
        self.k = float(d.scale)
        self.d = d

    def _f(self, x, y):
        w = self.world
        names = ("forest", "conifer", "rock", "snow", "meadow", "moisture", "slope", "path", "poi", "wet", "alpine",
                 "valley", "mount", "farm")
        return {n: w.field(n, x, y) for n in names}

    def _clear(self, f, path_m=2.5, poi_m=2.0):
        """Mask of where placed objects are allowed (not on paths / points of interest)."""
        return (f["path"] > path_m) & (f["poi"] > poi_m)

    # -- trees ------------------------------------------------------------------------------------------------------

    def trees(self, x0, y0, x1, y1, saplings=False):
        cat = "sapling" if saplings else "tree"
        dens = float(self.d.saplings_per_ha if saplings else self.d.trees_per_ha) * self.k / 1e4
        if dens <= 0:
            return Placement.concat([])
        sp = 1.0 / np.sqrt(dens)
        x, y, rnd = _grid_candidates(self.seed, cat, sp, x0, y0, x1, y1, self.world.x0)
        f = self._f(x, y)
        # Groves and glades: tree density varies strongly at the 40-100 m scale, so woods have open, light-filled
        # gaps between stands rather than a uniform cover.
        clump = np.clip(0.55 + 0.75 * nz.fbm(x, y, self.seed + 501, 38.0, 2), 0.0, 1.3) \
            * nz.smoothstep(-0.45, 0.05, nz.fbm(x, y, self.seed + 504, 110.0, 2))
        forest = f["forest"]
        if saplings:
            p = (0.25 * forest + 1.6 * forest * (1 - forest)) * clump
        else:
            p = forest ** 1.2 * clump
            # Lone and grouped trees in open grassland.
            p += 0.012 * (1 - forest) * (1 - f["alpine"]) * (1 - f["farm"]) * \
                nz.smoothstep(0.1, 0.6, nz.fbm(x, y, self.seed + 502, 90.0, 2))
        p *= (1 - f["snow"]) * (1 - nz.smoothstep(0.45, 0.8, f["rock"])) * (1 - nz.smoothstep(34, 42, f["slope"]))
        keep = (rnd(3) < p) & self._clear(f, 3.0, 4.0)
        x, y = x[keep], y[keep]
        f = {k: v[keep] for k, v in f.items()}
        forest = f["forest"]
        r = [rnd(c)[keep] for c in range(4, 12)]
        con = np.clip(f["conifer"] + 0.25 * (r[0] - 0.5), 0, 1)
        wet = np.clip(f["moisture"] - 0.55 + 0.6 * nz.smoothstep(150, 20, f["valley"]), 0, 1)
        w = np.stack([
            (1 - con) * (1 - 0.6 * wet),          # oak
            (1 - con) * (0.35 + 1.2 * wet),       # birch
            con * (1 - f["alpine"]) * (0.5 + 0.8 * (1 - f["moisture"])),  # pine
            con * (0.7 + 0.8 * f["moisture"]),    # spruce
            0.035 + 0.25 * f["alpine"] + 0.08 * (1 - forest),  # deadwood
        ], 1)
        sp_idx = _pick(w, r[1])
        species = np.array(TREE_SPECIES, dtype=object)[sp_idx]
        variant = np.array([_variants(s, rv) for s, rv in zip(species, r[2])], dtype=object) if len(species) else \
            np.array([], object)
        if saplings:
            scale = 0.16 + 0.26 * r[3]
        else:
            age = 0.6 + 0.3 * nz.fbm(x, y, self.seed + 503, 60.0, 2)
            scale = np.clip(age + 0.35 * (r[3] - 0.5), 0.55, 1.3)
            scale = np.where(forest < 0.3, scale * 1.12, scale)  # open-grown trees are larger
        z = self.world.height(x, y) - 0.12
        tilt = (np.stack([r[4], r[5]], 1) - 0.5) * 5.0
        return Placement(species, variant, np.c_[x, y, z], r[6] * 360.0, scale, tilt)

    # -- bushes -----------------------------------------------------------------------------------------------------

    def bushes(self, x0, y0, x1, y1):
        dens = float(self.d.bushes_per_ha) * self.k / 1e4
        sp = 1.0 / np.sqrt(dens)
        x, y, rnd = _grid_candidates(self.seed, "bush", sp, x0, y0, x1, y1, self.world.x0)
        f = self._f(x, y)
        forest = f["forest"]
        edge = forest * (1 - forest) * 4
        clump = nz.smoothstep(-0.2, 0.5, nz.fbm(x, y, self.seed + 511, 25.0, 2))
        p = (0.35 * forest + 0.7 * edge + 0.06 * f["meadow"] + 0.15 * f["alpine"] * (1 - f["snow"])) * clump
        p *= 1 - 0.9 * f["farm"]
        p *= (1 - f["snow"]) * (1 - nz.smoothstep(0.5, 0.9, f["rock"]))
        keep = (rnd(3) < p) & self._clear(f)
        x, y = x[keep], y[keep]
        f = {k: v[keep] for k, v in f.items()}
        r = [rnd(c)[keep] for c in range(4, 10)]
        jun = np.clip(0.2 + 0.6 * f["conifer"] + 0.5 * f["alpine"] - 0.3 * f["moisture"], 0.05, 0.95)
        asset = np.where(r[0] < jun, "juniper", "shrub").astype(object)
        variant = np.array([_variants(a, rv) for a, rv in zip(asset, r[1])], dtype=object) if len(asset) else \
            np.array([], object)
        scale = 0.65 + 0.7 * r[2]
        z = self.world.height(x, y) - 0.05
        tilt = (np.stack([r[3], r[4]], 1) - 0.5) * 8.0
        return Placement(asset, variant, np.c_[x, y, z], r[5] * 360.0, scale, tilt)

    # -- rocks ------------------------------------------------------------------------------------------------------

    def rocks(self, x0, y0, x1, y1):
        out = []
        for cat, base in (("rock", float(self.d.rocks_per_ha)), ("mrock", float(self.d.mountain_rocks_per_ha))):
            dens = base * self.k / 1e4
            sp = 1.0 / np.sqrt(dens)
            x, y, rnd = _grid_candidates(self.seed, cat, sp, x0, y0, x1, y1, self.world.x0)
            f = self._f(x, y)
            if cat == "rock":
                p = (0.08 + 0.45 * f["forest"] + 0.25 * f["meadow"]) * nz.smoothstep(0.0, 0.6, nz.fbm(x, y, self.seed + 521, 70.0, 2) + 0.3)
                p *= 1 - 0.9 * f["farm"]   # cleared from the fields
            else:
                p = np.clip(f["rock"] * 1.2 + 0.4 * f["mount"] * nz.smoothstep(12, 30, f["slope"]), 0, 1) * (1 - 0.7 * f["snow"])
            keep = (rnd(3) < p) & self._clear(f, 2.0, 1.0)
            x, y = x[keep], y[keep]
            f = {k: v[keep] for k, v in f.items()}
            r = [rnd(c)[keep] for c in range(4, 10)]
            # Log-normal sizes: many small stones, few large boulders; larger on steep mountain ground.
            big = 0.25 if cat == "rock" else 0.55 + 0.02 * np.clip(f["slope"] - 20, 0, 25)
            scale = np.exp(np.log(0.35 + big) + 0.75 * nz.fbm(x, y, self.seed + 522, 9.0, 1) + 0.6 * (r[0] - 0.5))
            scale = np.clip(scale, 0.12, 3.2)
            variant = _variants("boulder", r[1])
            z = self.world.height(x, y) - 0.15 * scale
            tilt = (np.stack([r[2], r[3]], 1) - 0.5) * 30.0
            out.append(Placement(np.full(len(x), "boulder", object), variant, np.c_[x, y, z], r[4] * 360.0, scale, tilt))
        # Cliff blocks on steep ground, turned so their long side follows the contour.
        dens = 4.0 * self.k / 1e4
        sp = 1.0 / np.sqrt(dens)
        x, y, rnd = _grid_candidates(self.seed, "cliff", sp, x0, y0, x1, y1, self.world.x0)
        f = self._f(x, y)
        p = nz.smoothstep(30.0, 45.0, f["slope"]) * (0.4 + 0.6 * f["mount"])
        keep = (rnd(3) < p) & self._clear(f, 4.0, 6.0)
        x, y = x[keep], y[keep]
        if len(x):
            r = [rnd(c)[keep] for c in range(4, 8)]
            n = self.world.normal(x, y, 3.0)
            rz = np.degrees(np.arctan2(n[:, 1], n[:, 0])) + 90.0
            scale = 0.6 + 0.8 * r[0]
            z = self.world.height(x, y) - 1.2 * scale
            out.append(Placement(np.full(len(x), "cliff", object), _variants("cliff", r[1]), np.c_[x, y, z], rz, scale,
                                 (np.stack([r[2], r[3]], 1) - 0.5) * 8.0))
        return Placement.concat(out)

    def logs(self, x0, y0, x1, y1):
        dens = float(self.d.logs_per_ha) * self.k / 1e4
        sp = 1.0 / np.sqrt(dens)
        x, y, rnd = _grid_candidates(self.seed, "log", sp, x0, y0, x1, y1, self.world.x0)
        f = self._f(x, y)
        p = f["forest"] ** 2 * (1 - nz.smoothstep(25, 35, f["slope"]))
        keep = (rnd(3) < p) & self._clear(f, 3.0, 3.0)
        x, y = x[keep], y[keep]
        r = [rnd(c)[keep] for c in range(4, 9)]
        variant = _variants("deadfall", r[0])
        scale = 0.8 + 0.4 * r[1]
        z = self.world.height(x, y) - 0.1
        # Logs lie along the slope; give them the terrain tilt.
        n = self.world.normal(x, y, 2.0) if len(x) else np.zeros((0, 3))
        tilt = np.stack([np.degrees(np.arcsin(np.clip(n[:, 1], -1, 1))) * -1, np.degrees(np.arcsin(np.clip(n[:, 0], -1, 1)))], 1) \
            if len(x) else np.zeros((0, 2))
        return Placement(np.full(len(x), "deadfall", object), variant, np.c_[x, y, z], r[2] * 360.0, scale, tilt)

    # -- ground cover -----------------------------------------------------------------------------------------------

    PROTOS = ["grass_a", "grass_b", "grass_dry", "flowers_yellow", "flowers_white", "flowers_purple", "flowers_red",
              "clover", "reeds", "fern", "pebbles"]

    def ground_cover(self, x0, y0, x1, y1):
        dens = float(self.d.scatter_per_m2) * self.k
        sp = 1.0 / np.sqrt(dens)
        x, y, rnd = _grid_candidates(self.seed, "scatter", sp, x0, y0, x1, y1, self.world.x0)
        f = self._f(x, y)
        forest = f["forest"]
        open_ = (1 - forest) * (1 - f["rock"]) * (1 - f["snow"])
        patchy = nz.smoothstep(-0.35, 0.35, nz.fbm(x, y, self.seed + 531, 14.0, 3))
        p = open_ * (0.35 + 0.65 * patchy) * (1 - 0.6 * f["alpine"]) + forest * 0.55 * patchy + 0.12 * f["rock"] * (1 - f["snow"])
        p *= np.where(f["path"] < 0.0, 0.0, np.where(f["path"] < 1.2, 0.35, 1.0))
        p *= 1 - 0.95 * f["farm"]     # crops and ploughland: the parcels are in the terrain texture
        keep = rnd(3) < p
        x, y = x[keep], y[keep]
        f = {k: v[keep] for k, v in f.items()}
        r = [rnd(c)[keep] for c in range(4, 12)]
        forest = f["forest"]
        open_ = (1 - forest) * (1 - f["rock"])
        dry = np.clip(1 - f["moisture"] + 0.2 * f["alpine"], 0, 1)
        flw = f["meadow"] * nz.smoothstep(0.0, 0.5, nz.fbm(x, y, self.seed + 532, 30.0, 2))
        fc = nz.fbm(x, y, self.seed + 533, 45.0, 1)
        reedy = nz.smoothstep(60, 10, f["valley"]) * f["moisture"] + 0.5 * nz.smoothstep(0.6, 1.2, f["wet"])
        reedy *= (1 - f["farm"]) * nz.smoothstep(0.0, 40.0, f["poi"])   # grazed and trodden around settlements
        w = np.stack([
            open_ * (1.0 - 0.5 * dry) + 0.15 * forest,      # grass_a
            open_ * 0.5 * (0.4 + f["moisture"]),            # grass_b
            open_ * dry * 0.8 + 0.3 * f["alpine"],          # grass_dry
            flw * np.clip(0.5 + fc, 0, 1) * 0.5,            # flowers (colours by patch)
            flw * np.clip(0.5 - fc, 0, 1) * 0.35,
            flw * nz.smoothstep(0.2, 0.6, fc) * 0.3,
            flw * nz.smoothstep(0.2, 0.6, -fc) * 0.2,
            0.15 * open_ + 0.35 * forest * (1 - f["conifer"]),  # clover
            reedy * 0.8,                                     # reeds
            forest * (0.7 + 0.5 * f["moisture"]) * 1.3,      # fern
            0.02 + 0.12 * f["rock"],                         # pebbles
        ], 1)
        idx = _pick(w, r[0])
        proto = np.array(self.PROTOS, dtype=object)[idx]
        z = self.world.height(x, y) - 0.03
        n = self.world.normal(x, y, 1.0) if len(x) else np.zeros((0, 3))
        # Orientation: random yaw, tilted half-way towards the terrain normal (pebbles fully).
        yaw = r[1] * 2 * np.pi
        follow = np.where(proto == "pebbles", 1.0, 0.45)[:, None]
        up = np.array([0, 0, 1.0]) * (1 - follow) + n * follow
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        q = _quat_yaw_then_up(yaw, up)
        scale = (0.75 + 0.5 * r[2]) * np.where(proto == "fern", 0.8 + 0.6 * forest, 1.0)   # <= GROUND_COVER_MAX_SCALE
        return Scatter(proto, np.c_[x, y, z], q, scale)


def _quat_yaw_then_up(yaw, up):
    """Quaternion (x, y, z, w): rotate about z by yaw, then tilt z onto `up`."""
    qy = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2), np.cos(yaw / 2)], 1)
    z = np.array([0, 0, 1.0])
    axis = np.cross(np.broadcast_to(z, up.shape), up)
    s = np.linalg.norm(axis, axis=1, keepdims=True)
    c = np.clip(up[:, 2:3], -1, 1)
    ang = np.arctan2(s, c)
    axis = np.where(s > 1e-9, axis / np.maximum(s, 1e-12), np.array([1.0, 0, 0]))
    qt = np.concatenate([axis * np.sin(ang / 2), np.cos(ang / 2)], 1)
    return _qmul(qt, qy)


def _qmul(a, b):
    ax, ay, az, aw = a.T
    bx, by, bz, bw = b.T
    return np.stack([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz], 1)
