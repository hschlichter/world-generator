"""Configuration loading and the derived cell partition."""

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass, field


class Cfg(dict):
    """A dict with attribute access, recursively."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    @staticmethod
    def wrap(v):
        if isinstance(v, dict):
            return Cfg({k: Cfg.wrap(x) for k, x in v.items()})
        if isinstance(v, list):
            return [Cfg.wrap(x) for x in v]
        return v


@dataclass
class CellLevel:
    """One level of the streaming hierarchy (e.g. regions, tiles)."""

    index: int
    name: str
    prefix: str
    size: float
    coarse: list
    distance_thresholds: list
    distance_blend: list
    screen_thresholds: list
    screen_blend: list
    physics_distance_thresholds: list
    streaming_distance_thresholds: list   # load radius per level boundary (streaming domain)
    streaming_unload_thresholds: list     # unload radius per level boundary (hysteresis), >= the load radius
    full_arc: str                         # "payload" or "reference": how the full level of a non-finest cell is added
    file_format_full: str
    subdivisions: int  # children per edge in the next level (0 for the finest level)

    @property
    def lod_names(self):
        return ["full"] + [c["name"] for c in self.coarse]

    def far_spec(self):
        """The coarse level that uses the far terrain material (and the region's far textures), if any."""
        far = [c for c in self.coarse if c.get("terrain_material", "far") == "far"]
        return far[0] if far else None


@dataclass
class Cell:
    """A cell of some level, addressed by integer coordinates within that level's grid."""

    level: CellLevel
    ix: int
    iy: int
    parent: "Cell | None" = None
    world_size: float = 0.0

    @property
    def name(self):
        if self.parent is None:
            return f"{self.level.prefix}_{self.ix}_{self.iy}"
        n = self.level.subdivisions_parent
        return f"{self.level.prefix}_{self.ix % n}_{self.iy % n}"

    @property
    def x0(self):
        return -self.world_size / 2 + self.ix * self.level.size

    @property
    def y0(self):
        return -self.world_size / 2 + self.iy * self.level.size

    @property
    def center(self):
        s = self.level.size
        return (self.x0 + s / 2, self.y0 + s / 2)

    @property
    def bounds2d(self):
        s = self.level.size
        return (self.x0, self.y0, self.x0 + s, self.y0 + s)

    @property
    def chain(self):
        c, out = self, []
        while c is not None:
            out.append(c)
            c = c.parent
        return list(reversed(out))

    @property
    def id(self):
        """Globally unique id, e.g. R_2_2/T_1_3."""
        return "/".join(c.name for c in self.chain)

    @property
    def file_stem(self):
        return "_".join(c.name for c in self.chain)


class Config:
    def __init__(self, path, overrides=None):
        self.path = os.path.abspath(path)
        with open(self.path, "rb") as f:
            raw = tomllib.load(f)
        for k, v in (overrides or {}).items():
            node = raw
            keys = k.split(".")
            for kk in keys[:-1]:
                node = node.setdefault(kk, {})
            node[keys[-1]] = v
        self.raw = raw
        self.c = Cfg.wrap(raw)
        # The project root is the parent of config/.
        self.root = os.path.dirname(os.path.dirname(self.path))
        self.world_size = float(self.c.world.size)
        self.seed = int(self.c.world.seed)
        self.out = os.path.join(self.root, self.c.world.output_dir)
        self.cache = os.path.join(self.root, self.c.world.cache_dir)
        self.levels = self._levels()

    def _levels(self):
        lv = []
        specs = self.c.partition.levels
        prev = self.world_size
        for i, s in enumerate(specs):
            size = float(s["size"])
            n = prev / size
            if abs(n - round(n)) > 1e-6:
                raise ValueError(f"cell level {s['name']}: size {size} does not divide {prev}")
            prev = size
            ncoarse = len(s.get("coarse", []))
            for key in ("distance_thresholds", "screen_thresholds", "streaming_distance_thresholds",
                        "streaming_unload_thresholds"):
                if len(s.get(key, [])) != ncoarse:
                    raise ValueError(f"cell level {s['name']}: {key} needs {ncoarse} values")
            for key in ("distance_blend", "screen_blend", "physics_distance_thresholds"):
                if s.get(key) and len(s[key]) != ncoarse:
                    raise ValueError(f"cell level {s['name']}: {key} needs {ncoarse} values")
            th, load, unload = s["distance_thresholds"], s["streaming_distance_thresholds"], s["streaming_unload_thresholds"]
            if any(l < t for l, t in zip(load, th)) or any(u < l for u, l in zip(unload, load)):
                raise ValueError(f"cell level {s['name']}: need distance <= streaming load <= streaming unload "
                                 f"per boundary, got {th} / {load} / {unload}")
            arc = s.get("full_arc", "payload")
            if arc not in ("payload", "reference"):
                raise ValueError(f"cell level {s['name']}: full_arc must be payload or reference")
            lv.append(CellLevel(
                index=i, name=s["name"], prefix=s["prefix"], size=size, coarse=list(s.get("coarse", [])),
                distance_thresholds=list(th), distance_blend=list(s.get("distance_blend", [])),
                screen_thresholds=list(s["screen_thresholds"]), screen_blend=list(s.get("screen_blend", [])),
                physics_distance_thresholds=list(s.get("physics_distance_thresholds", [])),
                streaming_distance_thresholds=list(load), streaming_unload_thresholds=list(unload), full_arc=arc,
                file_format_full=s.get("file_format_full", "usdc"), subdivisions=0))
        for i in range(len(lv) - 1):
            lv[i].subdivisions = int(round(lv[i].size / lv[i + 1].size))
            # A child cell's levels only exist while its parent shows (imaging) or wants (streaming) its full level,
            # so every child boundary must lie inside the parent's first boundary.
            p, c = lv[i], lv[i + 1]
            if c.distance_thresholds and c.distance_thresholds[-1] >= p.distance_thresholds[0]:
                raise ValueError(f"{c.name} distance thresholds must stay below {p.name}'s first ({p.distance_thresholds[0]})")
            if c.streaming_unload_thresholds and c.streaming_unload_thresholds[-1] > p.streaming_distance_thresholds[0]:
                raise ValueError(f"{c.name} streaming unload thresholds must stay within {p.name}'s first load radius "
                                 f"({p.streaming_distance_thresholds[0]})")

        for i, l in enumerate(lv):
            l.subdivisions_parent = int(round(self.world_size / l.size)) if i == 0 else lv[i - 1].subdivisions
        fq = int(self.c.terrain.full_quads)
        if abs(lv[-1].size / fq - float(self.c.terrain.spacing)) > 1e-9:
            raise ValueError("terrain.full_quads * terrain.spacing must equal the finest cell size")
        return lv

    # -- cells -------------------------------------------------------------------------------------------------------

    def top_cells(self):
        l0 = self.levels[0]
        n = int(round(self.world_size / l0.size))
        return [Cell(l0, ix, iy, None, self.world_size) for iy in range(n) for ix in range(n)]

    def children(self, cell):
        if cell.level.index + 1 >= len(self.levels):
            return []
        nl = self.levels[cell.level.index + 1]
        n = cell.level.subdivisions
        return [Cell(nl, cell.ix * n + i, cell.iy * n + j, cell, self.world_size) for j in range(n) for i in range(n)]

    def cell_by_id(self, cid):
        parts = cid.split("/")
        cell = None
        for depth, p in enumerate(parts):
            pool = self.top_cells() if cell is None else self.children(cell)
            match = [c for c in pool if c.name == p]
            if not match:
                raise KeyError(f"no cell {p} in {cid}")
            cell = match[0]
        return cell

    def leaf_cells(self, cell):
        kids = self.children(cell)
        if not kids:
            return [cell]
        out = []
        for k in kids:
            out += self.leaf_cells(k)
        return out

    def fingerprint(self, *sections):
        """Hash of config sections, used to key caches."""
        data = {s: self.raw.get(s) for s in sections}
        if isinstance(data.get("terrain"), dict):
            # The ground LOD only changes how the heightfield is meshed, not the heightfield itself.
            data["terrain"] = {k: v for k, v in data["terrain"].items() if k != "ground_lod"}
        data["seed"] = self.seed
        data["size"] = self.world_size
        return hashlib.sha1(json.dumps(data, sort_keys=True).encode()).hexdigest()[:12]

    def density_scale(self):
        return float(self.c.density.scale)
