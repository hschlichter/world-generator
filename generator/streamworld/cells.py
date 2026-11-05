"""Writes the streaming cells: every cell is a LOD root whose levels live in files of their own.

    root.usda                                   /World/Regions/<R>              LOD root (regions)
    regions/<R>/<R>_full.usda                   reference of /World/Regions/<R>/full: the region's tiles (LOD roots)
    regions/<R>/<R>_medium.usdc                 payload of .../<R>/medium: 5 m terrain, asset instances
    regions/<R>/<R>_proxy.usdc                  payload of .../<R>/proxy: 10 m terrain, the larger asset instances
    regions/<R>/tiles/<R>_<T>_full.usdc         payload of .../<R>/full/<T>/full: ground chunks (LOD roots, 1 m base
                                                level inline) + every placed asset
    regions/<R>/tiles/<R>_<T>_ground.usdc       payloads of .../<T>/full/<G>/fine: 0.5 m ground of every chunk
    regions/<R>/tiles/<R>_<T>_medium.usdc       payload of .../<T>/medium: 2 m terrain + asset instances
    regions/<R>/tiles/<R>_<T>_simplified.usdc   payload of .../<T>/simplified: 5 m terrain + asset instances

Coarse levels point-instance the assets' own LOD levels: each instance shows the level the asset would select at the
distance where the coarse level takes over, so switching a cell's level never swaps an asset for a different shape.

Level names and counts come from [[partition.levels]]; whether a non-finest cell's full level is a payload or a
reference from its `full_arc`. Every cell carries imaging, streaming and (finest cells) physics heuristics.

The code is written for any depth: a cell's `full` level holds its child cells, the finest cells hold content.
Cell prims carry a translate to their centre; payload content is in cell-local coordinates.
"""

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLod, UsdShade, Vt

from . import atlas
from . import ground as gr
from . import usdutil as uu
from .assets import library as L
from .assets.cache import AssetCache
from .meshutil import grid_mesh, skirt
from .pois import structures_in
from .scatter import GROUND_COVER_MAX_SCALE, Placement, Placer

GROUPS = {  # placed-asset group in a tile's full level, by asset
    "oak": "Trees", "birch": "Trees", "pine": "Trees", "spruce": "Trees", "deadwood": "Trees",
    "shrub": "Bushes", "juniper": "Bushes", "boulder": "Rocks", "cliff": "Rocks", "pillar": "Rocks",
    "deadfall": "Deadfall", "standing_stone": "Structures", "wall": "Structures", "tower": "Structures",
    "column": "Structures", "shrine": "Structures", "rock_arch": "Structures",
    "house": "Buildings", "windmill": "Buildings", "keep": "Buildings", "gatehouse": "Buildings",
    "well": "Props", "fence": "Props", "haystack": "Props", "gravestone": "Props", "banner": "Props",
    "wayside_shrine": "Props", "wizard_tower": "Structures", "runestone": "Structures", "crystal": "Structures",
    "dragon_bones": "Structures",
}
# Coarse-level instance filters per placed-asset group: trees and bushes by scaled height, the rest by largest extent.
HEIGHT_FILTER = {"Trees": "trees_min_height", "Bushes": "bushes_min_height"}


STREAMING = "streaming"                         # lod:domain of the streaming heuristics
UNLOAD_ATTR = "streaming:unloadThresholds"      # hysteresis radii, on each streaming heuristic


def _log(msg):
    print(f"[cells] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------------------------------------------------

def cell_level_path(cfg, cell, level_name):
    """File of one LOD level of a cell."""
    chain = cell.chain
    region = chain[0]
    base = os.path.join(cfg.out, "regions", region.name)
    if cell.level.index == 0:
        fmt = cell.level.file_format_full if level_name == "full" else "usdc"
        return os.path.join(base, f"{cell.file_stem}_{level_name}.{fmt}")
    sub = os.path.join(base, *[f"{c.level.name}s" for c in chain[1:]][:1])
    fmt = cell.level.file_format_full if level_name == "full" and cell.level.index < len(cfg.levels) - 1 else "usdc"
    return os.path.join(sub, f"{cell.file_stem}_{level_name}.{fmt}")


def root_path(cfg):
    return os.path.join(cfg.out, "root.usda")


# ---------------------------------------------------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------------------------------------------------

def _region_of(cell):
    return cell.chain[0]


def level_spacing(cfg, cell, level_name):
    """Terrain sample spacing (metres) of a cell level, and the level spec."""
    if level_name == "full":
        if cell.level.index < len(cfg.levels) - 1:
            return None, None
        return cell.level.size / int(cfg.c.terrain.full_quads), None
    spec = next(c for c in cell.level.coarse if c["name"] == level_name)
    return cell.level.size / int(spec["terrain_quads"]), spec


def all_terrain_spacings(cfg):
    out = {float(cfg.c.terrain.spacing)}
    for lv in cfg.levels:
        for c in lv.coarse:
            out.add(lv.size / int(c["terrain_quads"]))
    return sorted(out)


def _edge_deviation(world, xa, ya, xb, yb, step):
    """Max |linear interpolation at `step` - finest heights| along an axis-aligned edge."""
    n = int(round(max(abs(xb - xa), abs(yb - ya)) / world.spacing))
    xs = np.linspace(xa, xb, n + 1)
    ys = np.linspace(ya, yb, n + 1)
    hf = world.height(xs, ys)
    k = int(round(step / world.spacing))
    coarse = hf[::k]
    t = np.arange(n + 1) / k
    hc = np.interp(t, np.arange(len(coarse)), coarse)
    return float(np.max(np.abs(hc - hf)))


def skirt_depth(cfg, world, cell, step):
    """Skirt depth covering the worst gap against any level that can neighbour this edge (documented in the
    manifest; verified by `validate`)."""
    x0, y0, x1, y1 = cell.bounds2d
    edges = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x0, y1, x1, y1), (x0, y0, x0, y1)]
    spacings = all_terrain_spacings(cfg)
    worst = 0.0
    for e in edges:
        own = _edge_deviation(world, *e, step)
        other = max(_edge_deviation(world, *e, s) for s in spacings)
        worst = max(worst, own + other)
    return max(float(cfg.c.terrain.skirt_min), worst + float(cfg.c.terrain.skirt_margin))


def terrain_meshes(cfg, world, cell, step, depth=None, origin=None):
    """Ground and skirt meshes of a cell at a sample spacing, relative to `origin` (default: the cell's centre)."""
    x0, y0, x1, y1 = cell.bounds2d
    cx, cy = cell.center if origin is None else origin
    q = int(round(cell.level.size / step))
    k = int(round(step / world.spacing))
    H = world.grid(x0, y0, q, k)
    reg = _region_of(cell)
    rs = reg.level.size

    def st_fn(x, y):
        return np.stack([(x + cx - reg.x0) / rs, (y + cy - reg.y0) / rs], 1)
    g = grid_mesh(H, x0 - cx, y0 - cy, step, st_fn)
    # Normals: central differences of this level's own grid, from one row/column beyond the cell so shared
    # edges get identical normals in both neighbours.
    Hx = world.grid(x0 - step, y0 - step, q + 2, k)
    gy, gx = np.gradient(Hx, step)
    gx, gy = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
    n = np.stack([-gx.ravel(), -gy.ravel(), np.ones(gx.size)], 1)
    g.normals = n / np.linalg.norm(n, axis=1, keepdims=True)
    # Skirt along the boundary loop (counter-clockwise seen from above, so it faces outward).
    nx = q + 1
    loop = list(range(0, nx)) + [r * nx + q for r in range(1, nx)] + \
        [q * nx + c for c in range(q - 1, -1, -1)] + [r * nx for r in range(q - 1, 0, -1)]
    loop = np.array(loop)
    depth = skirt_depth(cfg, world, cell, step) if depth is None else depth
    sk = skirt(g.points[loop], g.normals[loop], g.st[loop], depth)
    return g, sk, depth


def _terrain_material(stage, comp_path, layer_path, cfg, cell, far):
    reg = _region_of(cell)
    paths = atlas.region_texture_paths(reg.name)
    lib = os.path.join(cfg.out, atlas.MATERIAL_LIBRARY)
    stage.DefinePrim(f"{comp_path}/Materials", "Scope")
    mp = f"{comp_path}/Materials/terrain"
    m = stage.DefinePrim(mp, "Material")
    m.GetReferences().AddReference(uu.rel(layer_path, lib), "/Materials/terrain_far" if far else "/Materials/terrain")
    mat = UsdShade.Material(m)
    if far:
        mat.CreateInput("color_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(uu.rel(layer_path, os.path.join(cfg.out, paths["color_far"]))))
        mat.CreateInput("normal_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(uu.rel(layer_path, os.path.join(cfg.out, paths["normal_far"]))))
    else:
        mat.CreateInput("color_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(uu.rel(layer_path, os.path.join(cfg.out, paths["color"]))))
        rep = reg.level.size / float(cfg.c.terrain.detail_tiling)
        mat.CreateInput("detail_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(rep, rep))
    return mp


def _lib_material(stage, comp_path, layer_path, cfg, name):
    lib = os.path.join(cfg.out, atlas.MATERIAL_LIBRARY)
    if not stage.GetPrimAtPath(f"{comp_path}/Materials"):
        stage.DefinePrim(f"{comp_path}/Materials", "Scope")
    mp = f"{comp_path}/Materials/{name}"
    stage.DefinePrim(mp, "Material").GetReferences().AddReference(uu.rel(layer_path, lib), f"/Materials/{name}")
    return mp


def _quat_xyz(rx, ry, rz):
    """Quaternions (x, y, z, w) of xformOp:rotateXYZ (rotate about x, then y, then z), radians, arrays."""
    def axis(a, k):
        q = np.zeros((len(a), 4))
        q[:, k] = np.sin(a / 2)
        q[:, 3] = np.cos(a / 2)
        return q

    def mul(p, q):  # Hamilton product p * q, (x, y, z, w)
        px, py, pz, pw = p.T
        qx, qy, qz, qw = q.T
        return np.stack([pw * qx + px * qw + py * qz - pz * qy, pw * qy - px * qz + py * qw + pz * qx,
                         pw * qz + px * qy - py * qx + pz * qw, pw * qw - px * qx - py * qy - pz * qz], 1)
    return mul(axis(rz, 2), mul(axis(ry, 1), axis(rx, 0)))


# ---------------------------------------------------------------------------------------------------------------------
# Content of the finest cells
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class TileContent:
    placed: Placement
    ground: object  # Scatter


class Generator:
    def __init__(self, cfg, world):
        self.cfg = cfg
        self.world = world
        self.placer = Placer(cfg, world)
        self.assets = AssetCache(cfg)
        self._content = {}
        self.stats = {}

    # -- placement ---------------------------------------------------------------------------------------------------

    def content(self, cell, ground=True):
        key = (cell.id, ground)
        if key in self._content:
            return self._content[key]
        x0, y0, x1, y1 = cell.bounds2d
        p = self.placer
        placed = Placement.concat([p.trees(x0, y0, x1, y1), p.trees(x0, y0, x1, y1, saplings=True),
                                   p.bushes(x0, y0, x1, y1), p.rocks(x0, y0, x1, y1), p.logs(x0, y0, x1, y1),
                                   structures_in(self.cfg, self.world, x0, y0, x1, y1)])
        gc = p.ground_cover(x0, y0, x1, y1) if ground else None
        c = TileContent(placed, gc)
        self._content[key] = c
        return c

    def _instance_bounds(self, pl):
        """Conservative world-space AABB of every placed instance (asset extents x scale, any rotation)."""
        if not len(pl):
            return None
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        keys = {}
        for i, (a, v) in enumerate(zip(pl.asset, pl.variant)):
            keys.setdefault((a, v), []).append(i)
        for (a, v), idx in keys.items():
            d = self.assets.levels(a, v)
            r = float(np.max(np.hypot(*np.abs(np.r_[[d["lo"][:2]], [d["hi"][:2]]]).T)))
            zl, zh = float(d["lo"][2]), float(d["hi"][2])
            idx = np.array(idx)
            s = pl.scale[idx][:, None]
            p = pl.pos[idx]
            # Bound of the asset's cylinder (radius r, z in [zl, zh]) under a tilt of up to th = |tx| + |ty|.
            th = np.radians(np.minimum(np.abs(pl.tilt[idx]).sum(1), 90.0))[:, None]
            c, sn = np.cos(th), np.sin(th)
            hz = max(abs(zl), abs(zh))
            rxy = r * c + hz * sn
            zmin = np.minimum(zl * c - r * sn, zl)
            zmax = zh * c + r * sn
            lo_i = p + s * np.c_[-rxy, -rxy, zmin]
            hi_i = p + s * np.c_[rxy, rxy, zmax]
            lo = np.minimum(lo, lo_i.min(0))
            hi = np.maximum(hi, hi_i.max(0))
        return lo, hi

    def _leaf_instance_bounds(self, leaf):
        """Instance bounds of a finest cell. Only the bounds are kept: bounding the whole world (root.usda) would
        otherwise hold every tile's placements in memory."""
        key = ("instance_bounds", leaf.id)
        if key not in self._content:
            had = (leaf.id, False) in self._content
            self._content[key] = self._instance_bounds(self.content(leaf, ground=False).placed)
            if not had:
                self._content.pop((leaf.id, False), None)
        return self._content[key]

    def cell_bounds(self, cell):
        """World-space bounds containing a cell's content in every level (exact terrain range at the finest
        spacing, skirts, conservative instance bounds, ground cover), plus a small margin."""
        key = ("bounds", cell.id)
        if key in self._content:
            return self._content[key]
        x0, y0, x1, y1 = cell.bounds2d
        q = int(round(cell.level.size / self.world.spacing))
        H = self.world.grid(x0, y0, q, 1)
        steps = [s for s in all_terrain_spacings(self.cfg) if s <= cell.level.size]
        depth = max(skirt_depth(self.cfg, self.world, cell, s) for s in steps)
        for leaf in self.cfg.leaf_cells(cell):
            depth = max(depth, max(skirt_depth(self.cfg, self.world, leaf, s) for s in steps if s <= leaf.level.size))
            depth = max(depth, max(self.chunk_skirt(c) for c in gr.chunks(self.cfg, leaf)))
        margin = 1.0
        # Ground cover is rooted inside the cell but its largest prototypes (ferns at up to 1.75x) reach about 2 m past
        # the edge: extend by the largest prototype's bounding radius at the largest scale.
        reach = max(self.assets.bound_radius(n) for n in Placer.PROTOS) * GROUND_COVER_MAX_SCALE
        lo = np.array([x0 - reach, y0 - reach, float(H.min()) - depth])
        # Ground cover is at most ~2.5 m tall; the fine ground's micro-relief stays well inside that.
        hi = np.array([x1 + reach, y1 + reach, float(H.max()) + 2.6])
        for leaf in self.cfg.leaf_cells(cell):
            b = self._leaf_instance_bounds(leaf)
            if b is not None:
                lo = np.minimum(lo, b[0])
                hi = np.maximum(hi, b[1])
        lo, hi = lo - margin, hi + margin
        self._content[key] = (lo, hi)
        return lo, hi

    # -- writers -----------------------------------------------------------------------------------------------------

    def write_leaf_full(self, cell):
        cfg = self.cfg
        path = cell_level_path(cfg, cell, "full")
        t0 = time.time()
        cx, cy = cell.center
        root_name = f"{cell.file_stem}_full"
        st = uu.new_stage(path, root_name, doc=f"{cell.id} full level: terrain at full resolution and every placed asset.")
        rootp = f"/{root_name}"
        rp = UsdGeom.Xform.Define(st, rootp).GetPrim()
        uu.set_kind(rp, Kind.Tokens.group)
        stats = dict(file=os.path.relpath(path, cfg.out), meshes=0, triangles=0, instances=0, point_instances=0,
                     assets=set(), by_group={})
        # Terrain: ground chunks, each a LOD root (fine level in the tile's ground file, the 1 m base level inline).
        self._write_ground_chunks(st, rootp, path, cell, stats)
        content = self.content(cell)
        st.GetRootLayer().Save()
        layer = st.GetRootLayer()
        # Native instances, authored with Sdf for speed.
        pl = content.placed
        order = np.lexsort((pl.pos[:, 0], pl.pos[:, 1]))
        counters = {}
        group_specs = {}
        group_pts = {}
        for i in order:
            a, v = pl.asset[i], pl.variant[i]
            grp = GROUPS[a]
            if grp not in group_specs:
                gs = Sdf.PrimSpec(layer.GetPrimAtPath(rootp), grp, Sdf.SpecifierDef, "Xform")
                gs.SetInfo("kind", "group")
                group_specs[grp] = gs
                group_pts[grp] = []
            k = counters.get(a, 0)
            counters[a] = k + 1
            spec = L.ASSETS_BY_NAME[a]
            ps = Sdf.PrimSpec(group_specs[grp], f"{a}_{k:05d}", Sdf.SpecifierDef, "Xform")
            ps.SetInfo("instanceable", True)
            ps.referenceList.Prepend(Sdf.Reference(uu.rel(path, spec.interface_path(cfg.out))))
            ps.variantSelections[spec.variant_set] = v
            p = pl.pos[i]
            uu.sdf_xform_trs(ps, (p[0] - cx, p[1] - cy, p[2]), pl.rz[i], pl.scale[i], pl.tilt[i])
            stats["assets"].add(a)
            stats["by_group"][grp] = stats["by_group"].get(grp, 0) + 1
        stats["instances"] = int(len(pl))
        # Group bounds.
        for grp in group_specs:
            idx = np.array([i for i in range(len(pl)) if GROUPS[pl.asset[i]] == grp])
            sub = Placement(pl.asset[idx], pl.variant[idx], pl.pos[idx], pl.rz[idx], pl.scale[idx], pl.tilt[idx])
            lo, hi = self._instance_bounds(sub)
            off = np.array([cx, cy, 0])
            group_specs[grp].SetInfo("customData", {"instances": int(len(idx))})
            uu.sdf_attr(group_specs[grp], "extentsHint", Sdf.ValueTypeNames.Float3Array,
                        Vt.Vec3fArray([Gf.Vec3f(*(lo - off)), Gf.Vec3f(*(hi - off))]), Sdf.VariabilityUniform)
            prim_spec = group_specs[grp]
            prim_spec.SetInfo("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["GeomModelAPI"]))
        layer.Save()
        # Ground cover: point instancers in chunks.
        st = Usd.Stage.Open(path, Usd.Stage.LoadNone)
        gc = content.ground
        n = int(cfg.c.density.scatter_chunks)
        if gc is not None and len(gc.pos):
            gp = f"{rootp}/GroundCover"
            gpr = UsdGeom.Xform.Define(st, gp).GetPrim()
            uu.set_kind(gpr, Kind.Tokens.component)
            size = cell.level.size / n
            ci = np.clip(((gc.pos[:, 0] - cell.x0) / size).astype(int), 0, n - 1)
            cj = np.clip(((gc.pos[:, 1] - cell.y0) / size).astype(int), 0, n - 1)
            glo = np.full(3, np.inf)
            ghi = np.full(3, -np.inf)
            for j in range(n):
                for i in range(n):
                    m = (ci == i) & (cj == j)
                    if not m.any():
                        continue
                    lo, hi = self._write_instancer(st, f"{gp}/chunk_{i}_{j}", path, gc, m, cx, cy)
                    glo = np.minimum(glo, lo)
                    ghi = np.maximum(ghi, hi)
                    stats["point_instances"] += int(m.sum())
            uu.set_extents_hint(gpr, glo, ghi)
        lo, hi = self.cell_bounds(cell)
        uu.set_extents_hint(st.GetPrimAtPath(rootp), lo - [cx, cy, 0], hi - [cx, cy, 0])
        st.GetRootLayer().Save()
        stats["assets"] = sorted(stats["assets"])
        stats["prototype_counts"] = {str(k): int(v) for k, v in zip(*np.unique(gc.proto, return_counts=True))} \
            if gc is not None and len(gc.pos) else {}
        self._tile_triangle_stats(stats, pl, gc)
        # At its finest every chunk shows its fine ground instead of the 1 m base.
        stats["triangles_finest"] += sum(g["triangles"] - g["base_triangles"] for g in stats["ground_chunks"].values())
        _log(f"{cell.id} full: {stats['instances']} instances, {stats['point_instances']} point instances, "
             f"{stats['triangles']} own tris ({time.time() - t0:.1f}s)")
        return stats

    def _tile_triangle_stats(self, stats, pl, gc):
        """Expanded triangle counts: every instance at its default LOD level, and at its finest."""
        d_tris = 0
        f_tris = 0
        for a, v in zip(pl.asset, pl.variant):
            t = self.assets.levels(a, v)["tris"]
            spec = L.ASSETS_BY_NAME[a]
            dflt = len(t) - 1 if self.cfg.c.lod.asset_default_level == "coarsest" else 0
            d_tris += t[dflt]
            f_tris += t[0]
        if gc is not None and len(gc.pos):
            names, counts = np.unique(gc.proto, return_counts=True)
            for nme, c in zip(names, counts):
                t = self.assets.scatter(nme)["tris"] * int(c)
                d_tris += t
                f_tris += t
        stats["triangles_instanced_default"] = int(d_tris)
        stats["triangles_instanced_finest"] = int(f_tris)
        stats["triangles_default"] = int(stats["triangles"] + d_tris)
        stats["triangles_finest"] = int(stats["triangles"] + f_tris)

    def _write_instancer(self, st, ip, layer_path, gc, mask, cx, cy):
        pi = UsdGeom.PointInstancer.Define(st, ip)
        protos = sorted(set(gc.proto[mask]))
        pr = st.DefinePrim(f"{ip}/Prototypes", "Scope")
        targets = []
        for nme in protos:
            pp = st.DefinePrim(f"{ip}/Prototypes/{nme}", "Xform")
            pp.GetReferences().AddReference(uu.rel(layer_path, L.scatter_path(self.cfg.out, nme)))
            targets.append(pp.GetPath())
        pi.CreatePrototypesRel().SetTargets(targets)
        lut = {n: i for i, n in enumerate(protos)}
        names = gc.proto[mask]
        pidx = np.array([lut[x] for x in names], dtype=np.int32)
        pos = gc.pos[mask] - np.array([cx, cy, 0.0])
        q = gc.orient[mask].astype(np.float16)
        sc = gc.scale[mask].astype(np.float32)
        pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(pidx))
        pi.CreatePositionsAttr(uu.vec3f_array(pos))
        pi.CreateOrientationsAttr(Vt.QuathArray.FromNumpy(np.ascontiguousarray(q)))
        pi.CreateScalesAttr(uu.vec3f_array(np.repeat(sc[:, None], 3, 1)))
        ids = (np.round(gc.pos[mask, 0] * 64).astype(np.int64) * 1000003 + np.round(gc.pos[mask, 1] * 64).astype(np.int64))
        pi.CreateIdsAttr(Vt.Int64Array.FromNumpy(ids))
        # Conservative but tight: prototype footprint radius, its height, and up to ~20 degrees of tilt.
        ex = {n: self.assets.scatter(n) for n in protos}
        rxy = np.array([float(np.max(np.abs(np.r_[ex[n]["lo"][:2], ex[n]["hi"][:2]]))) for n in protos])[pidx] * sc
        zl = np.array([float(ex[n]["lo"][2]) for n in protos])[pidx] * sc
        zh = np.array([float(ex[n]["hi"][2]) for n in protos])[pidx] * sc
        lo = (pos - np.c_[rxy + 0.35 * zh, rxy + 0.35 * zh, -(zl - 0.35 * rxy)]).min(0)
        hi = (pos + np.c_[rxy + 0.35 * zh, rxy + 0.35 * zh, zh + 0.35 * rxy]).max(0)
        pi.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*lo), Gf.Vec3f(*hi)]))
        return lo, hi

    # -- coarse-level instances --------------------------------------------------------------------------------------

    def _switch_distances(self, a, v):
        """The asset's own LOD switch distances (metres, reference camera, unit scale) per level boundary."""
        key = ("switch", a, v)
        if key not in self._content:
            from .manifest import switch_distance
            d = self.assets.levels(a, v)
            th = self.cfg.c.lod.assets[L.ASSETS_BY_NAME[a].lod_class]["screen_thresholds"]
            self._content[key] = np.array([switch_distance(self.cfg, d["lo"], d["hi"], t) for t in th])
        return self._content[key]

    def _write_instances(self, st, rootp, layer_path, pl, center, spec, distance):
        """Point instances of the assets' own LOD levels for a coarse level that takes over at `distance`: each
        instance gets the level its asset would select at that distance (switch distances scale with the instance),
        after the level's size filters. Returns (instances, instanced triangles, assets)."""
        cfg = self.cfg
        keep = np.zeros(len(pl), bool)
        level = np.zeros(len(pl), int)
        for i in range(len(pl)):
            a, v = pl.asset[i], pl.variant[i]
            d = self.assets.levels(a, v)
            key = HEIGHT_FILTER.get(GROUPS[a])
            size = (d["hi"][2] if key else float(np.max(d["hi"] - d["lo"]))) * pl.scale[i]
            if size < float(spec.get(key or "min_size", 0.0)):
                continue
            keep[i] = True
            level[i] = min(len(d["levels"]) - 1, int(np.sum(self._switch_distances(a, v) * pl.scale[i] <= distance)))
        idx = np.nonzero(keep)[0]
        if not len(idx):
            return 0, 0, []
        ip = f"{rootp}/Instances"
        pi = UsdGeom.PointInstancer.Define(st, ip)
        st.DefinePrim(f"{ip}/Prototypes", "Scope")
        protos = sorted({(pl.asset[i], pl.variant[i], int(level[i])) for i in idx})
        lut = {p: k for k, p in enumerate(protos)}
        targets = []
        mats = set()
        tris = 0
        for a, v, li in protos:
            spec_a = L.ASSETS_BY_NAME[a]
            pp = st.DefinePrim(f"{ip}/Prototypes/{a}_{v}_lod{li}", "Xform")
            pp.GetReferences().AddReference(uu.rel(layer_path, spec_a.geom_path(cfg.out)), f"/{a}_geom/{v}/lod{li}")
            for part, mat in self.assets.materials(a, v).items():
                if part not in self.assets.levels(a, v)["levels"][li]:
                    continue
                mp = f"{rootp}/Materials/{mat}"
                if mat not in mats:
                    _lib_material(st, rootp, layer_path, cfg, mat)
                    mats.add(mat)
                part_prim = st.OverridePrim(pp.GetPath().AppendChild(part))
                uu.bind(part_prim, mp)
                if spec_a.collision:
                    # Far stand-ins never collide (physics uses the cell's own levels).
                    part_prim.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
            targets.append(pp.GetPath())
        pi.CreatePrototypesRel().SetTargets(targets)
        pidx = np.array([lut[(pl.asset[i], pl.variant[i], int(level[i]))] for i in idx], dtype=np.int32)
        pi.CreateProtoIndicesAttr(Vt.IntArray.FromNumpy(pidx))
        pi.CreatePositionsAttr(uu.vec3f_array(pl.pos[idx] - center))
        q = _quat_xyz(np.radians(pl.tilt[idx, 0]), np.radians(pl.tilt[idx, 1]), np.radians(pl.rz[idx]))
        pi.CreateOrientationsAttr(Vt.QuathArray.FromNumpy(np.ascontiguousarray(q.astype(np.float16))))
        pi.CreateScalesAttr(uu.vec3f_array(np.repeat(pl.scale[idx, None], 3, 1)))
        sub = Placement(pl.asset[idx], pl.variant[idx], pl.pos[idx], pl.rz[idx], pl.scale[idx], pl.tilt[idx])
        lo, hi = self._instance_bounds(sub)
        pi.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*(lo - center)), Gf.Vec3f(*(hi - center))]))
        for i in idx:
            tris += self.assets.levels(pl.asset[i], pl.variant[i])["tris"][level[i]]
        return int(len(idx)), int(tris), sorted({pl.asset[i] for i in idx})

    def write_coarse(self, cell, spec):
        """A coarse level of any cell: terrain at reduced resolution plus point instances of the assets' LOD levels."""
        cfg = self.cfg
        lname = spec["name"]
        path = cell_level_path(cfg, cell, lname)
        t0 = time.time()
        cx, cy = cell.center
        center = np.array([cx, cy, 0.0])
        root_name = f"{cell.file_stem}_{lname}"
        st = uu.new_stage(path, root_name, doc=f"{cell.id} {lname} level: reduced terrain and point instances of the "
                                               f"assets' own LOD levels.")
        rootp = f"/{root_name}"
        rp = UsdGeom.Xform.Define(st, rootp).GetPrim()
        uu.set_kind(rp, Kind.Tokens.group)
        stats = dict(file=os.path.relpath(path, cfg.out), meshes=0, triangles=0, instances=0, point_instances=0,
                     assets=[])
        step, _ = level_spacing(cfg, cell, lname)
        g, sk, depth = terrain_meshes(cfg, self.world, cell, step)
        # Far material (macro normal baked against this level's grid) for coarse levels of non-finest cells, unless
        # the level asks for the near material (shared with the finest cells' coarse levels at the same spacing).
        far = cell.level.index < len(cfg.levels) - 1 and spec.get("terrain_material", "far") == "far"
        tp = f"{rootp}/Terrain"
        tpr = UsdGeom.Xform.Define(st, tp).GetPrim()
        uu.set_kind(tpr, Kind.Tokens.component)
        mat = _terrain_material(st, tp, path, cfg, cell, far=far)
        uu.write_mesh(st, f"{tp}/ground", g, material=mat, collision="none")
        uu.write_mesh(st, f"{tp}/skirt", sk, material=mat)
        # Authored on every level, on only on the default one (no reliance on the schema fallback).
        st.GetPrimAtPath(f"{tp}/ground").CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(
            self._default_index(cell.level) == cell.level.lod_names.index(lname))
        uu.set_extents_hint(tpr, *g.extent())
        stats["meshes"] += 2
        stats["triangles"] += g.n_tris + sk.n_tris
        stats["terrain_triangles"] = g.n_tris
        stats["skirt_depth"] = depth
        leaves = cfg.leaf_cells(cell)
        pl = Placement.concat([self.content(l, ground=False).placed for l in leaves])
        takeover = cell.level.distance_thresholds[cell.level.lod_names.index(lname) - 1]
        n, itris, assets = self._write_instances(st, rootp, path, pl, center, spec, takeover)
        stats["point_instances"] = n
        stats["instanced_triangles"] = itris
        stats["assets"] = assets
        lo, hi = self.cell_bounds(cell)
        uu.set_extents_hint(rp, lo - center, hi - center)
        st.GetRootLayer().Save()
        stats["triangles_default"] = stats["triangles"] + itris
        stats["triangles_finest"] = stats["triangles"] + itris
        _log(f"{cell.id} {lname}: {stats['triangles']} terrain tris, {n} instances ({itris} tris) "
             f"({time.time() - t0:.1f}s)")
        return stats

    # -- ground chunks -----------------------------------------------------------------------------------------------

    def chunk_skirt(self, chunk):
        """Skirt depth of a ground chunk (both levels): covers the 1 m base against every terrain level that can
        neighbour it, and the fine level against the base."""
        key = ("chunk_skirt", chunk.id)
        if key not in self._content:
            g = gr.ground_cfg(self.cfg)
            x0, y0, x1, y1 = chunk.bounds2d
            edges = [(x0, y0, x1, y0), (x1, y0, x1, y1), (x0, y1, x1, y1), (x0, y0, x0, y1)]
            sp = float(self.cfg.c.terrain.spacing)
            gap = max(gr.fine_edge_gap(self.world, *e, float(g.fine_spacing), sp) for e in edges)
            self._content[key] = max(skirt_depth(self.cfg, self.world, chunk, sp),
                                     gap + float(self.cfg.c.terrain.skirt_margin))
        return self._content[key]

    def _fine_meshes(self, chunk, tile, depth):
        """Fine ground and skirt of a chunk in tile-local coordinates; st spans the tile (its fine texture)."""
        g = gr.ground_cfg(self.cfg)
        step = float(g.fine_spacing)
        q = int(round(chunk.level.size / step))
        x0, y0 = chunk.x0, chunk.y0
        tx, ty = tile.center
        Hx = gr.fine_grid(self.world, x0 - step, y0 - step, q + 2, step)
        H = Hx[1:-1, 1:-1]
        S = tile.level.size

        def st_fn(x, y):
            return np.stack([(x + tx - tile.x0) / S, (y + ty - tile.y0) / S], 1)
        m = grid_mesh(H, x0 - tx, y0 - ty, step, st_fn)
        gy, gx = np.gradient(Hx, step)
        gx, gy = gx[1:-1, 1:-1], gy[1:-1, 1:-1]
        n = np.stack([-gx.ravel(), -gy.ravel(), np.ones(gx.size)], 1)
        m.normals = n / np.linalg.norm(n, axis=1, keepdims=True)
        nx = q + 1
        loop = np.array(list(range(0, nx)) + [r * nx + q for r in range(1, nx)] +
                        [q * nx + c for c in range(q - 1, -1, -1)] + [r * nx for r in range(q - 1, 0, -1)])
        sk = skirt(m.points[loop], m.normals[loop], m.st[loop], depth)
        return m, sk

    def chunk_bounds(self, chunk):
        """World-space bounds of a ground chunk's levels (both meshes and skirts), plus a small margin."""
        key = ("chunk_bounds", chunk.id)
        if key not in self._content:
            step = float(gr.ground_cfg(self.cfg).fine_spacing)
            q = int(round(chunk.level.size / step))
            hf = gr.fine_grid(self.world, chunk.x0, chunk.y0, q, step)
            hb = self.world.grid(chunk.x0, chunk.y0, int(round(chunk.level.size / self.world.spacing)), 1)
            x0, y0, x1, y1 = chunk.bounds2d
            depth = self.chunk_skirt(chunk)
            m = 0.25
            self._content[key] = (np.array([x0 - m, y0 - m, min(hf.min(), hb.min()) - depth - m]),
                                  np.array([x1 + m, y1 + m, max(hf.max(), hb.max()) + m]))
        return self._content[key]

    def write_ground(self, tile):
        """The tile's ground file: the fine level of every chunk, one prim per chunk (each a payload target), in
        tile-local coordinates with its own material on the tile's fine texture. Returns {chunk name: stats}."""
        cfg = self.cfg
        path = gr.ground_path(cfg, tile)
        tex = gr.bake_tile_fine(cfg, self.world, tile)
        root_name = f"{tile.file_stem}_ground"
        st = uu.new_stage(path, root_name, doc=f"{tile.id} fine ground: one prim per chunk, each the payload of "
                                               f"the chunk's `fine` level in the tile's full level.")
        rootp = f"/{root_name}"
        uu.set_kind(UsdGeom.Xform.Define(st, rootp).GetPrim(), Kind.Tokens.group)
        lib = os.path.join(cfg.out, atlas.MATERIAL_LIBRARY)
        rep = tile.level.size / float(cfg.c.terrain.detail_tiling)
        out = {}
        for c in gr.chunks(cfg, tile):
            cp = f"{rootp}/{c.name}"
            cpr = UsdGeom.Xform.Define(st, cp).GetPrim()
            uu.set_kind(cpr, Kind.Tokens.component)
            st.DefinePrim(f"{cp}/Materials", "Scope")
            mp = f"{cp}/Materials/terrain"
            mprim = st.DefinePrim(mp, "Material")
            mprim.GetReferences().AddReference(uu.rel(path, lib), "/Materials/terrain")
            mat = UsdShade.Material(mprim)
            mat.CreateInput("color_texture", Sdf.ValueTypeNames.Asset).Set(
                Sdf.AssetPath(uu.rel(path, os.path.join(cfg.out, tex))))
            mat.CreateInput("detail_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(rep, rep))
            depth = self.chunk_skirt(c)
            g, sk = self._fine_meshes(c, tile, depth)
            uu.write_mesh(st, f"{cp}/ground", g, material=mp)
            uu.write_mesh(st, f"{cp}/skirt", sk, material=mp)
            uu.set_extents_hint(cpr, *g.extent())
            nbytes = sum(m.n_points * (12 + 12 + 8) + len(m.indices) * 4 + len(m.counts) * 4 for m in (g, sk))
            out[c.name] = dict(triangles=g.n_tris + sk.n_tris, terrain_triangles=g.n_tris, bytes=int(nbytes),
                               skirt_depth=depth)
        st.GetRootLayer().Save()
        return out

    def _write_ground_chunks(self, st, rootp, layer_path, tile, stats):
        """The ground chunks of a tile's full level: LOD roots (children of the level) with the fine level a payload
        into the tile's ground file and the 1 m base level inline; heuristics under LODHeuristics."""
        cfg = self.cfg
        fine = self.write_ground(tile)
        gpath = gr.ground_path(cfg, tile)
        mat = _terrain_material(st, rootp, layer_path, cfg, tile, far=False)
        hc = f"{rootp}/LODHeuristics"
        UsdGeom.Xform.Define(st, hc)
        tx, ty = tile.center
        off = np.array([tx, ty, 0.0])
        stats["terrain_triangles"] = 0
        stats["skirt_depth"] = 0.0
        stats["ground_chunks"] = {}
        for c in gr.chunks(cfg, tile):
            lo, hi = self.chunk_bounds(c)
            depth = self.chunk_skirt(c)
            levels = [("fine", "payload", uu.rel(layer_path, gpath), f"/{tile.file_stem}_ground/{c.name}"),
                      ("base", "inline", None, None)]
            cp = self._author_lod_root(st, rootp, hc, c, c.level, levels, (0.0, 0.0, 0.0), lo - off, hi - off,
                                       screen=False, custom={"ground": True})
            st.GetPrimAtPath(f"{cp}/fine").SetCustomDataByKey("bytes", fine[c.name]["bytes"])
            g, sk, _ = terrain_meshes(cfg, self.world, c, self.world.spacing, depth=depth, origin=(tx, ty))
            uu.write_mesh(st, f"{cp}/base/ground", g, material=mat, collision="none")
            uu.write_mesh(st, f"{cp}/base/skirt", sk, material=mat)
            # base is the chunk's default level, so it is the collision level.
            st.GetPrimAtPath(f"{cp}/base/ground").CreateAttribute(
                "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
            stats["meshes"] += 2
            stats["triangles"] += g.n_tris + sk.n_tris
            stats["terrain_triangles"] += g.n_tris
            stats["skirt_depth"] = max(stats["skirt_depth"], depth)
            stats["ground_chunks"][c.name] = dict(fine[c.name], base_triangles=g.n_tris + sk.n_tris,
                                                  bounds=[lo.round(2).tolist(), hi.round(2).tolist()])

    def _default_index(self, level):
        return len(level.lod_names) - 1 if self.cfg.c.partition.default_level == "coarsest" else 0

    # -- structure files ---------------------------------------------------------------------------------------------

    def _author_cell_root(self, stage, container_path, heur_container, cell, parent_center, layer_path):
        """A cell prim (LOD root) with one payload per level (the full level of a non-finest cell may be a reference
        instead, see `full_arc`), its heuristics and its bounds."""
        cfg = self.cfg
        lv = cell.level
        cx, cy = cell.center
        lo, hi = self.cell_bounds(cell)
        off = np.array([cx, cy, 0.0])
        levels = []
        for ln in lv.lod_names:
            arc = "reference" if ln == "full" and cfg.children(cell) and lv.full_arc == "reference" else "payload"
            levels.append((ln, arc, uu.rel(layer_path, cell_level_path(cfg, cell, ln)), None))
        return stage.GetPrimAtPath(self._author_lod_root(
            stage, container_path, heur_container, cell, lv, levels,
            (cx - parent_center[0], cy - parent_center[1], 0.0), lo - off, hi - off))

    def _author_lod_root(self, stage, container_path, heur_container, cell, lv, levels, translate, llo, lhi,
                         screen=True, custom=None):
        """A LOD root prim for `cell` with `levels` [(name, arc, asset, prim path)] (arc payload, reference or inline),
        the heuristics of level `lv` (imaging distance [+ screen size], physics, streaming) beside it under
        `heur_container`, and its bounds (llo, lhi in the root's own space). Returns the root's path."""
        cp = f"{container_path}/{cell.name}"
        xf = UsdGeom.Xform.Define(stage, cp)
        if any(translate):
            uu.xform_ops(xf, translate=translate)
        prim = xf.GetPrim()
        uu.set_kind(prim, Kind.Tokens.group)
        uu.set_extents_hint(prim, llo, lhi)
        prim.SetCustomDataByKey("cell", cell.id)
        for k, v in (custom or {}).items():
            prim.SetCustomDataByKey(k, v)
        default = self._default_index(lv)
        for i, (ln, arc, asset, target) in enumerate(levels):
            lp = stage.DefinePrim(f"{cp}/{ln}", "Xform")
            uu.set_kind(lp, Kind.Tokens.group if arc != "inline" and target is None else Kind.Tokens.component)
            if arc == "reference":
                lp.GetReferences().AddReference(asset)
            elif arc == "payload":
                lp.GetPayloads().AddPayload(Sdf.Payload(asset, target) if target else Sdf.Payload(asset))
            if i != default:
                UsdGeom.Imageable(lp).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            uu.set_extents_hint(lp, llo, lhi)
        lod = UsdLod.RootAPI.Apply(prim)
        lod.CreateLodDefaultIndexAttr(default)
        # Heuristics live beside the roots (never inside a root: a root's children are its levels).
        hx = UsdGeom.Xform.Define(stage, f"{heur_container}/{cell.name}")
        if any(translate):
            uu.xform_ops(hx, translate=translate)
        bp = f"{heur_container}/{cell.name}/bounds"
        bm = UsdGeom.Mesh.Define(stage, bp)
        corners = np.array([[x, y, z] for z in (llo[2], lhi[2]) for y in (llo[1], lhi[1]) for x in (llo[0], lhi[0])])
        faces = [0, 2, 3, 1, 4, 5, 7, 6, 0, 1, 5, 4, 2, 6, 7, 3, 0, 4, 6, 2, 1, 3, 7, 5]
        bm.CreatePointsAttr(uu.vec3f_array(corners))
        bm.CreateFaceVertexCountsAttr([4] * 6)
        bm.CreateFaceVertexIndicesAttr(faces)
        bm.CreateExtentAttr(uu.extent_of(corners))
        bm.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        bm.CreatePurposeAttr(UsdGeom.Tokens.guide)
        ext = Vt.Vec3fArray([Gf.Vec3f(*llo), Gf.Vec3f(*lhi)])
        center3 = Gf.Vec3f(*((llo + lhi) / 2))
        targets = []
        hd = UsdLod.DistanceHeuristic.Define(stage, f"{heur_container}/{cell.name}/imaging_distance")
        hd.CreateLodDomainAttr(UsdLod.Tokens.imaging)
        hd.CreateCenterAttr(center3)
        hd.CreateBoundingVolumeRel().SetTargets([bm.GetPath()])
        hd.CreateThresholdsAttr(Vt.FloatArray(lv.distance_thresholds))
        if lv.distance_blend:
            hd.CreateBlendThresholdsAttr(Vt.FloatArray(lv.distance_blend))
        targets.append(hd.GetPath())
        if screen:
            hs = UsdLod.ScreenSizeHeuristic.Define(stage, f"{heur_container}/{cell.name}/imaging_screen")
            hs.CreateLodDomainAttr(UsdLod.Tokens.imaging)
            hs.CreateExtentAttr(ext)
            hs.CreateProjectionMethodAttr(UsdLod.Tokens.projectedSphere)
            hs.CreateBoundingVolumeRel().SetTargets([bm.GetPath()])
            hs.CreateThresholdsAttr(Vt.FloatArray(lv.screen_thresholds))
            if lv.screen_blend:
                hs.CreateBlendThresholdsAttr(Vt.FloatArray(lv.screen_blend))
            targets.append(hs.GetPath())
        if lv.physics_distance_thresholds:
            hp = UsdLod.DistanceHeuristic.Define(stage, f"{heur_container}/{cell.name}/physics_distance")
            hp.CreateLodDomainAttr(UsdLod.Tokens.physics)
            hp.CreateCenterAttr(center3)
            hp.CreateBoundingVolumeRel().SetTargets([bm.GetPath()])
            hp.CreateThresholdsAttr(Vt.FloatArray(lv.physics_distance_thresholds))
            targets.append(hp.GetPath())
        # Streaming: which levels should be resident. The level on the finer side of boundary i is loaded inside
        # thresholds[i] and released beyond streaming:unloadThresholds[i] (hysteresis, client policy).
        hl = UsdLod.DistanceHeuristic.Define(stage, f"{heur_container}/{cell.name}/streaming_distance")
        hl.CreateLodDomainAttr(STREAMING)
        hl.CreateCenterAttr(center3)
        hl.CreateBoundingVolumeRel().SetTargets([bm.GetPath()])
        hl.CreateThresholdsAttr(Vt.FloatArray(lv.streaming_distance_thresholds))
        hl.GetPrim().CreateAttribute(UNLOAD_ATTR, Sdf.ValueTypeNames.FloatArray, custom=True,
                                     variability=Sdf.VariabilityUniform).Set(Vt.FloatArray(lv.streaming_unload_thresholds))
        targets.append(hl.GetPath())
        lod.CreateLodHeuristicsRel().SetTargets(targets)
        return cp

    def write_cell_full(self, cell):
        """Full level of a non-finest cell: its child cells as LOD roots (structure only, small and readable)."""
        cfg = self.cfg
        path = cell_level_path(cfg, cell, "full")
        cx, cy = cell.center
        root_name = f"{cell.file_stem}_full"
        kids = cfg.children(cell)
        st = uu.new_stage(path, root_name, doc=f"{cell.id} full level: its {len(kids)} {kids[0].level.name}s, "
                                               f"each a LOD root with one payload per level.")
        rootp = f"/{root_name}"
        rp = UsdGeom.Xform.Define(st, rootp).GetPrim()
        uu.set_kind(rp, Kind.Tokens.group)
        hc = f"{rootp}/LODHeuristics"
        UsdGeom.Xform.Define(st, hc)
        for k in kids:
            self._author_cell_root(st, rootp, hc, k, (cx, cy), path)
        lo, hi = self.cell_bounds(cell)
        uu.set_extents_hint(rp, lo - [cx, cy, 0], hi - [cx, cy, 0])
        st.GetRootLayer().Save()
        return dict(file=os.path.relpath(path, cfg.out), meshes=0, triangles=0, instances=0, point_instances=0,
                    children=[k.id for k in kids])

    def write_root(self, top_cells):
        cfg = self.cfg
        path = root_path(cfg)
        st = uu.new_stage(path, "World", doc=f"{cfg.c.world.name}: {cfg.world_size:.0f} m x {cfg.world_size:.0f} m "
                                             f"open world. See manifest.json and docs/.")
        layer = st.GetRootLayer()
        layer.subLayerPaths.append("./layers/environment.usda")
        layer.subLayerPaths.append("./layers/viewpoints.usda")
        layer.customLayerData = {"manifest": "./manifest.json", "generator": "generator/ (streamworld)",
                                 "seed": cfg.seed}
        w = UsdGeom.Xform.Define(st, "/World").GetPrim()
        uu.set_kind(w, Kind.Tokens.assembly)
        rg = UsdGeom.Xform.Define(st, "/World/Regions").GetPrim()
        uu.set_kind(rg, Kind.Tokens.group)
        hc = "/World/Regions/LODHeuristics"
        UsdGeom.Xform.Define(st, hc)
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for c in top_cells:
            self._author_cell_root(st, "/World/Regions", hc, c, (0.0, 0.0), path)
            b = self.cell_bounds(c)
            lo = np.minimum(lo, b[0])
            hi = np.maximum(hi, b[1])
        uu.set_extents_hint(rg, lo, hi)
        uu.set_extents_hint(w, lo, hi)
        layer.Save()
        return path
