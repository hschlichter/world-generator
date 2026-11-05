"""The asset library: every reusable asset, written as an interface layer (.usda) plus a geometry layer (.usdc).

Interface layer  assets/<category>/<name>.usda, defaultPrim /<name> (kind component):
    /<name>                 variant set (shape / state / form) selecting the geometry; extentsHint per variant
      Geom                  LODRootAPI; its children lod0..lodN are the levels, finest first
        lod0 .. lodN        each a reference into the geometry layer (inside the variant); non-default levels
                            are authored invisible so consumers that ignore LOD draw exactly the default level
      LODHeuristics         LODScreenSizeHeuristic (imaging) [+ LODDistanceHeuristic (physics)]
      Materials             materials referenced from the shared library, bound inside the asset
Geometry layer   assets/<category>/<name>_geom.usdc, defaultPrim /<name>_geom:
    /<name>_geom/<variant>/lod<i>/<part>   meshes
"""

import os
import time
from dataclasses import dataclass

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLod, UsdPhysics, UsdShade, Vt

from .. import atlas
from .. import usdutil as uu
from . import buildings, fantasy, plants, rocks, structures, trees


@dataclass
class AssetSpec:
    name: str
    category: str
    lod_class: str
    variant_set: str
    variants: list
    collision: bool = False

    def interface_path(self, out):
        return os.path.join(out, "assets", self.category, f"{self.name}.usda")

    def geom_path(self, out):
        return os.path.join(out, "assets", self.category, f"{self.name}_geom.usdc")


ASSETS = [
    AssetSpec("oak", "trees", "tree", "shape", ["a", "b", "c"]),
    AssetSpec("birch", "trees", "tree", "shape", ["a", "b", "c"]),
    AssetSpec("pine", "trees", "tree", "shape", ["a", "b", "c"]),
    AssetSpec("spruce", "trees", "tree", "shape", ["a", "b", "c"]),
    AssetSpec("deadwood", "trees", "tree", "shape", ["a", "b"]),
    AssetSpec("shrub", "bushes", "bush", "shape", ["a", "b", "c"]),
    AssetSpec("juniper", "bushes", "bush", "shape", ["a", "b"]),
    AssetSpec("boulder", "rocks", "rock", "shape", ["round", "angular", "flat"], True),
    AssetSpec("cliff", "rocks", "rock", "shape", ["a", "b", "c"], True),
    AssetSpec("pillar", "rocks", "rock", "shape", ["a", "b"], True),
    AssetSpec("deadfall", "logs", "log", "form", ["log_a", "log_b", "stump"]),
    AssetSpec("standing_stone", "structures", "structure", "shape", ["a", "b", "c"], True),
    AssetSpec("wall", "structures", "structure", "state", ["intact", "ruined", "collapsed"], True),
    AssetSpec("tower", "structures", "structure", "state", ["intact", "ruined"], True),
    AssetSpec("column", "structures", "structure", "state", ["intact", "broken"], True),
    AssetSpec("shrine", "structures", "structure", "state", ["intact", "collapsed"], True),
    AssetSpec("rock_arch", "structures", "structure", "shape", ["a", "b"], True),
    # Medieval settlement (assets/buildings.py).
    AssetSpec("house", "buildings", "building", "style", ["thatch", "slate", "long", "chapel"], True),
    AssetSpec("windmill", "buildings", "building", "shape", ["a"], True),
    AssetSpec("keep", "buildings", "building", "shape", ["a"], True),
    AssetSpec("gatehouse", "buildings", "building", "shape", ["a"], True),
    AssetSpec("well", "props", "structure", "shape", ["a"], True),
    AssetSpec("fence", "props", "structure", "state", ["intact", "broken"], True),
    AssetSpec("haystack", "props", "structure", "shape", ["a"], True),
    AssetSpec("gravestone", "props", "structure", "shape", ["cross", "slab", "round"], True),
    AssetSpec("banner", "props", "structure", "colour", ["red", "blue"], True),
    AssetSpec("wayside_shrine", "props", "structure", "shape", ["a"], True),
    # The uncanny (assets/fantasy.py).
    AssetSpec("wizard_tower", "fantasy", "building", "shape", ["a"], True),
    AssetSpec("runestone", "fantasy", "structure", "shape", ["a", "b", "way"], True),
    AssetSpec("crystal", "fantasy", "structure", "shape", ["a", "b"], True),
    AssetSpec("dragon_bones", "fantasy", "structure", "shape", ["a"], True),
]
KIT_ASSETS = {"house": buildings.house, "windmill": buildings.windmill, "keep": buildings.keep,
              "gatehouse": buildings.gatehouse, "well": buildings.well, "fence": buildings.fence,
              "haystack": buildings.haystack, "gravestone": buildings.gravestone, "banner": buildings.banner,
              "wayside_shrine": buildings.wayside_shrine, "wizard_tower": fantasy.wizard_tower,
              "runestone": fantasy.runestone, "crystal": fantasy.crystal_cluster, "dragon_bones": fantasy.dragon_bones}
ASSETS_BY_NAME = {a.name: a for a in ASSETS}
SCATTER = list(plants.PROTOTYPES)

MATERIAL_FOR_PART = {
    "rock": {"boulder": "rock", "cliff": "rock_cliff", "pillar": "rock_cliff", "standing_stone": "rock",
             "rock_arch": "rock_cliff"},
    "stone": "stone_blocks",
    "bark": "bark_dead",
    "wood": "wood",
}


# Decimation targets (triangles) of rock levels lod1..lod7; lod0 is an icosphere of subdivision 5 (big) or 4.
ROCK_TARGETS_BIG = (9000, 4500, 2200, 1100, 520, 240, 110)
ROCK_TARGETS = (2600, 1300, 650, 330, 170, 90, 48)


def physics_thresholds(lodcfg, nlev):
    """Physics distance thresholds from physics_levels = [near, far] and physics_distance: the near level within
    the distance, the far level beyond it (repeated thresholds skip the levels physics never uses)."""
    if not lodcfg.get("physics_levels"):
        return None
    near, far = (int(x) for x in lodcfg["physics_levels"])
    d = float(lodcfg["physics_distance"])
    return [0.0] * near + [d] * (far - near) + [3.0e38] * (nlev - 1 - far)


def build_asset_levels(spec, variant, vi, seed, cache):
    if spec.name in KIT_ASSETS:
        levels = KIT_ASSETS[spec.name](variant, seed)
        parts = {p for lv in levels for p in lv}
        return levels, {p: buildings.PART_MATERIALS[p] for p in parts}
    if spec.category in ("trees", "bushes"):
        levels, mats = trees.build_levels(spec.name, vi, seed, cache)
        return levels, mats
    if spec.name in ("boulder", "cliff", "pillar", "standing_stone"):
        shape = {"boulder": variant, "cliff": f"cliff_{variant}", "pillar": f"pillar_{variant}",
                 "standing_stone": f"menhir_{variant}"}[spec.name]
        big = spec.name in ("cliff", "pillar")
        lv = rocks.build_levels(shape, seed, cache, ROCK_TARGETS_BIG if big else ROCK_TARGETS)
        return lv, {"rock": MATERIAL_FOR_PART["rock"][spec.name]}
    if spec.name == "rock_arch":
        return structures.arch(variant, seed, cache), {"rock": "rock_cliff"}
    if spec.name == "deadfall":
        return plants.log_levels(variant, seed, cache), {"bark": "bark_dead" if variant != "log_a" else "bark_oak",
                                                         "wood": "wood"}
    if spec.name == "shrine":
        return structures.shrine(variant, seed, cache), {"stone": "stone_blocks"}
    fn = {"wall": structures.wall, "tower": structures.tower, "column": structures.column}[spec.name]
    return fn(variant, seed), {"stone": "stone_blocks"}


def phys_levels_for(spec, lodcfg):
    """The levels physics can select (they carry collision); none without physics_levels."""
    return [int(x) for x in lodcfg.get("physics_levels", [])] if spec.collision else []


def _ext(meshes):
    pts = np.concatenate([m.points for m in meshes if m is not None])
    return pts.min(0), pts.max(0)


def write_asset(cfg, spec, force=False):
    out = cfg.out
    ipath = spec.interface_path(out)
    gpath = spec.geom_path(out)
    if not force and os.path.exists(ipath) and os.path.exists(gpath):
        return ipath
    seed = cfg.seed
    lodcfg = cfg.c.lod.assets[spec.lod_class]
    thresholds = list(lodcfg["screen_thresholds"])
    nlev = len(thresholds) + 1
    default = nlev - 1 if cfg.c.lod.asset_default_level == "coarsest" else 0
    t0 = time.time()

    built = {}
    for vi, v in enumerate(spec.variants):
        levels, mats = build_asset_levels(spec, v, vi, seed, cfg.cache)
        if len(levels) != nlev:
            raise ValueError(f"{spec.name}: built {len(levels)} levels, thresholds need {nlev}")
        built[v] = (levels, mats)

    # Geometry layer.
    gname = f"{spec.name}_geom"
    gst = uu.new_stage(gpath, gname, doc=f"Geometry of asset {spec.name}: /{gname}/<variant>/lod<i>/<part>.")
    root = gst.DefinePrim(f"/{gname}", "Scope")
    tri_counts = {}
    for v, (levels, mats) in built.items():
        gst.DefinePrim(f"/{gname}/{v}", "Scope")
        tri_counts[v] = []
        for li, parts in enumerate(levels):
            lp = f"/{gname}/{v}/lod{li}"
            UsdGeom.Xform.Define(gst, lp)
            nt = 0
            for part, mesh in parts.items():
                if mesh is None:
                    continue
                coll = None
                if spec.collision and li in phys_levels_for(spec, lodcfg):
                    coll = "convexHull" if spec.lod_class == "rock" else "none"
                uu.write_mesh(gst, f"{lp}/{part}", mesh, collision=coll)
                nt += mesh.n_tris
            tri_counts[v].append(nt)
    gst.GetRootLayer().Save()

    # Interface layer.
    st = uu.new_stage(ipath, spec.name, doc=f"Asset {spec.name} ({spec.category}); LOD via UsdLod on /{spec.name}/Geom.")
    rootp = f"/{spec.name}"
    rp = UsdGeom.Xform.Define(st, rootp).GetPrim()
    uu.set_kind(rp, Kind.Tokens.component)
    rp.SetAssetInfoByKey("name", spec.name)
    rp.SetAssetInfoByKey("identifier", Sdf.AssetPath(f"./{spec.name}.usda"))
    geom = UsdGeom.Xform.Define(st, f"{rootp}/Geom").GetPrim()
    lod = UsdLod.RootAPI.Apply(geom)
    lod.CreateLodDefaultIndexAttr(default)
    heur = st.DefinePrim(f"{rootp}/LODHeuristics", "Scope")
    hs = UsdLod.ScreenSizeHeuristic.Define(st, f"{rootp}/LODHeuristics/imaging_screen")
    hs.CreateLodDomainAttr(UsdLod.Tokens.imaging)
    hs.CreateProjectionMethodAttr(UsdLod.Tokens.projectedSphere)
    hs.CreateThresholdsAttr(Vt.FloatArray(thresholds))
    if lodcfg.get("screen_blend"):
        hs.CreateBlendThresholdsAttr(Vt.FloatArray(list(lodcfg["screen_blend"])))
    targets = [hs.GetPath()]
    hp = None
    phys = physics_thresholds(lodcfg, nlev) if spec.collision else None
    phys_levels = [int(x) for x in lodcfg["physics_levels"]] if phys else []
    if phys:
        hp = UsdLod.DistanceHeuristic.Define(st, f"{rootp}/LODHeuristics/physics_distance")
        hp.CreateLodDomainAttr(UsdLod.Tokens.physics)
        hp.CreateThresholdsAttr(Vt.FloatArray(phys))
        targets.append(hp.GetPath())
    lod.CreateLodHeuristicsRel().SetTargets(targets)
    st.DefinePrim(f"{rootp}/Materials", "Scope")
    all_mats = sorted({m for _, (lv, mats) in built.items() for m in mats.values()})
    lib = os.path.join(out, atlas.MATERIAL_LIBRARY)
    for m in all_mats:
        mp = st.DefinePrim(f"{rootp}/Materials/{m}", "Material")
        mp.GetReferences().AddReference(uu.rel(ipath, lib), f"/Materials/{m}")
    for li in range(nlev):
        lp = st.OverridePrim(f"{rootp}/Geom/lod{li}")
        if li != default:
            UsdGeom.Imageable(lp).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    # Variants: geometry reference, bounds, bindings.
    vset = rp.GetVariantSets().AddVariantSet(spec.variant_set)
    for v, (levels, mats) in built.items():
        vset.AddVariant(v)
        vset.SetVariantSelection(v)
        with vset.GetVariantEditContext():
            g = st.OverridePrim(f"{rootp}/Geom")
            g.GetReferences().AddReference(uu.rel(ipath, gpath), f"/{gname}/{v}")
            allm = [m for lv in levels for m in lv.values() if m is not None]
            lo, hi = _ext(allm)
            uu.set_extents_hint(rp, lo, hi)
            hs_v = UsdLod.ScreenSizeHeuristic(st.OverridePrim(hs.GetPath()))
            hs_v.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
            if hp is not None:
                hp_v = UsdLod.DistanceHeuristic(st.OverridePrim(hp.GetPath()))
                hp_v.CreateCenterAttr(Gf.Vec3f(*map(float, (lo + hi) / 2)))
            for li, parts in enumerate(levels):
                for part, mesh in parts.items():
                    if mesh is None:
                        continue
                    pp = st.OverridePrim(f"{rootp}/Geom/lod{li}/{part}")
                    uu.bind(pp, f"{rootp}/Materials/{mats[part]}")
                    if li in phys_levels:
                        pp.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(li == default)
    vset.SetVariantSelection(spec.variants[0])
    rp.SetCustomDataByKey("triangles_per_level", {v: Vt.IntArray(t) for v, t in tri_counts.items()})
    st.GetRootLayer().Save()
    print(f"[assets] {spec.category}/{spec.name}: {len(spec.variants)} variants, tris "
          f"{ {v: t for v, t in tri_counts.items()} } ({time.time() - t0:.1f}s)", flush=True)
    return ipath


def scatter_path(out, name):
    return os.path.join(out, "assets", "scatter", f"{name}.usda")


def write_scatter_asset(cfg, name, force=False):
    path = scatter_path(cfg.out, name)
    if not force and os.path.exists(path):
        return path
    mesh = plants.scatter_mesh(name, cfg.seed)
    st = uu.new_stage(path, name, doc=f"Ground-cover prototype {name} (point-instanced; single level).")
    rp = UsdGeom.Xform.Define(st, f"/{name}").GetPrim()
    # Prototypes live under a point instancer inside a component (a tile's GroundCover), so the asset is a
    # subcomponent rather than a component.
    uu.set_kind(rp, Kind.Tokens.subcomponent)
    st.DefinePrim(f"/{name}/Materials", "Scope")
    mat = plants.PROTOTYPES[name]
    mp = st.DefinePrim(f"/{name}/Materials/{mat}", "Material")
    mp.GetReferences().AddReference(uu.rel(path, os.path.join(cfg.out, atlas.MATERIAL_LIBRARY)), f"/Materials/{mat}")
    uu.write_mesh(st, f"/{name}/Geom", mesh, material=f"/{name}/Materials/{mat}")
    lo, hi = mesh.extent()
    uu.set_extents_hint(rp, lo, hi)
    rp.SetCustomDataByKey("triangles", mesh.n_tris)
    st.GetRootLayer().Save()
    return path


def write_all(cfg, names=None, force=False):
    for spec in ASSETS:
        if names and spec.name not in names:
            continue
        write_asset(cfg, spec, force)
    for n in SCATTER:
        if names and n not in names:
            continue
        write_scatter_asset(cfg, n, force)


def asset_info(cfg, name):
    """Triangles per level per variant and the extents hint per variant, read back from the written asset."""
    spec = ASSETS_BY_NAME.get(name)
    path = spec.interface_path(cfg.out) if spec else scatter_path(cfg.out, name)
    layer = Sdf.Layer.FindOrOpen(path)
    prim = layer.GetPrimAtPath(f"/{name}")
    return dict(prim.customData)
