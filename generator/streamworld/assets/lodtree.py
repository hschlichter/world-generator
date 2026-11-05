"""LOD test harness for one tree: the 8-level tree as a standalone asset, a lineup stage and switch renders.

The world's trees and bushes are built by trees.build_levels with 8 levels. This module writes one of them
(one species, one variant) as a standalone test asset without variants, with both an imaging screen-size and an
imaging distance heuristic, plus a lineup stage with every level side by side, and renders each switch from its
switch distance with a pop metric. Thresholds come from config [lod.lodtest].

    generator/run.sh lodtree [--species oak] [--variant 0] [--render]

writes world/assets/lodtest/<species>_lod8.usda (interface), <species>_lod8_geom.usdc (geometry) and
<species>_lod8_lineup.usda (every level side by side, and a camera per switch distance).
"""

import os

import numpy as np
from pxr import Gf, Kind, Sdf, UsdGeom, UsdLod, Vt

from .. import atlas
from .. import usdutil as uu
from .trees import N_LEVELS, build_levels


# ---------------------------------------------------------------------------------------------------------------------
# USD
# ---------------------------------------------------------------------------------------------------------------------

def out_dir(cfg):
    return os.path.join(cfg.out, "assets", "lodtest")


def write(cfg, species="oak", variant=0):
    from ..manifest import switch_distance
    lc = cfg.c.lod.lodtest
    thresholds = [float(x) for x in lc["screen_thresholds"]]
    blend = [float(x) for x in lc["screen_blend"]]
    if len(thresholds) != N_LEVELS - 1 or len(blend) != N_LEVELS - 1:
        raise ValueError(f"[lod.lodtest] needs {N_LEVELS - 1} screen_thresholds and screen_blend values")
    default = N_LEVELS - 1 if cfg.c.lod.asset_default_level == "coarsest" else 0
    levels, mats = build_levels(species, variant, cfg.seed, cfg.cache)

    name = f"{species}_lod8"
    d = out_dir(cfg)
    ipath = os.path.join(d, f"{name}.usda")
    gpath = os.path.join(d, f"{name}_geom.usdc")

    # Geometry layer: /<name>_geom/lod<i>/<part>.
    gname = f"{name}_geom"
    gst = uu.new_stage(gpath, gname, doc=f"Geometry of the LOD test tree {name}: /{gname}/lod<i>/<part>.")
    gst.DefinePrim(f"/{gname}", "Scope")
    tris = []
    for li, parts in enumerate(levels):
        UsdGeom.Xform.Define(gst, f"/{gname}/lod{li}")
        for part, mesh in parts.items():
            uu.write_mesh(gst, f"/{gname}/lod{li}/{part}", mesh)
        tris.append(sum(m.n_tris for m in parts.values()))
    gst.GetRootLayer().Save()

    allp = np.concatenate([m.points for lv in levels for m in lv.values()])
    lo, hi = allp.min(0), allp.max(0)
    dist = [switch_distance(cfg, lo, hi, t) for t in thresholds]
    dist_blend = [switch_distance(cfg, lo, hi, t) for t in blend]

    # Interface layer.
    st = uu.new_stage(ipath, name, doc=f"LOD test tree ({species}, {N_LEVELS} levels); LOD via UsdLod on /{name}/Geom.")
    rootp = f"/{name}"
    rp = UsdGeom.Xform.Define(st, rootp).GetPrim()
    uu.set_kind(rp, Kind.Tokens.component)
    rp.SetAssetInfoByKey("name", name)
    rp.SetAssetInfoByKey("identifier", Sdf.AssetPath(f"./{name}.usda"))
    uu.set_extents_hint(rp, lo, hi)
    geom = UsdGeom.Xform.Define(st, f"{rootp}/Geom").GetPrim()
    geom.GetReferences().AddReference(f"./{name}_geom.usdc", f"/{gname}")
    lod = UsdLod.RootAPI.Apply(geom)
    lod.CreateLodDefaultIndexAttr(default)
    st.DefinePrim(f"{rootp}/LODHeuristics", "Scope")
    hs = UsdLod.ScreenSizeHeuristic.Define(st, f"{rootp}/LODHeuristics/imaging_screen")
    hs.CreateLodDomainAttr(UsdLod.Tokens.imaging)
    hs.CreateProjectionMethodAttr(UsdLod.Tokens.projectedSphere)
    hs.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))
    hs.CreateThresholdsAttr(Vt.FloatArray(thresholds))
    hs.CreateBlendThresholdsAttr(Vt.FloatArray(blend))
    # The same switches as distances (for the reference camera), for consumers that prefer a distance heuristic.
    hd = UsdLod.DistanceHeuristic.Define(st, f"{rootp}/LODHeuristics/imaging_distance")
    hd.CreateLodDomainAttr(UsdLod.Tokens.imaging)
    hd.CreateCenterAttr(Gf.Vec3f(*map(float, (lo + hi) / 2)))
    hd.CreateThresholdsAttr(Vt.FloatArray(dist))
    hd.CreateBlendThresholdsAttr(Vt.FloatArray(dist_blend))
    lod.CreateLodHeuristicsRel().SetTargets([hs.GetPath(), hd.GetPath()])
    st.DefinePrim(f"{rootp}/Materials", "Scope")
    lib = os.path.join(cfg.out, atlas.MATERIAL_LIBRARY)
    for m in sorted(set(mats.values())):
        mp = st.DefinePrim(f"{rootp}/Materials/{m}", "Material")
        mp.GetReferences().AddReference(uu.rel(ipath, lib), f"/Materials/{m}")
    for li, parts in enumerate(levels):
        lp = st.OverridePrim(f"{rootp}/Geom/lod{li}")
        if li != default:
            UsdGeom.Imageable(lp).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        for part in parts:
            uu.bind(st.OverridePrim(f"{rootp}/Geom/lod{li}/{part}"), f"{rootp}/Materials/{mats[part]}")
    rp.SetCustomDataByKey("triangles_per_level", Vt.IntArray(tris))
    rp.SetCustomDataByKey("switch_distances_reference_camera", Vt.FloatArray(dist))
    st.GetRootLayer().Save()

    lpath = write_lineup(cfg, name, ipath, lo, hi, dist)
    print(f"[lodtree] {ipath}\n  triangles per level {tris}\n  screen thresholds {thresholds} (blend {blend})\n"
          f"  switch distances {dist} m (blend to {dist_blend} m)\n  lineup {lpath}", flush=True)
    return ipath, lpath, tris, dist


def write_lineup(cfg, name, ipath, lo, hi, dist):
    """Every level side by side (non-instanced references with the level visibility overridden), with lighting,
    a camera framing the row, and a camera at each switch distance looking at the tree in /Switch."""
    from ..viewpoints import look_at
    path = os.path.join(out_dir(cfg), f"{name}_lineup.usda")
    st = uu.new_stage(path, "Lineup", doc=f"{name}: lod0..lod{N_LEVELS - 1} left to right; /Switch is a LOD-driven "
                                          f"copy with cameras at the switch distances.")
    UsdGeom.Xform.Define(st, "/Lineup")
    env = st.DefinePrim("/Lineup/Environment")
    env.GetReferences().AddReference(uu.rel(path, os.path.join(cfg.out, "layers", "environment.usda")))
    w = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 1.05
    zc = float(lo[2] + hi[2]) / 2
    for li in range(N_LEVELS):
        p = UsdGeom.Xform.Define(st, f"/Lineup/lod{li}")
        p.AddTranslateOp().Set(Gf.Vec3d((li - (N_LEVELS - 1) / 2) * w, 0, 0))
        p.GetPrim().GetReferences().AddReference(f"./{name}.usda")
        for lj in range(N_LEVELS):
            vis = UsdGeom.Tokens.inherited if lj == li else UsdGeom.Tokens.invisible
            UsdGeom.Imageable(st.OverridePrim(f"/Lineup/lod{li}/Geom/lod{lj}")).CreateVisibilityAttr(vis)
    ground = UsdGeom.Mesh.Define(st, "/Lineup/Ground")
    g = N_LEVELS * w
    ground.CreatePointsAttr([(-g, -g, 0), (g, -g, 0), (g, g, 0), (-g, g, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateExtentAttr([(-g, -g, 0), (g, g, 0)])
    ground.CreateDisplayColorAttr([(0.32, 0.36, 0.22)])
    sw = UsdGeom.Xform.Define(st, "/Lineup/Switch")
    sw.AddTranslateOp().Set(Gf.Vec3d(0, -2.5 * w, 0))
    sw.GetPrim().GetReferences().AddReference(f"./{name}.usda")
    st.DefinePrim("/Lineup/Cameras", "Scope")
    rc = cfg.c.lod.reference_camera

    def cam(cname, eye, target):
        c = UsdGeom.Camera.Define(st, f"/Lineup/Cameras/{cname}")
        c.CreateFocalLengthAttr(float(rc.focal_length))
        c.CreateHorizontalApertureAttr(float(rc.horizontal_aperture))
        c.CreateVerticalApertureAttr(float(rc.horizontal_aperture) / float(rc.aspect))
        c.CreateClippingRangeAttr(Gf.Vec2f(0.1, 5000.0))
        c.AddTransformOp().Set(look_at(eye, target))

    row = N_LEVELS * w
    cam("row", (0, -row * 0.74, zc + row * 0.04), (0, 0, zc))
    center = np.array([0, -2.5 * w, zc])
    for i, dd in enumerate(dist):
        cam(f"switch_{i}_{i + 1}", tuple(center + [0, -dd, 0]), tuple(center))
    st.GetRootLayer().Save()
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Renders
# ---------------------------------------------------------------------------------------------------------------------

def _session(path, show_row, switch_level):
    """Session layer: the row on or off, /Lineup/Switch showing one level (None: hidden)."""
    layer = Sdf.Layer.CreateAnonymous(".usda")
    for li in range(N_LEVELS):
        Sdf.CreatePrimInLayer(layer, f"/Lineup/lod{li}").SetInfo("active", show_row)
        spec = Sdf.CreatePrimInLayer(layer, f"/Lineup/Switch/Geom/lod{li}")
        a = Sdf.AttributeSpec(spec, "visibility", Sdf.ValueTypeNames.Token)
        a.default = "inherited" if li == switch_level else "invisible"
    layer.Export(path)
    return path


def render(cfg, species="oak", width=1280):
    """usdrecord renders to docs/previews/lodtest/: the lineup, and per switch the two levels it blends between,
    seen from its camera at the switch distance. Each switch gets a crop of both (enlarged, side by side) and a pop
    metric: the mean colour difference over the tree's pixels and the change in covered pixels."""
    from ..imgutil import read_image, write_image
    from ..preview import _record
    name = f"{species}_lod8"
    lineup = os.path.join(out_dir(cfg), f"{name}_lineup.usda")
    od = os.path.join(cfg.root, "docs", "previews", "lodtest")
    sd = os.path.join(cfg.root, "build", "preview_sessions", "lodtest")
    os.makedirs(od, exist_ok=True)
    os.makedirs(sd, exist_ok=True)
    _record(lineup, "/Lineup/Cameras/row", os.path.join(od, f"{name}_lineup.png"), width,
            _session(os.path.join(sd, "row.usda"), True, None))
    report = []
    for i in range(N_LEVELS - 1):
        camp = f"/Lineup/Cameras/switch_{i}_{i + 1}"
        imgs = []
        for lvl in (None, i, i + 1):
            png = os.path.join(sd, f"switch_{i}_{lvl}.png")
            _record(lineup, camp, png, width, _session(os.path.join(sd, f"s{i}_{lvl}.usda"), False, lvl))
            imgs.append(read_image(png)[:, :, :3])
        bg, a, b = imgs
        ma = np.abs(a - bg).max(2) > 0.02
        mb = np.abs(b - bg).max(2) > 0.02
        m = ma | mb
        ys, xs = np.nonzero(m)
        pad = 4
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, a.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, a.shape[1])
        diff = float(np.abs(a - b)[m].mean())
        cov = float(mb.sum() / max(ma.sum(), 1))
        k = max(1, int(360 / (y1 - y0)))
        crop = lambda im: np.repeat(np.repeat(im[y0:y1, x0:x1], k, 0), k, 1)
        sep = np.ones((crop(a).shape[0], 6, 3))
        write_image(os.path.join(od, f"{name}_switch_{i}_{i + 1}.png"), np.concatenate([crop(a), sep, crop(b)], 1))
        report.append(dict(switch=f"lod{i}->lod{i + 1}", tree_pixels=int(ma.sum()), mean_diff=round(diff, 4),
                           coverage_ratio=round(cov, 3)))
        print(f"[lodtree] lod{i}->lod{i + 1}: {int(ma.sum())} px, mean colour diff {diff:.4f}, "
              f"coverage {cov:.3f}", flush=True)
    import json
    with open(os.path.join(od, f"{name}_switches.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report
