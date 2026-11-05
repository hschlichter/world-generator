"""world/manifest.json: bounds, the cell hierarchy, per cell and level its file and counts, totals, parameters and
the LOD each viewpoint is expected to see."""

import json
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLod

from .assets import library as L
from .assets.cache import AssetCache
from .cells import cell_level_path, root_path

DEFAULT_RATIONALE = (
    "Every LOD root (regions, tiles, ground chunks and assets) defaults to its coarsest level, and every non-default level is "
    "authored invisible. A consumer that does not understand UsdLod therefore draws exactly one level per root: "
    "the cheapest complete representation, which for the root stage is the whole map as region proxies (the far "
    "field). That is the only default whose cost is bounded for the full world, and it is what a LOD-aware "
    "renderer converges to with the camera far away. A LOD-aware consumer owns the visibility of LOD levels: it "
    "shows the selected level and hides the others (generator/streamworld/lodselect.py and the usdview plugin "
    "do this with session-layer opinions). For physics the same rule applies through physics:collisionEnabled.")


def reference_frustum(cfg, eye=(0.0, 0.0, 0.0), target=(0.0, 1.0, 0.0)):
    from .viewpoints import look_at
    rc = cfg.c.lod.reference_camera
    cam = Gf.Camera(look_at(eye, target), Gf.Camera.Perspective, float(rc.horizontal_aperture),
                    float(rc.horizontal_aperture) / float(rc.aspect), 0.0, 0.0, float(rc.focal_length))
    cam.clippingRange = Gf.Range1f(0.3, 20000.0)
    return cam.frustum


def switch_distance(cfg, lo, hi, threshold):
    """Distance at which a projected-sphere screen size of an extent equals `threshold`, for the reference camera."""
    q = UsdLod.ScreenSizeHeuristicQuery(UsdLod.Tokens.imaging, Gf.Range3d(Gf.Vec3d(*lo), Gf.Vec3d(*hi)),
                                        UsdLod.Tokens.projectedSphere, [threshold], [])
    a, b = 0.5, 50000.0
    for _ in range(60):
        m = (a * b) ** 0.5
        fr = reference_frustum(cfg, (0, -m, (lo[2] + hi[2]) / 2), (0, 0, (lo[2] + hi[2]) / 2))
        s = q.ComputeScreenSize(fr, Gf.Matrix4d(1))
        if s > threshold:
            a = m
        else:
            b = m
    return round((a * b) ** 0.5, 1)


def expected_cell_lods(cfg, gen, eye, cells):
    """LOD level each cell root should select from `eye` by the generator's own geometry (distance from the eye to
    the cell's bounding box against the cell level's distance thresholds), recursing into selected full levels."""
    from . import ground as gr
    out = {}

    def visit(cell):
        lo, hi = gen.chunk_bounds(cell) if cell.level.name == "ground" else gen.cell_bounds(cell)
        d = float(np.linalg.norm(np.maximum(0, np.maximum(lo - eye, eye - hi))))
        th = cell.level.distance_thresholds
        bl = cell.level.distance_blend or th
        # LOD value with the schema's blend semantics (linear from threshold to blend threshold), rounded the way a
        # renderer that does not blend picks a level.
        val = 0.0
        for t, b in zip(th, bl):
            if d >= t:
                val += 1.0 if b <= t else min(1.0, (d - t) / (b - t))
        idx = int(np.floor(val + 0.5))
        name = (gr.GROUND_LOD_NAMES if cell.level.name == "ground" else cell.level.lod_names)[idx]
        out[cell.id] = dict(level=name, distance=round(d, 1), lod_value=round(val, 3))
        if idx == 0 and cell.level.name != "ground":
            # The finest cells' full level holds the ground chunks (LOD roots of their own).
            for k in cfg.children(cell) or gr.chunks(cfg, cell):
                visit(k)
    for c in cells:
        visit(c)
    return out


def gr_names():
    from .ground import GROUND_LOD_NAMES
    return GROUND_LOD_NAMES


def _geom_bytes(levels):
    b = 0
    for m in levels.values():
        b += m.n_points * (12 + 12 + 8) + len(m.indices) * 4 + len(m.counts) * 4
    return b


def build_manifest(cfg, world):
    from .cells import Generator
    from .viewpoints import choose_viewpoints, flight_path
    gen = Generator(cfg, world)
    ac = AssetCache(cfg)
    st = Usd.Stage.Open(root_path(cfg), Usd.Stage.LoadNone)
    regions = [p for p in st.GetPrimAtPath("/World/Regions").GetChildren() if p.HasAPI(UsdLod.RootAPI)]
    present = [cfg.cell_by_id(p.GetCustomDataByKey("cell")) for p in regions]
    stats_dir = os.path.join(cfg.out, "stats")

    def stats(cell, level):
        p = os.path.join(stats_dir, f"{cell.file_stem}_{level}.json")
        with open(p) as f:
            return json.load(f)

    # Asset geometry per level, for expanded totals.
    asset_rows = []
    lod0_bytes = {}
    for spec in L.ASSETS:
        rows = {}
        for v in spec.variants:
            d = ac.levels(spec.name, v)
            rows[v] = d["tris"]
            lod0_bytes[(spec.name, v)] = _geom_bytes(d["levels"][0])
        lc = cfg.c.lod.assets[spec.lod_class]
        d0 = ac.levels(spec.name, spec.variants[0])
        asset_rows.append(dict(
            name=spec.name, category=spec.category, file=os.path.relpath(spec.interface_path(cfg.out), cfg.out),
            geometry=os.path.relpath(spec.geom_path(cfg.out), cfg.out), variant_set=spec.variant_set,
            variants=spec.variants, levels=len(lc["screen_thresholds"]) + 1, triangles_per_level=rows,
            heuristics=dict(imaging="LODScreenSizeHeuristic (projectedSphere)", thresholds=lc["screen_thresholds"],
                            blend=lc.get("screen_blend", []), physics_distance=L.physics_thresholds(lc, len(lc["screen_thresholds"]) + 1) if spec.collision else None,
                            physics_levels=L.phys_levels_for(spec, lc) or None),
            reference_switch_distances_m=[switch_distance(cfg, d0["lo"], d0["hi"], t) for t in lc["screen_thresholds"]]))
    scatter_rows = [dict(name=n, file=os.path.relpath(L.scatter_path(cfg.out, n), cfg.out),
                         triangles=ac.scatter(n)["tris"]) for n in L.SCATTER]

    def cell_entry(cell):
        lv = cell.level
        lo, hi = gen.cell_bounds(cell)
        e = dict(id=cell.id, level=lv.name, bounds=[lo.round(2).tolist(), hi.round(2).tolist()],
                 lod=dict(levels=lv.lod_names, default_index=gen._default_index(lv),
                          distance_thresholds=lv.distance_thresholds, distance_blend=lv.distance_blend,
                          screen_thresholds=lv.screen_thresholds, screen_blend=lv.screen_blend,
                          physics_distance_thresholds=lv.physics_distance_thresholds,
                          streaming_distance_thresholds=lv.streaming_distance_thresholds,
                          streaming_unload_thresholds=lv.streaming_unload_thresholds,
                          arcs={ln: (lv.full_arc if ln == "full" and cfg.children(cell) else "payload")
                                for ln in lv.lod_names}),
                 levels={})
        for ln in lv.lod_names:
            s = stats(cell, ln)
            s.pop("cell", None)
            s.pop("level", None)
            e["levels"][ln] = s
        kids = cfg.children(cell)
        if kids:
            e["children"] = [cell_entry(k) for k in kids]
            # The full level of a structural cell: totals of its children at their finest / default.
            f = e["levels"]["full"]
            for key in ("triangles_finest", "instances", "point_instances"):
                f[key + "_subtree"] = sum(_subtree(c, key) for c in e["children"])
        return e

    def _subtree(entry, key):
        if "children" in entry:
            return sum(_subtree(c, key) for c in entry["children"])
        s = entry["levels"]["full"]
        return s.get(key, 0) if key != "triangles_finest" else s.get("triangles_finest", s["triangles"])

    hierarchy = [cell_entry(c) for c in present]

    # Totals.
    far = dict(triangles=0, meshes=0, files=0)
    for c in present:
        s = stats(c, c.level.lod_names[-1])
        far["triangles"] += s["triangles"]
        far["meshes"] += s["meshes"]
        far["files"] += 1
    fin = dict(triangles=0, instances=0, point_instances=0, files=0, geometry_bytes=0)
    for c in present:
        for leaf in cfg.leaf_cells(c):
            s = stats(leaf, "full")
            fin["triangles"] += s["triangles_finest"]
            fin["instances"] += s["instances"]
            fin["point_instances"] += s["point_instances"]
            fin["files"] += 1
            pl = gen.content(leaf, ground=False).placed
            gen._content.pop((leaf.id, False), None)   # one tile at a time, not the world's placements
            keys, counts = np.unique(np.char.add(pl.asset.astype(str), np.char.add("|", pl.variant.astype(str))),
                                     return_counts=True)
            for k, n in zip(keys, counts):
                a, v = k.split("|")
                fin["geometry_bytes"] += lod0_bytes[(a, v)] * int(n)
            for nme, n in s.get("prototype_counts", {}).items():
                m = ac.scatter(nme)["mesh"]
                fin["geometry_bytes"] += _geom_bytes({"m": m}) * n
            fin["geometry_bytes"] += sum(g["bytes"] for g in s["ground_chunks"].values())
    fin["geometry_gb"] = round(fin["geometry_bytes"] / 1e9, 2)
    ntop = len(cfg.top_cells())
    scale = ntop / max(1, len(present))
    totals = dict(
        generated_regions=len(present), world_regions=ntop,
        far_field=far, finest=fin,
        far_field_projected_full_world=dict(triangles=int(far["triangles"] * scale)),
        finest_projected_full_world=dict(triangles=int(fin["triangles"] * scale),
                                         instances=int(fin["instances"] * scale),
                                         point_instances=int(fin["point_instances"] * scale),
                                         geometry_gb=round(fin["geometry_gb"] * scale, 1)),
        note="far_field: every root at its coarsest level (the region proxies). finest: every root at its finest "
             "level with every instance and point instance expanded at lod0. geometry_bytes: points, normals, st "
             "and topology of the expanded finest geometry. *_projected_full_world scales the generated regions "
             "to all regions, for budgeting before the full world is generated.")

    # Viewpoints and their expected LOD.
    from .viewpoints import choose_viewpoints
    vps = choose_viewpoints(cfg, world, present)
    vp_out = {}
    for n, v in vps.items():
        vp_out[n] = dict(camera=f"/cameras/{n}", eye=[round(x, 2) for x in v["eye"]],
                         target=[round(x, 2) for x in v["target"]], note=v["note"],
                         expected_cells=expected_cell_lods(cfg, gen, np.array(v["eye"]), present))
    fl = flight_path(cfg, world)
    flight = [dict(camera=f"/cameras/flight_{i:02d}", eye=[round(x, 1) for x in c["eye"]], fast=c["fast"],
                   expected_cells=expected_cell_lods(cfg, gen, np.array(c["eye"]), present)) for i, c in enumerate(fl)]

    man = dict(
        world=dict(name=cfg.c.world.name, seed=cfg.seed, size_m=cfg.world_size, root="root.usda",
                   metersPerUnit=1.0, upAxis="Z", origin="world centred on (0, 0); x east, y north",
                   bounds=[[-cfg.world_size / 2] * 2, [cfg.world_size / 2] * 2],
                   generated_regions=[c.id for c in present], density_scale=cfg.density_scale()),
        parameters=dict(
            partition=[dict(name=l.name, size_m=l.size, levels=l.lod_names, distance_thresholds_m=l.distance_thresholds,
                            distance_blend_m=l.distance_blend, screen_thresholds=l.screen_thresholds,
                            screen_blend=l.screen_blend, physics_distance_thresholds_m=l.physics_distance_thresholds,
                            streaming_distance_thresholds_m=l.streaming_distance_thresholds,
                            streaming_unload_thresholds_m=l.streaming_unload_thresholds,
                            full_arc=l.full_arc if l.index < len(cfg.levels) - 1 else "payload",
                            coarse=l.coarse) for l in cfg.levels],
            streaming=dict(domain="streaming", heuristic="LODDistanceHeuristic per cell (LODHeuristics/<cell>/"
                           "streaming_distance), thresholds = load radii, streaming:unloadThresholds = unload radii",
                           policy="lodselect.Streamer: resident = the finest level the load radii ask for plus every "
                                  "coarser level; children only while the parent wants its full level; drawing falls "
                                  "back to the nearest coarser resident level"),
            hierarchy_depth=len(cfg.levels),
            ground_lod=dict(cfg.raw["terrain"]["ground_lod"], levels=gr_names(),
                            note="every finest cell's full level holds its ground chunks as LOD roots: `fine` a "
                                 "payload into <tile>_ground.usdc (one prim per chunk), `base` the 1 m terrain inline; "
                                 "imaging and streaming distance heuristics per chunk; per-chunk stats in the tile's "
                                 "full level (ground_chunks)"),
            terrain=dict(spacing_m=float(cfg.c.terrain.spacing), full_quads=int(cfg.c.terrain.full_quads),
                         crack_hiding="skirts: every terrain mesh carries a vertical skirt whose depth covers the worst "
                                      "edge gap against every terrain level that can neighbour it (validated)",
                         normals="central differences of each level's own sample grid; shared edges identical"),
            lod_defaults=dict(cells=cfg.c.partition.default_level, assets=cfg.c.lod.asset_default_level,
                              rationale=DEFAULT_RATIONALE),
            reference_camera=dict(cfg.c.lod.reference_camera),
            density=dict(cfg.raw["density"]),
            heuristic_bounds="cells: LODDistanceHeuristic and LODScreenSizeHeuristic with boundingVolume -> a guide-"
                             "purpose box mesh (LODHeuristics/<cell>/bounds) and an authored extent; assets: authored "
                             "extent per variant (screen size) and center (physics distance)",
        ),
        hierarchy=hierarchy,
        assets=asset_rows,
        scatter_prototypes=scatter_rows,
        totals=totals,
        viewpoints=vp_out,
        flight_path=flight,
    )
    path = os.path.join(cfg.out, "manifest.json")
    with open(path, "w") as f:
        json.dump(man, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"[manifest] {path}: far field {far['triangles']} tris, finest {fin['triangles']} tris, "
          f"{fin['geometry_gb']} GB expanded geometry ({len(present)} regions)", flush=True)
    return man
