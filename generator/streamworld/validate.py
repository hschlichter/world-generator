"""Validation of the generated world (PROMPT section 10). Writes world/validation.json and prints a summary.

Checks:
  usdchecker          usdchecker on every file (cell files with --skipVariants: their variant selections are
                      placement data; every variant is checked in the asset files); the root and the regions' full
                      levels get the same validators with payloads unloaded (fully loaded they need 5-12+ GB)
  conventions         every layer: metersPerUnit 1, upAxis Z; defaultPrim on every referenced/payloaded file;
                      asset paths relative and resolving inside world/
  composition         every reference / payload resolves; no composition errors with everything loaded
  payload_bounds      every payload prim has an extentsHint that contains its loaded content
  gprim_extents       every gprim has an authored extent; models in cell files carry extentsHint
  lod_structure       every LOD root: levels = children, thresholds = levels - 1 and ordered, default index in
                      range, exactly the default level visible, heuristic bounds finite, non-empty and containing
                      the content of every level
  levels_standalone   every level of every cell loads on its own (stage with only that payload loaded, and the
                      level file opened by itself); every asset variant's levels compose on their own
  collision           every collider (PhysicsCollisionAPI) of every level authors physics:collisionEnabled: true on
                      its root's default level, false elsewhere and on point-instanced stand-ins
  lod_viewpoints      UsdLod's heuristic queries at every viewpoint and flight camera select exactly the cell
                      levels the manifest says that viewpoint should see
  streaming           every cell root has a streaming-domain heuristic with imaging <= load <= unload radius per
                      boundary; each level is authored with the arc the config asks for (payload / reference); at
                      every viewpoint and flight camera the drawn levels are a subset of what the streamer wants
  terrain_seams       neighbouring terrain edges at every pair of levels: shared vertices identical at the same
                      level, gaps covered by skirts across levels
  manifest_counts     the manifest's per-level counts recomputed from the stage
"""

import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdLod, UsdPhysics, UsdUtils

from . import lodselect as ls
from .cells import cell_level_path, root_path


class Report:
    def __init__(self):
        self.checks = {}

    def add(self, name, ok, detail=None, **info):
        c = self.checks.setdefault(name, dict(ok=True, failures=[], info={}))
        if not ok:
            c["ok"] = False
            if detail is not None and len(c["failures"]) < 200:
                c["failures"].append(detail)
        c["info"].update(info)

    def count(self, name, key, n=1):
        c = self.checks.setdefault(name, dict(ok=True, failures=[], info={}))
        c["info"][key] = c["info"].get(key, 0) + n


def _all_files(out):
    files = []
    for ext in ("usda", "usdc"):
        files += glob.glob(os.path.join(out, "**", f"*.{ext}"), recursive=True)
    return sorted(f for f in files if "/stats/" not in f)


_UNLOADED_CHECK = """
import sys
from pxr import Usd, UsdValidation
st = Usd.Stage.Open(sys.argv[1], Usd.Stage.LoadNone)
errs = UsdValidation.ValidationContext(UsdValidation.ValidationRegistry().GetOrLoadAllValidators()).Validate(st)
for e in errs:
    print(f"{e.GetType()} {e.GetName()}: {e.GetMessage()}")
sys.exit(1 if errs else 0)
"""


def check_usdchecker(cfg, rep, files):
    def run(f):
        skip = "/regions/" in f or f.endswith("root.usda")
        cmd = ["usdchecker"] + (["-s"] if skip else []) + [f]
        if f in aggregates:
            # usdchecker always loads every payload: with 8-level assets a region's full level needs ~5 GB and
            # the root >12 GB. Same validators on the unloaded stage; the payloaded tiles are checked file by file.
            cmd = [sys.executable, "-c", _UNLOADED_CHECK, f]
        r = subprocess.run(cmd, capture_output=True, text=True)
        txt = r.stdout + r.stderr
        warn = [l for l in txt.splitlines() if "Warning" in l or "Error" in l or "Fail" in l]
        return f, r.returncode, warn
    aggregates = {f for f in files if os.path.realpath(f) == os.path.realpath(root_path(cfg))
                  or ("/regions/" in f and "/tiles/" not in f and f.endswith("_full.usda"))}
    # A tile's full level still reaches ~0.8 GB under usdchecker; 8 workers stay well inside run.sh's cap.
    with ThreadPoolExecutor(8) as ex:
        results = list(ex.map(run, files))
    for f, code, warn in results:
        rel = os.path.relpath(f, cfg.out)
        rep.add("usdchecker", code == 0, f"{rel}: exit {code} {warn[:5]}")
        rep.count("usdchecker", "files")
        if warn:
            rep.add("usdchecker", True, None, **{f"warnings:{rel}": warn[:5]})


def check_conventions(cfg, rep, files):
    world_dir = os.path.realpath(cfg.out)
    referenced = set()
    for f in files:
        layer = Sdf.Layer.FindOrOpen(f)
        rel = os.path.relpath(f, cfg.out)
        mpu = layer.pseudoRoot.GetInfo("metersPerUnit") if layer.pseudoRoot.HasInfo("metersPerUnit") else None
        up = layer.pseudoRoot.GetInfo("upAxis") if layer.pseudoRoot.HasInfo("upAxis") else None
        rep.add("conventions", mpu == 1.0 and up == "Z", f"{rel}: metersPerUnit={mpu} upAxis={up}")
        subs, refs, pays = UsdUtils.ExtractExternalReferences(f)
        for kind, lst in (("sublayer", subs), ("reference", refs), ("payload", pays)):
            for a in lst:
                if not (a.startswith("./") or a.startswith("../")):
                    rep.add("conventions", False, f"{rel}: {kind} path not relative: {a}")
                target = os.path.realpath(os.path.join(os.path.dirname(f), a))
                rep.add("composition", os.path.exists(target), f"{rel}: unresolved {kind} {a}")
                rep.add("conventions", target.startswith(world_dir), f"{rel}: {kind} outside world/: {a}")
                if kind != "sublayer" and target.endswith((".usd", ".usda", ".usdc")):
                    referenced.add(target)
                rep.count("composition", f"{kind}s")
        # Texture and other asset-valued attributes.
        def visit(path):
            spec = layer.GetObjectAtPath(path)
            if isinstance(spec, Sdf.AttributeSpec) and spec.typeName in (Sdf.ValueTypeNames.Asset,):
                v = spec.default
                if v is not None and v.path:
                    target = os.path.realpath(os.path.join(os.path.dirname(f), v.path))
                    rep.add("conventions", v.path.startswith("."), f"{rel}: asset path not relative {v.path}")
                    rep.add("composition", os.path.exists(target), f"{rel}: missing asset {v.path}")
                    rep.count("composition", "asset_paths")
        layer.Traverse(layer.pseudoRoot.path, visit)
    for t in referenced:
        layer = Sdf.Layer.FindOrOpen(t)
        if layer is not None:
            rep.add("conventions", bool(layer.defaultPrim), f"{os.path.relpath(t, cfg.out)}: no defaultPrim")
    rep.add("conventions", True, None, layers=len(files), referenced_files=len(referenced))


def _bbox_cache():
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                      UsdGeom.Tokens.proxy], useExtentsHint=False,
                             ignoreVisibility=True)


def _contains(outer, inner, tol=0.05):
    if inner.IsEmpty():
        return True
    return all(outer.GetMin()[i] - tol <= inner.GetMin()[i] and inner.GetMax()[i] <= outer.GetMax()[i] + tol
               for i in range(3))


def _range(v):
    return Gf.Range3d(Gf.Vec3d(*v[0]), Gf.Vec3d(*v[1]))


def check_stage(cfg, rep, man):
    """Composition, payload bounds, gprim extents and LOD structure on the loaded stage, one region at a time.

    The stage opens unloaded; everything outside the regions is checked as composed, then each region is loaded
    with all its payloads, checked and unloaded again, so memory stays at one region rather than the world."""
    st = Usd.Stage.Open(root_path(cfg), Usd.Stage.LoadNone)
    bc = _bbox_cache()
    pred = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    tally = dict(gprims=0, errors=0, cells=0, chunks=0, assets=0)
    asset_keys = set()
    regions_scope = st.GetPrimAtPath("/World/Regions")
    regions = [p for p in regions_scope.GetChildren() if p.HasAPI(UsdLod.RootAPI)] if regions_scope else []

    def scan(it):
        lod_roots = []
        for p in it:
            if p.IsA(UsdGeom.Gprim):
                tally["gprims"] += 1
                if not p.IsInstanceProxy():
                    ok = UsdGeom.Boundable(p).GetExtentAttr().HasAuthoredValue()
                    rep.add("gprim_extents", ok, f"{p.GetPath()}: no extent")
            if p.HasAPI(UsdLod.RootAPI):
                lod_roots.append(p)
            if p.HasAuthoredPayloads():
                hint = UsdGeom.ModelAPI(p).GetExtentsHint()
                ok = hint is not None and len(hint) == 2
                rep.add("payload_bounds", ok, f"{p.GetPath()}: payload prim without extentsHint")
                if ok:
                    content = bc.ComputeUntransformedBound(p).ComputeAlignedRange()
                    inside = _contains(_range(hint), content, 0.1)
                    rep.add("payload_bounds", inside, f"{p.GetPath()}: content {content} outside extentsHint {list(hint)}")
                rep.count("payload_bounds", "payload_prims")
            elif p.IsModel() and not p.IsInstanceProxy() and Usd.ModelAPI(p).GetKind() in ("group", "assembly", "component") \
                    and not p.IsInstance():
                if p.GetPath().pathString.startswith("/World"):
                    hint = UsdGeom.ModelAPI(p).GetExtentsHint()
                    rep.add("gprim_extents", hint is not None and len(hint) == 2, f"{p.GetPath()}: model without extentsHint")
        for root in lod_roots:
            if root.IsInstanceProxy():
                # One check per asset and variant: prototypes are re-created on every load, so key by what the
                # instance (the placement) references and selects rather than by prototype path.
                inst = root.GetParent()
                while inst and not inst.IsInstance():
                    inst = inst.GetParent()
                key = (str(inst.GetAssetInfo().get("name")) if inst else str(root.GetPath()),
                       str(sorted(inst.GetVariantSets().GetAllVariantSelections().items())) if inst else "")
                if key in asset_keys:
                    continue
                asset_keys.add(key)
                tally["assets"] += 1
            elif root.GetCustomDataByKey("ground"):
                tally["chunks"] += 1
            else:
                tally["cells"] += 1
            _check_lod_root(rep, root, bc)

    def errors(where):
        errs = st.GetCompositionErrors() if hasattr(st, "GetCompositionErrors") else []
        for e in errs[:20]:
            rep.add("composition", False, f"{where}: {e.GetErrorMessage() if hasattr(e, 'GetErrorMessage') else e}")
        tally["errors"] += len(errs)

    # Everything outside the regions (environment, cameras, the regions' own roots and heuristics).
    scan(p for p in Usd.PrimRange.Stage(st, pred) if not any(p.GetPath().HasPrefix(r.GetPath()) for r in regions))
    errors("unloaded stage")
    for r in regions:
        st.Load(r.GetPath())
        bc.Clear()
        errors(r.GetName())
        scan(iter(Usd.PrimRange(r, pred)))
        st.Unload(r.GetPath())
    rep.add("composition", tally["errors"] == 0, None, composition_errors=tally["errors"], regions_loaded=len(regions))
    rep.add("gprim_extents", True, None, gprims=tally["gprims"])
    rep.add("lod_structure", True, None, cell_roots=tally["cells"], ground_chunks=tally["chunks"],
            asset_prototypes=tally["assets"])
    return st


def _check_lod_root(rep, root, bc):
    path = root.GetPath()
    lod = UsdLod.RootAPI(root)
    levels = ls.lod_levels(root)
    n = len(levels)
    d = lod.GetLodDefaultIndexAttr().Get()
    rep.add("lod_structure", n >= 2, f"{path}: {n} levels")
    rep.add("lod_structure", d is not None and 0 <= d < n, f"{path}: default index {d} of {n}")
    vis = [UsdGeom.Imageable(l).GetVisibilityAttr().Get() for l in levels]
    want = ["inherited" if i == d else "invisible" for i in range(n)]
    rep.add("lod_structure", vis == want, f"{path}: level visibility {vis}, expected {want}")
    rel = lod.GetLodHeuristicsRel()
    targets = rel.GetForwardedTargets()
    rep.add("lod_structure", len(targets) > 0, f"{path}: no heuristics")
    stage = root.GetStage()
    content = Gf.Range3d()
    for l in levels:
        b = bc.ComputeRelativeBound(l, root).ComputeAlignedRange()
        content.UnionWith(b)
    for t in targets:
        hp = stage.GetPrimAtPath(t)
        if not hp:
            rep.add("lod_structure", False, f"{path}: heuristic target {t} missing")
            continue
        dom = UsdLod.Heuristic(hp).GetLodDomainAttr().Get()
        rep.add("lod_structure", dom in ("imaging", "physics", "audio", ls.STREAMING), f"{t}: domain {dom}")
        if hp.IsA(UsdLod.DistanceHeuristic):
            h = UsdLod.DistanceHeuristic(hp)
            q = h.CreateDistanceHeuristicQuery()
            th = list(h.GetThresholdsAttr().Get() or [])
            ok_order = all(a <= b for a, b in zip(th, th[1:]))
            bound = q.extent
            finite = np.all(np.isfinite(list(q.center)))
            if not bound.IsEmpty():
                finite = finite and np.all(np.isfinite(list(bound.GetMin()) + list(bound.GetMax())))
                rep.add("lod_structure", _contains(bound, content, 0.1),
                        f"{t}: boundingVolume {bound} does not contain content {content}")
            rep.add("lod_structure", finite, f"{t}: non-finite bound")
        else:
            h = UsdLod.ScreenSizeHeuristic(hp)
            q = h.CreateScreenSizeHeuristicQuery()
            th = list(h.GetThresholdsAttr().Get() or [])
            ok_order = all(a >= b for a, b in zip(th, th[1:]))
            bound = q.extent
            ok = (not bound.IsEmpty()) and np.all(np.isfinite(list(bound.GetMin()) + list(bound.GetMax())))
            rep.add("lod_structure", ok, f"{t}: empty or non-finite extent {bound}")
            if ok:
                rep.add("lod_structure", _contains(bound, content, 0.1), f"{t}: extent {bound} does not contain {content}")
        rep.add("lod_structure", len(th) == n - 1, f"{t}: {len(th)} thresholds for {n} levels")
        rep.add("lod_structure", ok_order, f"{t}: thresholds out of order {th}")
        bt = list(h.GetBlendThresholdsAttr().Get() or [])
        rep.add("lod_structure", len(bt) in (0, n - 1), f"{t}: {len(bt)} blend thresholds")
        rep.count("lod_structure", "heuristics")
    _check_streaming(rep, root)


def _check_streaming(rep, root):
    """Cell roots: streaming load radius >= imaging distance threshold <= unload radius, per boundary, so a level
    is resident before it is drawn and released only after it is no longer drawn."""
    if root.IsInstanceProxy():
        return
    h, load, unload = ls.streaming_heuristic(root)
    if h is None:
        rep.add("streaming", False, f"{root.GetPath()}: no streaming heuristic")
        return
    _, img = ls._pick_heuristic(root, "distance", UsdLod.Tokens.imaging)
    th = list(img.GetThresholdsAttr().Get() or []) if img else []
    n = len(ls.lod_levels(root))
    rep.add("streaming", len(load) == n - 1 and len(unload) == n - 1,
            f"{root.GetPath()}: {len(load)} load / {len(unload)} unload thresholds for {n} levels")
    rep.add("streaming", all(t <= l <= u for t, l, u in zip(th, load, unload)),
            f"{root.GetPath()}: need imaging <= load <= unload, got {th} / {load} / {unload}")
    rep.count("streaming", "ground_chunks" if root.GetCustomDataByKey("ground") else "cell_roots")


def _check_collision(rep, level_prim):
    """Every collider of a LOD level authors physics:collisionEnabled (no reliance on the schema fallback): true
    on the root's default level, false elsewhere. Point-instanced stand-ins (prototypes) are always false. Nested
    LOD roots (tiles, ground chunks, asset instances) are checked as their own roots."""
    root = level_prim.GetParent()
    is_default = ls.lod_levels(root).index(level_prim) == UsdLod.RootAPI(root).GetLodDefaultIndexAttr().Get()
    it = iter(Usd.PrimRange(level_prim))
    for p in it:
        if p != level_prim and p.HasAPI(UsdLod.RootAPI):
            it.PruneChildren()
            continue
        if p.IsA(UsdGeom.PointInstancer):
            it.PruneChildren()
            for proto in Usd.PrimRange(p):
                if proto.HasAPI(UsdPhysics.CollisionAPI):
                    a = proto.GetAttribute("physics:collisionEnabled")
                    rep.add("collision", a.HasAuthoredValue() and a.Get() is False,
                            f"{proto.GetPath()}: instanced stand-in must author collisionEnabled = false")
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI):
            a = p.GetAttribute("physics:collisionEnabled")
            val = a.Get() if a.HasAuthoredValue() else None
            rep.add("collision", val == is_default,
                    f"{p.GetPath()}: collisionEnabled {val}, expected {is_default} ({'default' if is_default else 'non-default'} level)")
            rep.count("collision", "default_colliders" if is_default else "disabled_colliders")


def _check_ground_chunks(cfg, rep, st, level_prim, cell):
    """A loaded tile full level: every ground chunk's `fine` payload loads on its own (only that payload) and its
    `base` level is inline; the tile's ground file opens by itself."""
    from . import ground as gr
    for c in gr.chunks(cfg, cell):
        root = level_prim.GetChild(c.name)
        ok = root.IsValid() and root.HasAPI(UsdLod.RootAPI)
        rep.add("levels_standalone", ok, f"{level_prim.GetPath()}/{c.name}: ground chunk root missing")
        if not ok:
            continue
        fine, base = root.GetChild("fine"), root.GetChild("base")
        rep.add("streaming", fine.HasAuthoredPayloads() and not base.HasAuthoredPayloads()
                and not base.HasAuthoredReferences(), f"{root.GetPath()}: expected fine payload, base inline")
        st.Load(fine.GetPath(), Usd.LoadWithoutDescendants)
        meshes = [p for p in fine.GetChildren() if p.IsA(UsdGeom.Mesh)]
        rep.add("levels_standalone", fine.IsLoaded() and len(meshes) == 2 and UsdShade_binding(meshes[0]),
                f"{fine.GetPath()}: did not load on its own with bound ground and skirt")
        rep.add("levels_standalone", any(p.IsA(UsdGeom.Mesh) for p in base.GetChildren()),
                f"{base.GetPath()}: no inline base ground")
        _check_collision(rep, base)
        _check_collision(rep, fine)
        st.Unload(fine.GetPath())
        rep.count("levels_standalone", "ground_levels", 2)
    f = gr.ground_path(cfg, cell)
    s2 = Usd.Stage.Open(f)
    dp = s2.GetDefaultPrim()
    rep.add("levels_standalone", dp.IsValid() and len(dp.GetChildren()) == len(gr.chunks(cfg, cell)),
            f"{f}: empty, no default prim or missing chunks")


def check_levels_standalone(cfg, rep, man):
    from .assets import library as L
    cells = []

    def walk(e):
        cells.append(e)
        for k in e.get("children", []):
            walk(k)
    for e in man["hierarchy"]:
        walk(e)
    st = Usd.Stage.Open(root_path(cfg), Usd.Stage.LoadNone)
    for e in cells:
        cell = cfg.cell_by_id(e["id"])
        chain = cell.chain
        root_prim = "/World/Regions/" + "/full/".join(c.name for c in chain)
        for ln in e["levels"]:
            # (a) In the root stage, load only this level's payload (and the ancestors' full levels it lives in).
            # Referenced levels (a region's full level) are composed without any load.
            st.Unload("/World")
            anc = "/World/Regions"
            for c in chain[:-1]:
                anc = f"{anc}/{c.name}/full"
                if st.GetPrimAtPath(anc).HasAuthoredPayloads():
                    st.Load(anc, Usd.LoadWithoutDescendants)
            lp = f"{root_prim}/{ln}"
            prim = st.GetPrimAtPath(lp)
            arc = "payload" if prim.HasAuthoredPayloads() else "reference"
            want = cell.level.full_arc if (ln == "full" and cfg.children(cell)) else "payload"
            rep.add("streaming", arc == want, f"{lp}: authored as {arc}, config says {want}")
            if arc == "payload":
                st.Load(lp, Usd.LoadWithoutDescendants)
            ok = prim.IsValid() and prim.IsLoaded() and len(prim.GetChildren()) > 0
            rep.add("levels_standalone", ok, f"{lp}: did not load on its own")
            if ok:
                _check_collision(rep, prim)
            if ok and ln == "full" and not cfg.children(cell):
                _check_ground_chunks(cfg, rep, st, prim, cell)
            loaded = [str(p) for p in st.GetLoadSet()]
            # (b) The level file by itself.
            f = cell_level_path(cfg, cell, ln)
            s2 = Usd.Stage.Open(f)
            dp = s2.GetDefaultPrim()
            rep.add("levels_standalone", dp.IsValid() and len(dp.GetChildren()) > 0, f"{f}: empty or no default prim")
            rep.count("levels_standalone", "cell_levels")
    # Asset variants: every level of every variant composes and has geometry.
    for spec in L.ASSETS:
        s = Usd.Stage.Open(spec.interface_path(cfg.out))
        root = s.GetDefaultPrim()
        vs = root.GetVariantSets().GetVariantSet(spec.variant_set)
        for v in vs.GetVariantNames():
            vs.SetVariantSelection(v)
            geom = s.GetPrimAtPath(root.GetPath().AppendChild("Geom"))
            lv = ls.lod_levels(geom)
            for l in lv:
                meshes = [p for p in Usd.PrimRange(l) if p.IsA(UsdGeom.Mesh)]
                rep.add("levels_standalone", len(meshes) > 0, f"{spec.name}:{v}:{l.GetName()} has no geometry")
                for m in meshes:
                    mb = UsdShade_binding(m)
                    rep.add("levels_standalone", mb, f"{m.GetPath()} ({spec.name}:{v}) has no material binding")
                _check_collision(rep, l)
                rep.count("levels_standalone", "asset_levels")


def UsdShade_binding(prim):
    from pxr import UsdShade
    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    return bool(mat)


def check_lod_viewpoints(cfg, rep, man):
    st = Usd.Stage.Open(root_path(cfg), Usd.Stage.LoadNone)
    rows = {}
    entries = [(k, v) for k, v in man["viewpoints"].items()] + [(f["camera"].split("/")[-1], f) for f in man["flight_path"]]
    for name, v in entries:
        st.Unload("/World")
        fr, _ = ls.camera_frustum(st, v["camera"])
        sel = ls.Selector(st, cell_heuristic="distance").select(fr, load=True, assets=(name in man["viewpoints"]))
        got = {}
        for path, c in sel.cells.items():
            cid = st.GetPrimAtPath(path).GetCustomDataByKey("cell")
            got[cid] = c.levels[c.index]
        exp = {k: e["level"] for k, e in v["expected_cells"].items()}
        # Every drawn payload level is resident under the streaming policy (fresh streamer, no hysteresis state).
        _, _, wanted = ls.Streamer(st).update(fr.GetPosition())
        drawn = {f"{p}/{c.levels[c.index]}" for p, c in sel.cells.items()}
        drawn = {p for p in drawn if st.GetPrimAtPath(p).HasAuthoredPayloads()}
        rep.add("streaming", drawn <= wanted, f"{name}: drawn but not wanted by streaming: {sorted(drawn - wanted)[:5]}")
        ok = got == exp
        diff = {k: (exp.get(k), got.get(k)) for k in set(exp) | set(got) if exp.get(k) != got.get(k)}
        rep.add("lod_viewpoints", ok, f"{name}: mismatches {diff}")
        # Screen-size heuristics (alternative engines): record where they agree with the distance heuristics.
        st.Unload("/World")
        sel2 = ls.Selector(st, cell_heuristic="screen").select(fr, load=True, assets=False)
        got2 = {st.GetPrimAtPath(p).GetCustomDataByKey("cell"): c.levels[c.index] for p, c in sel2.cells.items()}
        agree = sum(1 for k in got2 if got2.get(k) == got.get(k))
        rows[name] = dict(full_cells=sorted(k for k, l in got.items() if l == "full"),
                          fine_ground_chunks=sum(1 for l in got.values() if l == "fine"),
                          cells_evaluated=len(got), screen_size_agreement=f"{agree}/{len(got2)}",
                          asset_levels=sel.asset_histogram() if sel.assets else None)
    rep.add("lod_viewpoints", True, None, viewpoints=rows)


def _edge_profile(mesh_pts, axis, value, lo, hi, tol=1e-3):
    """Points of a ground mesh lying on the line coord[axis] == value, sorted along the other axis."""
    m = np.abs(mesh_pts[:, axis] - value) < tol
    p = mesh_pts[m]
    other = 1 - axis
    p = p[(p[:, other] >= lo - tol) & (p[:, other] <= hi + tol)]
    order = np.argsort(p[:, other])
    return p[order][:, other], p[order][:, 2]


def _edge_profile_depth(pts, depth, axis, value, lo, hi, tol=1e-3):
    """_edge_profile with a skirt depth per point (`depth` a scalar or one value per point)."""
    d = np.broadcast_to(np.asarray(depth, dtype=np.float64), (len(pts),))
    m = np.abs(pts[:, axis] - value) < tol
    other = 1 - axis
    m &= (pts[:, other] >= lo - tol) & (pts[:, other] <= hi + tol)
    p, dd = pts[m], d[m]
    order = np.lexsort((dd, p[:, other]))   # duplicates at chunk corners: the shallower skirt first
    return p[order][:, other], p[order][:, 2], dd[order]


def _depth_at(t, s, d):
    i = np.searchsorted(s, t, side="right")
    return np.minimum(d[np.clip(i - 1, 0, len(d) - 1)], d[np.clip(i, 0, len(d) - 1)])


def check_terrain_seams(cfg, rep, man):
    """Edges of neighbouring cells, at every pair of levels, read back from the written meshes; then the ground
    chunks' fine and base levels against each other."""
    from . import ground as gr
    cache = {}

    def meshes(s, prim, origin):
        """(ground points in world space, skirt depth) of a ground + skirt pair under `prim`."""
        g = UsdGeom.Mesh(prim.GetChild("ground"))
        sk = UsdGeom.Mesh(prim.GetChild("skirt"))
        p = np.asarray(g.GetPointsAttr().Get(), dtype=np.float64) + [origin[0], origin[1], 0]
        sp = np.asarray(sk.GetPointsAttr().Get(), dtype=np.float64)
        n = len(sp) // 2
        return p, float(np.min(sp[:n, 2] - sp[n:, 2]))

    def ground(cell, ln):
        key = (cell.id, ln)
        if key not in cache:
            f = cell_level_path(cfg, cell, ln)
            s = Usd.Stage.Open(f, Usd.Stage.LoadNone)
            dp = s.GetDefaultPrim()
            if ln == "full" and not cfg.children(cell):
                # The finest cells' full level: the base levels of its ground chunks, each point with its own
                # chunk's skirt depth.
                parts = [meshes(s, dp.GetChild(c.name).GetChild("base"), cell.center) for c in gr.chunks(cfg, cell)]
                cache[key] = (np.vstack([p for p, _ in parts]),
                              np.concatenate([np.full(len(p), d) for p, d in parts]))
            else:
                cache[key] = meshes(s, dp.GetChild("Terrain"), cell.center)
        return cache[key]

    present = [cfg.cell_by_id(e["id"]) for e in man["hierarchy"]]
    cells = []
    for r in present:
        cells += cfg.leaf_cells(r)
    cells += present
    by_level = {}
    for c in cells:
        by_level.setdefault(c.level.index, []).append(c)
    pairs = 0
    worst_same = 0.0
    worst_ratio = 0.0
    # Neighbouring pairs: same-level neighbours, and finest cells against neighbouring coarser cells.
    def neighbours(a, b):
        ax0, ay0, ax1, ay1 = a.bounds2d
        bx0, by0, bx1, by1 = b.bounds2d
        if abs(ax1 - bx0) < 1e-6 or abs(bx1 - ax0) < 1e-6:
            lo, hi = max(ay0, by0), min(ay1, by1)
            if hi - lo > 1e-6:
                return 0, (ax1 if abs(ax1 - bx0) < 1e-6 else ax0), lo, hi
        if abs(ay1 - by0) < 1e-6 or abs(by1 - ay0) < 1e-6:
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            if hi - lo > 1e-6:
                return 1, (ay1 if abs(ay1 - by0) < 1e-6 else ay0), lo, hi
        return None
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            if a.chain[0].id == b.chain[0].id and a.level.index != b.level.index:
                continue  # a region and its own tiles are never shown side by side
            nb = neighbours(a, b)
            if nb is None:
                continue
            axis, value, lo, hi = nb
            for la in a.level.lod_names:
                if la == "full" and a.level.index < len(cfg.levels) - 1:
                    continue
                for lb in b.level.lod_names:
                    if lb == "full" and b.level.index < len(cfg.levels) - 1:
                        continue
                    pa, da = ground(a, la)
                    pb, db = ground(b, lb)
                    sa, za, dpa = _edge_profile_depth(pa, da, axis, value, lo, hi)
                    sb, zb, dpb = _edge_profile_depth(pb, db, axis, value, lo, hi)
                    t = np.union1d(sa, sb)
                    ia = np.interp(t, sa, za)
                    ib = np.interp(t, sb, zb)
                    # Skirt depth along the edge (per chunk for a tile's full level): at each point, the shallower
                    # of the two samples around it (conservative at chunk corners).
                    da, db = _depth_at(t, sa, dpa), _depth_at(t, sb, dpb)
                    gap = np.abs(ia - ib)
                    # The higher edge's skirt must reach the lower edge.
                    need = np.where(ia > ib, gap / da, gap / db)
                    worst_ratio = max(worst_ratio, float(need.max()))
                    pairs += 1
                    if a.level.index == b.level.index and la == lb:
                        common = np.intersect1d(np.round(sa, 4), np.round(sb, 4))
                        same = float(np.max(np.abs(np.interp(common, sa, za) - np.interp(common, sb, zb)))) if len(common) else 0
                        worst_same = max(worst_same, same)
                        rep.add("terrain_seams", same < 1e-4, f"{a.id}:{la} / {b.id}:{lb}: shared vertices differ by {same}")
                    rep.add("terrain_seams", need.max() <= 1.0,
                            f"{a.id}:{la} / {b.id}:{lb}: gap {gap.max():.2f} m exceeds skirt "
                            f"({np.min(da):.2f}/{np.min(db):.2f} m)")
    rep.add("terrain_seams", pairs > 0, "no neighbouring pairs found", edge_pairs_checked=pairs,
            worst_shared_vertex_difference_m=worst_same, worst_gap_to_skirt_ratio=round(worst_ratio, 3))
    cache.clear()
    _check_chunk_seams(cfg, rep, [c for c in cells if not cfg.children(c)])


def _check_chunk_seams(cfg, rep, tiles):
    """Ground chunks: every chunk edge against its right and upper neighbour (inside the tile or across the tile
    border), at fine/fine, fine/base, base/fine and base/base. A fine chunk only ever neighbours chunks of full tiles
    (its 50 m radius lies far inside the tile's 220 m first boundary), so these are all the pairs that can show."""
    from functools import lru_cache
    from . import ground as gr

    def pts_depth(mesh_parent, origin):
        g = UsdGeom.Mesh(mesh_parent.GetChild("ground"))
        sk = UsdGeom.Mesh(mesh_parent.GetChild("skirt"))
        p = np.asarray(g.GetPointsAttr().Get(), dtype=np.float64) + [origin[0], origin[1], 0]
        sp = np.asarray(sk.GetPointsAttr().Get(), dtype=np.float64)
        n = len(sp) // 2
        return p, float(np.min(sp[:n, 2] - sp[n:, 2]))

    by_id = {t.id: t for t in tiles}

    @lru_cache(maxsize=96)
    def tile_chunks(tid):
        t = by_id[tid]
        s = Usd.Stage.Open(cell_level_path(cfg, t, "full"), Usd.Stage.LoadNone)
        dp = s.GetDefaultPrim()
        g = Usd.Stage.Open(gr.ground_path(cfg, t))
        gp = g.GetDefaultPrim()
        out = {}
        for c in gr.chunks(cfg, t):
            out[c.name] = dict(base=pts_depth(dp.GetChild(c.name).GetChild("base"), t.center),
                               fine=pts_depth(gp.GetChild(c.name), t.center), cell=c)
        return out

    size = cfg.levels[-1].size
    key = {(round(t.x0 / size), round(t.y0 / size)): t.id for t in tiles}
    pairs = 0
    worst_ratio = 0.0
    worst_same = 0.0
    for t in sorted(tiles, key=lambda t: (t.y0, t.x0)):
        tc = tile_chunks(t.id)
        gx, gy = round(t.x0 / size), round(t.y0 / size)
        right = key.get((gx + 1, gy))
        up = key.get((gx, gy + 1))
        nb = {}
        for tid in (t.id, right, up):
            if tid:
                for cname, d in tile_chunks(tid).items():
                    c = d["cell"]
                    nb[(round(c.x0 / c.level.size), round(c.y0 / c.level.size))] = d
        for d in tc.values():
            c = d["cell"]
            cx, cy = round(c.x0 / c.level.size), round(c.y0 / c.level.size)
            for axis, other in ((0, nb.get((cx + 1, cy))), (1, nb.get((cx, cy + 1)))):
                if other is None:
                    continue
                x0, y0, x1, y1 = c.bounds2d
                value, lo, hi = (x1, y0, y1) if axis == 0 else (y1, x0, x1)
                for la in ("fine", "base"):
                    for lb in ("fine", "base"):
                        pa, da = d[la]
                        pb, db = other[lb]
                        sa, za = _edge_profile(pa, axis, value, lo, hi, tol=1e-2)
                        sb, zb = _edge_profile(pb, axis, value, lo, hi, tol=1e-2)
                        tt = np.union1d(sa, sb)
                        ia = np.interp(tt, sa, za)
                        ib = np.interp(tt, sb, zb)
                        gap = np.abs(ia - ib)
                        need = np.where(ia > ib, gap / da, gap / db)
                        worst_ratio = max(worst_ratio, float(need.max()))
                        pairs += 1
                        if la == lb:
                            same = float(gap.max())
                            worst_same = max(worst_same, same)
                            rep.add("terrain_seams", same < 1e-3,
                                    f"{c.id}:{la} / {other['cell'].id}:{lb}: shared vertices differ by {same}")
                        rep.add("terrain_seams", need.max() <= 1.0,
                                f"{c.id}:{la} / {other['cell'].id}:{lb}: gap {gap.max():.2f} m exceeds skirt "
                                f"({da:.2f}/{db:.2f} m)")
    rep.add("terrain_seams", pairs > 0, "no ground chunk pairs found", chunk_edge_pairs_checked=pairs,
            chunk_worst_shared_vertex_difference_m=round(worst_same, 5),
            chunk_worst_gap_to_skirt_ratio=round(worst_ratio, 3))


def check_manifest_counts(cfg, rep, man):
    def walk(e):
        cell = cfg.cell_by_id(e["id"])
        for ln, s in e["levels"].items():
            f = cell_level_path(cfg, cell, ln)
            st = Usd.Stage.Open(f, Usd.Stage.LoadNone)  # the level's own content, not nested levels
            inst = 0
            pts = 0
            tris = 0
            meshes = 0
            assets = set()
            for p in Usd.PrimRange(st.GetPseudoRoot()):
                if p.IsInstance():
                    inst += 1
                    for prim_spec in p.GetPrimStack():
                        for r in prim_spec.referenceList.prependedItems:
                            assets.add(os.path.splitext(os.path.basename(r.assetPath))[0])
                elif p.IsA(UsdGeom.PointInstancer):
                    pts += len(UsdGeom.PointInstancer(p).GetProtoIndicesAttr().Get() or [])
                    # Coarse levels point-instance asset levels (prototypes reference <asset>_geom files).
                    for t in UsdGeom.PointInstancer(p).GetPrototypesRel().GetTargets():
                        for prim_spec in st.GetPrimAtPath(t).GetPrimStack():
                            for r in prim_spec.referenceList.prependedItems:
                                stem = os.path.splitext(os.path.basename(r.assetPath))[0]
                                if stem.endswith("_geom"):
                                    assets.add(stem[:-5])
                elif p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).GetPurposeAttr().Get() != UsdGeom.Tokens.guide \
                        and not any(a.IsA(UsdGeom.PointInstancer) for a in _ancestors(p)):
                    c = np.asarray(UsdGeom.Mesh(p).GetFaceVertexCountsAttr().Get())
                    tris += int(np.sum(c - 2))
                    meshes += 1
            for key, got in (("instances", inst), ("point_instances", pts), ("triangles", tris), ("meshes", meshes)):
                rep.add("manifest_counts", s.get(key) == got, f"{e['id']}:{ln} {key}: manifest {s.get(key)} stage {got}")
            if s.get("assets"):
                rep.add("manifest_counts", sorted(assets) == sorted(s["assets"]),
                        f"{e['id']}:{ln} assets: manifest {s['assets']} stage {sorted(assets)}")
            rep.count("manifest_counts", "levels")
        for k in e.get("children", []):
            walk(k)
    for e in man["hierarchy"]:
        walk(e)


def _ancestors(p):
    q = p.GetParent()
    while q and not q.IsPseudoRoot():
        yield q
        q = q.GetParent()


def run_all(cfg, quick=False):
    t0 = time.time()
    man = json.load(open(os.path.join(cfg.out, "manifest.json")))
    rep = Report()
    files = _all_files(cfg.out)
    steps = [("conventions", lambda: check_conventions(cfg, rep, files)),
             ("stage", lambda: check_stage(cfg, rep, man)),
             ("levels_standalone", lambda: check_levels_standalone(cfg, rep, man)),
             ("lod_viewpoints", lambda: check_lod_viewpoints(cfg, rep, man)),
             ("terrain_seams", lambda: check_terrain_seams(cfg, rep, man)),
             ("manifest_counts", lambda: check_manifest_counts(cfg, rep, man))]
    if not quick:
        steps.insert(0, ("usdchecker", lambda: check_usdchecker(cfg, rep, files)))
    timings = {}
    for name, fn in steps:
        t = time.time()
        fn()
        timings[name] = round(time.time() - t, 1)
        print(f"[validate] {name} ({timings[name]}s)", flush=True)
    ok = all(c["ok"] for c in rep.checks.values())
    out = dict(ok=ok, seconds=round(time.time() - t0, 1), timings=timings, checks=rep.checks)
    with open(os.path.join(cfg.out, "validation.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    for name, c in rep.checks.items():
        status = "PASS" if c["ok"] else "FAIL"
        info = {k: v for k, v in c["info"].items() if not k.startswith("warnings:") and k != "viewpoints"}
        print(f"  {status:4s} {name:18s} {info}")
        for f in c["failures"][:8]:
            print(f"         - {f}")
    print(f"[validate] {'ALL PASSED' if ok else 'FAILURES'} in {out['seconds']}s -> world/validation.json")
    return ok
