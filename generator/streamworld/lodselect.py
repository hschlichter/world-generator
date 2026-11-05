"""LOD selection with UsdLod's own heuristic queries, as a LOD-aware consumer would do it.

Starting at the regions, every LOD root is evaluated from a camera with its heuristic (distance or screen size, in
the requested domain); the selected level's payload is loaded and its nested LOD roots (the LOD roots among the
level's children: tiles in a region, ground chunks in a tile) are evaluated in turn. Optionally, asset LOD roots
inside the loaded full levels (the ones inside instances) are evaluated too.

`Streamer` decides what should be resident, separately from what is drawn: it evaluates each cell root's heuristic
in the `streaming` domain (load radii, with `streaming:unloadThresholds` as hysteresis) and returns the payload
loads and unloads to issue. A cell's children are only considered while the cell wants its full level.

`apply_to_session` turns a selection into session-layer opinions (visibility of LOD levels, and for assets whose
selection differs from their default, de-instancing so a single instance can show another level). This is how the
usdview plugin and the preview renders show LOD with a renderer (Storm) that does not implement UsdLod itself.

This module has no dependency on the generator, so it can be copied next to the world and used on its own.
"""

from dataclasses import dataclass, field

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLod

# LOD levels are listed whether or not their payload is loaded.
_LEVEL_PRED = Usd.PrimIsActive & Usd.PrimIsDefined & ~Usd.PrimIsAbstract


def lod_levels(prim):
    """The LOD items of a root: its direct children in order (index 0 = finest)."""
    pred = Usd.TraverseInstanceProxies(_LEVEL_PRED) if prim.IsInstanceProxy() else _LEVEL_PRED
    return list(prim.GetFilteredChildren(pred))


@dataclass
class RootChoice:
    path: str
    index: int
    value: float
    levels: list
    heuristic: str
    metric: float
    is_cell: bool


STREAMING = "streaming"
UNLOAD_ATTR = "streaming:unloadThresholds"


@dataclass
class Selection:
    cells: dict = field(default_factory=dict)     # path -> RootChoice
    assets: dict = field(default_factory=dict)    # path -> RootChoice (only evaluated asset roots)
    loaded: list = field(default_factory=list)
    late: list = field(default_factory=list)      # roots drawn coarser than selected: the level was not resident

    def asset_histogram(self):
        h = {}
        for c in self.assets.values():
            h[c.index] = h.get(c.index, 0) + 1
        return dict(sorted(h.items()))


def camera_frustum(stage, camera_path, aspect=None):
    cam = UsdGeom.Camera(stage.GetPrimAtPath(camera_path)).GetCamera(Usd.TimeCode.Default())
    if aspect:
        cam.verticalAperture = cam.horizontalAperture / aspect
    return cam.frustum, cam


def _pick_heuristic(prim, kind, domain):
    """The first targeted heuristic of the preferred type in the domain (falls back to the other type)."""
    rel = UsdLod.RootAPI(prim).GetLodHeuristicsRel()
    targets = rel.GetForwardedTargets() if rel else []
    stage = prim.GetStage()
    found = {}
    for t in targets:
        hp = stage.GetPrimAtPath(t)
        if not hp:
            continue
        h = UsdLod.Heuristic(hp)
        if h.GetLodDomainAttr().Get() != domain:
            continue
        if hp.IsA(UsdLod.DistanceHeuristic):
            found.setdefault("distance", UsdLod.DistanceHeuristic(hp))
        elif hp.IsA(UsdLod.ScreenSizeHeuristic):
            found.setdefault("screen", UsdLod.ScreenSizeHeuristic(hp))
    if kind in found:
        return kind, found[kind]
    for k in ("distance", "screen"):
        if k in found:
            return k, found[k]
    return None, None


class Selector:
    def __init__(self, stage, domain=UsdLod.Tokens.imaging, cell_heuristic="distance", asset_heuristic="screen"):
        self.stage = stage
        self.domain = domain
        self.cell_heuristic = cell_heuristic
        self.asset_heuristic = asset_heuristic
        self._qcache = {}

    def _query(self, hkind, h):
        hp = h.GetPrim()
        key = hp.GetPrimInPrototype().GetPath() if hp.IsInstanceProxy() else hp.GetPath()
        q = self._qcache.get((hkind, key))
        if q is None:
            q = h.CreateDistanceHeuristicQuery() if hkind == "distance" else h.CreateScreenSizeHeuristicQuery()
            self._qcache[(hkind, key)] = q
        return q

    def evaluate(self, prim, xc, eye, frustum, prefer):
        lod = UsdLod.RootAPI(prim)
        levels = lod_levels(prim)
        hk, h = _pick_heuristic(prim, prefer, self.domain)
        l2w = xc.GetLocalToWorldTransform(prim)
        if h is None:
            idx = int(lod.GetLodDefaultIndexAttr().Get())
            return idx, float(idx), [c.GetName() for c in levels], "default", 0.0
        q = self._query(hk, h)
        if hk == "distance":
            metric = q.ComputeDistance(Gf.Vec3d(*eye), l2w)
            val = q.ComputeLOD(metric)
        else:
            metric = q.ComputeScreenSize(frustum, l2w)
            val = q.ComputeLOD(metric)
        idx = max(0, min(len(levels) - 1, int(round(val))))
        return idx, float(val), [c.GetName() for c in levels], hk, float(metric)

    def select(self, frustum, load=True, assets=True, asset_max_distance=None, regions_path="/World/Regions",
               eye=None):
        """Walk the cell hierarchy from the regions, loading selected levels. Returns a Selection.

        load=False: never load; a selected level whose payload is not resident falls back to the nearest coarser
        resident level (recorded in Selection.late), the way a streaming engine draws while a load is in flight.
        eye: camera position, when no frustum is given (distance heuristics only)."""
        stage = self.stage
        sel = Selection()
        eye = frustum.GetPosition() if frustum is not None else Gf.Vec3d(*eye)
        xc = UsdGeom.XformCache()
        top = stage.GetPrimAtPath(regions_path)
        queue = [p for p in top.GetChildren() if p.HasAPI(UsdLod.RootAPI)]
        leaf_levels = []
        while queue:
            root = queue.pop(0)
            idx, val, names, hk, metric = self.evaluate(root, xc, eye, frustum, self.cell_heuristic)
            levels = lod_levels(root)
            lvl = levels[idx]
            if load and lvl.HasAuthoredPayloads() and not lvl.IsLoaded():
                stage.Load(lvl.GetPath(), Usd.LoadWithoutDescendants)
                sel.loaded.append(str(lvl.GetPath()))
                xc.Clear()
            elif not load and not resident(lvl):
                coarser = [i for i in range(idx + 1, len(levels)) if resident(levels[i])]
                sel.late.append(str(root.GetPath()))
                if coarser:
                    idx = coarser[0]
            sel.cells[str(root.GetPath())] = RootChoice(str(root.GetPath()), idx, val, names, hk, metric, True)
            lvl = stage.GetPrimAtPath(root.GetPath().AppendChild(names[idx]))
            nested = [c for c in lvl.GetChildren() if c.HasAPI(UsdLod.RootAPI)]
            queue += nested
            if idx == 0 and lvl.IsLoaded():
                leaf_levels.append(lvl)
        if assets:
            for lvl in leaf_levels:
                it = iter(Usd.PrimRange(lvl, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
                for p in it:
                    if p.HasAPI(UsdLod.RootAPI):
                        it.PruneChildren()
                        if not p.IsInstanceProxy():
                            continue    # a nested cell or ground chunk: the walk above evaluates it
                        if asset_max_distance is not None:
                            t = xc.GetLocalToWorldTransform(p).ExtractTranslation()
                            if (t - eye).GetLength() > asset_max_distance:
                                continue
                        idx, val, names, hk, metric = self.evaluate(p, xc, eye, frustum, self.asset_heuristic)
                        sel.assets[str(p.GetPath())] = RootChoice(str(p.GetPath()), idx, val, names, hk, metric, False)
                    elif p.IsA(UsdGeom.PointInstancer) or p.IsA(UsdGeom.Gprim):
                        it.PruneChildren()
        return sel

    def unload_unselected(self, sel):
        """Unload payloads of levels that are not selected (streaming out)."""
        for path, c in sel.cells.items():
            root = self.stage.GetPrimAtPath(path)
            for i, lvl in enumerate(lod_levels(root)):
                if i != c.index and lvl.HasAuthoredPayloads() and lvl.IsLoaded():
                    self.stage.Unload(lvl.GetPath())


def resident(level_prim):
    """A LOD level is resident when its content is composed: a payload level once loaded, any other level always."""
    return not level_prim.HasAuthoredPayloads() or level_prim.IsLoaded()


def streaming_heuristic(prim, domain=STREAMING):
    """(DistanceHeuristic, load thresholds, unload thresholds) of a root in the streaming domain, or Nones."""
    _, h = _pick_heuristic(prim, "distance", domain)
    if h is None:
        return None, None, None
    load = list(h.GetThresholdsAttr().Get() or [])
    a = h.GetPrim().GetAttribute(UNLOAD_ATTR)
    unload = list(a.Get()) if a and a.HasAuthoredValue() else list(load)
    return h, load, unload


class Streamer:
    """Residency from the streaming domain, with hysteresis. Engine-independent: `update` returns what to load and
    unload, `apply` performs it on a USD stage (a streaming engine would issue asynchronous requests instead).

    Per root, with d = distance from the eye to the root's bounding volume:
      s_load   = number of load thresholds <= d          (finest level the load radii ask for)
      s_unload = number of unload thresholds <= d
      s        = s_load when moving finer, else max(previous s, s_unload) capped at s_load   (hysteresis)
    Resident levels of the root: s and every coarser level (keep_coarser, the cheap fallback chain) or s alone.
    The root's children are visited only while s == 0 (its full level is wanted); otherwise all their levels are
    released.

    domain="imaging" without unload thresholds and keep_coarser=False reproduces load-what-is-drawn streaming.
    """

    def __init__(self, stage, domain=STREAMING, keep_coarser=True, regions_path="/World/Regions"):
        self.stage = stage
        self.domain = domain
        self.keep_coarser = keep_coarser
        self.regions_path = regions_path
        self.state = {}        # root path -> s
        self.wanted = set()    # payload level paths currently wanted
        self._q = {}

    def _root_s(self, root, xc, eye):
        key = str(root.GetPath())
        if key not in self._q:
            h, load, unload = streaming_heuristic(root, self.domain)
            if self.domain != STREAMING:
                unload = list(load)
            self._q[key] = (h.CreateDistanceHeuristicQuery() if h else None, load, unload)
        q, load, unload = self._q[key]
        n = len(lod_levels(root))
        if q is None:
            return n - 1, 0.0
        d = q.ComputeDistance(eye, xc.GetLocalToWorldTransform(root))
        s_load = sum(1 for t in load if t <= d)
        s_unload = sum(1 for t in unload if t <= d)
        prev = self.state.get(key)
        s = s_load if prev is None or s_load <= prev else min(s_load, max(prev, s_unload))
        self.state[key] = s
        return min(s, n - 1), d

    def update(self, eye):
        """Evaluate every reachable root from `eye`. Returns (loads, unloads, wanted) as level path lists/sets."""
        eye = Gf.Vec3d(*eye)
        xc = UsdGeom.XformCache()
        wanted = set()
        top = self.stage.GetPrimAtPath(self.regions_path)
        queue = [p for p in top.GetChildren() if p.HasAPI(UsdLod.RootAPI)]
        visited = set()
        while queue:
            root = queue.pop(0)
            visited.add(str(root.GetPath()))
            s, _ = self._root_s(root, xc, eye)
            levels = lod_levels(root)
            keep = range(s, len(levels)) if self.keep_coarser else [s]
            for i in keep:
                if levels[i].HasAuthoredPayloads():
                    wanted.add(str(levels[i].GetPath()))
            if s == 0 and resident(levels[0]):
                queue += [c for c in levels[0].GetChildren() if c.HasAPI(UsdLod.RootAPI)]
        for k in [k for k in self.state if k not in visited]:
            del self.state[k]           # roots out of reach start fresh next time
        loads = sorted(wanted - self.wanted)
        unloads = sorted(self.wanted - wanted)
        self.wanted = wanted
        return loads, unloads, wanted

    def apply(self, loads, unloads):
        with Sdf.ChangeBlock():
            for p in unloads:
                self.stage.Unload(p)
        for p in loads:
            self.stage.Load(p, Usd.LoadWithoutDescendants)


def apply_to_session(stage, sel, layer=None, deactivate=False):
    """Author session-layer opinions so a renderer without UsdLod shows exactly the selected levels.

    deactivate: also deactivate unselected cell levels, so a consumer that loads every payload (usdrecord) still
    only composes what a streaming engine would have loaded."""
    layer = layer or stage.GetSessionLayer()
    with Sdf.ChangeBlock():
        for c in list(sel.cells.values()) + list(sel.assets.values()):
            root = Sdf.Path(c.path)
            if not c.is_cell:
                # Asset roots sit inside instances: de-instance the placement only when it needs a non-default level.
                prim = stage.GetPrimAtPath(c.path)
                dflt = UsdLod.RootAPI(prim).GetLodDefaultIndexAttr().Get()
                if c.index == dflt:
                    continue
                inst = root.GetParentPath()
                spec = Sdf.CreatePrimInLayer(layer, inst)
                spec.SetInfo("instanceable", False)
            for i, name in enumerate(c.levels):
                spec = Sdf.CreatePrimInLayer(layer, root.AppendChild(name))
                a = spec.attributes.get("visibility")
                if a is None:
                    a = Sdf.AttributeSpec(spec, "visibility", Sdf.ValueTypeNames.Token, Sdf.VariabilityVarying)
                a.default = "inherited" if i == c.index else "invisible"
                if deactivate and c.is_cell and i != c.index:
                    spec.SetInfo("active", False)
    return layer
