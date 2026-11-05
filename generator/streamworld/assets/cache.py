"""Reads written assets back (geometry per variant and level, bounds, triangle counts) for merging and statistics."""

import os

import numpy as np
from pxr import Usd, UsdGeom, UsdShade

from ..meshutil import Mesh
from . import library as L


def _mesh(prim):
    m = UsdGeom.Mesh(prim)
    pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
    pv = UsdGeom.PrimvarsAPI(prim)
    n = pv.GetPrimvar("normals")
    st = pv.GetPrimvar("st")
    return Mesh(pts, counts, idx, np.asarray(n.Get()) if n and n.HasValue() else None,
                np.asarray(st.Get()) if st and st.HasValue() else None)


class AssetCache:
    def __init__(self, cfg):
        self.cfg = cfg
        self._lv = {}
        self._mats = {}
        self._scatter = {}

    def levels(self, name, variant):
        """[{part: Mesh}] per LOD level (asset local space)."""
        key = (name, variant)
        if key not in self._lv:
            spec = L.ASSETS_BY_NAME[name]
            st = Usd.Stage.Open(spec.geom_path(self.cfg.out), Usd.Stage.LoadAll)
            root = st.GetPrimAtPath(f"/{name}_geom/{variant}")
            if not root:
                raise KeyError(f"asset {name} has no variant {variant}; regenerate assets")
            lvls = []
            for lp in root.GetChildren():
                lvls.append({c.GetName(): _mesh(c) for c in lp.GetChildren() if c.IsA(UsdGeom.Mesh)})
            pts = np.concatenate([m.points for lv in lvls for m in lv.values()])
            self._lv[key] = dict(levels=lvls, lo=pts.min(0), hi=pts.max(0), tris=[sum(m.n_tris for m in lv.values())
                                                                                     for lv in lvls])
        return self._lv[key]

    def materials(self, name, variant):
        """{part: library material name}, as bound in the asset's interface layer."""
        key = (name, variant)
        if key not in self._mats:
            spec = L.ASSETS_BY_NAME[name]
            st = Usd.Stage.Open(spec.interface_path(self.cfg.out), Usd.Stage.LoadAll)
            root = st.GetDefaultPrim()
            root.GetVariantSets().GetVariantSet(spec.variant_set).SetVariantSelection(variant)
            out = {}
            for lp in st.GetPrimAtPath(root.GetPath().AppendChild("Geom")).GetChildren():
                for part in lp.GetChildren():
                    rel = UsdShade.MaterialBindingAPI(part).GetDirectBindingRel()
                    t = rel.GetTargets() if rel else []
                    if t:
                        out[part.GetName()] = t[0].name
            self._mats[key] = out
        return self._mats[key]

    def scatter(self, name):
        if name not in self._scatter:
            st = Usd.Stage.Open(L.scatter_path(self.cfg.out, name))
            m = _mesh(st.GetPrimAtPath(f"/{name}/Geom"))
            lo, hi = m.extent()
            self._scatter[name] = dict(mesh=m, lo=lo, hi=hi, tris=m.n_tris)
        return self._scatter[name]

    def bound_radius(self, name, variant=None):
        d = self.scatter(name) if variant is None else self.levels(name, variant)
        return float(np.max(np.abs(np.r_[d["lo"], d["hi"]])))
