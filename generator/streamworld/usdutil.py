"""Helpers for authoring USD layers the same way everywhere (stage metadata, meshes, kinds, bounds)."""

import os

import numpy as np
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt


def new_stage(path, default_prim=None, doc=None):
    """Create (overwrite) a layer with the project's stage conventions."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    # LoadNone: payloads authored into the new layer are not composed while writing it (root.usda would otherwise
    # load every region of the world as its cell roots are added).
    stage = Usd.Stage.CreateNew(path, Usd.Stage.LoadNone)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    if doc:
        stage.GetRootLayer().documentation = doc
    if default_prim:
        stage.GetRootLayer().defaultPrim = default_prim
    return stage


def new_layer(path, default_prim=None, doc=None):
    """Create (overwrite) an Sdf layer with the project's conventions (for bulk prim authoring)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    layer = Sdf.Layer.CreateNew(path)
    layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    layer.pseudoRoot.SetInfo("upAxis", "Z")
    if doc:
        layer.documentation = doc
    if default_prim:
        layer.defaultPrim = default_prim
    return layer


def rel(from_path, to_path):
    """Relative asset path from the directory of layer `from_path` to file `to_path`, always starting with ./ or ../"""
    r = os.path.relpath(to_path, os.path.dirname(from_path)).replace(os.sep, "/")
    return r if r.startswith(".") else "./" + r


def set_kind(prim, kind):
    Usd.ModelAPI(prim).SetKind(kind)


def vec3f_array(a):
    return Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(np.asarray(a, dtype=np.float32)))


def extent_of(points):
    p = np.asarray(points)
    return [Gf.Vec3f(*map(float, p.min(axis=0))), Gf.Vec3f(*map(float, p.max(axis=0)))]


def write_mesh(stage, path, mesh, material=None, purpose=None, collision=None):
    """Author a UsdGeomMesh from a meshutil.Mesh. Returns the UsdGeom.Mesh."""
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(vec3f_array(mesh.points))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.counts.astype(np.int32))))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(mesh.indices.astype(np.int32))))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateExtentAttr(extent_of(mesh.points))
    pv = UsdGeom.PrimvarsAPI(m)
    if mesh.normals is not None:
        n = pv.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.vertex)
        n.Set(vec3f_array(mesh.normals))
    if mesh.st is not None:
        st = pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.ascontiguousarray(mesh.st.astype(np.float32))))
    if mesh.color is not None:
        c = pv.CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        c.Set(vec3f_array(mesh.color))
    if mesh.double_sided:
        m.CreateDoubleSidedAttr(True)
    if purpose:
        m.CreatePurposeAttr(purpose)
    if material is not None:
        bind(m.GetPrim(), material)
    if collision:
        from pxr import UsdPhysics
        UsdPhysics.CollisionAPI.Apply(m.GetPrim())
        mc = UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim())
        mc.CreateApproximationAttr(collision)
    return m


def bind(prim, material_path):
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    mat = UsdShade.Material(prim.GetStage().GetPrimAtPath(material_path))
    if not mat:
        # Bind by path even when the material is not composed yet (e.g. comes in through a reference later).
        api.GetDirectBindingRel().SetTargets([Sdf.Path(material_path)])
    else:
        api.Bind(mat)


def set_extents_hint(prim, lo, hi):
    Usd.ModelAPI(prim)  # noqa: ensure model API is available
    UsdGeom.ModelAPI.Apply(prim).SetExtentsHint(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))


def compute_extents_hint(prim):
    """Union of the composed extents of the prim's subtree (default + render + proxy purposes)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render,
                                                        UsdGeom.Tokens.proxy], useExtentsHint=True,
                              ignoreVisibility=True)
    b = cache.ComputeUntransformedBound(prim).ComputeAlignedRange()
    return b.GetMin(), b.GetMax()


def xform_ops(xf, translate=None, rotate_xyz=None, scale=None, orient=None):
    xf = UsdGeom.Xformable(xf)
    if translate is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, translate)))
    if orient is not None:
        xf.AddOrientOp().Set(Gf.Quatf(*map(float, orient)))
    if rotate_xyz is not None:
        xf.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, rotate_xyz)))
    if scale is not None:
        s = scale if np.ndim(scale) else (scale, scale, scale)
        xf.AddScaleOp().Set(Gf.Vec3f(*map(float, s)))


# -- Sdf-level helpers (bulk authoring) -------------------------------------------------------------------------------

def sdf_attr(spec, name, type_name, value, variability=Sdf.VariabilityVarying):
    a = Sdf.AttributeSpec(spec, name, type_name, variability)
    a.default = value
    return a


def sdf_xform_trs(spec, t, rz_deg, s, tilt=None):
    """translate / rotateXYZ / scale on a prim spec."""
    sdf_attr(spec, "xformOp:translate", Sdf.ValueTypeNames.Double3, Gf.Vec3d(*map(float, t)))
    r = (0.0, 0.0, float(rz_deg)) if tilt is None else (float(tilt[0]), float(tilt[1]), float(rz_deg))
    sdf_attr(spec, "xformOp:rotateXYZ", Sdf.ValueTypeNames.Float3, Gf.Vec3f(*r))
    sdf_attr(spec, "xformOp:scale", Sdf.ValueTypeNames.Float3, Gf.Vec3f(float(s), float(s), float(s)))
    sdf_attr(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray,
             Vt.TokenArray(["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]), Sdf.VariabilityUniform)
