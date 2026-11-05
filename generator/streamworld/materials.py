"""The shared material library: UsdPreviewSurface networks for every material in atlas.LIBRARY_MATERIALS.

Assets and cells reference these materials (`references = @.../materials/library.usda@</Materials/<name>>`) into
their own Materials scope and bind them there. Materials whose texture varies per use (`terrain`, `terrain_far`)
expose the texture as a Material interface input that the referencing layer overrides.
"""

import os

from pxr import Gf, Sdf, UsdShade

from . import atlas
from .usdutil import new_stage

T = Sdf.ValueTypeNames
NORMAL_SCALE = Gf.Vec4f(2, 2, 2, 1)
NORMAL_BIAS = Gf.Vec4f(-1, -1, -1, 0)


def _tex(name):
    return f"../{atlas.TEXTURE_DIR}/{name}"


class _Builder:
    def __init__(self, stage, name, doc=None):
        self.stage = stage
        self.path = Sdf.Path(f"/Materials/{name}")
        self.mat = UsdShade.Material.Define(stage, self.path)
        if doc:
            self.mat.GetPrim().SetDocumentation(doc)
        self.pbr = UsdShade.Shader.Define(stage, self.path.AppendChild("Surface"))
        self.pbr.CreateIdAttr("UsdPreviewSurface")
        self.mat.CreateSurfaceOutput().ConnectToSource(self.pbr.ConnectableAPI(), "surface")
        self.pbr.CreateInput("metallic", T.Float).Set(0.0)
        self._st = None

    def st(self):
        if self._st is None:
            r = UsdShade.Shader.Define(self.stage, self.path.AppendChild("StReader"))
            r.CreateIdAttr("UsdPrimvarReader_float2")
            r.CreateInput("varname", T.String).Set("st")
            r.CreateInput("fallback", T.Float2).Set(Gf.Vec2f(0, 0))
            self._st = r.CreateOutput("result", T.Float2)
        return self._st

    def tiled_st(self, name, scale_input):
        """st through a UsdTransform2d whose scale comes from a Material interface input."""
        x = UsdShade.Shader.Define(self.stage, self.path.AppendChild(name))
        x.CreateIdAttr("UsdTransform2d")
        x.CreateInput("in", T.Float2).ConnectToSource(self.st())
        x.CreateInput("scale", T.Float2).ConnectToSource(scale_input)
        return x.CreateOutput("result", T.Float2)

    def texture(self, name, file, color_space, st=None, fallback=None, scale=None, bias=None):
        t = UsdShade.Shader.Define(self.stage, self.path.AppendChild(name))
        t.CreateIdAttr("UsdUVTexture")
        f = t.CreateInput("file", T.Asset)
        if isinstance(file, UsdShade.Input):
            f.ConnectToSource(file)
        else:
            f.Set(Sdf.AssetPath(file))
        t.CreateInput("sourceColorSpace", T.Token).Set(color_space)
        t.CreateInput("wrapS", T.Token).Set("repeat")
        t.CreateInput("wrapT", T.Token).Set("repeat")
        t.CreateInput("st", T.Float2).ConnectToSource(st if st is not None else self.st())
        if fallback is not None:
            t.CreateInput("fallback", T.Float4).Set(Gf.Vec4f(*fallback))
        if scale is not None:
            t.CreateInput("scale", T.Float4).Set(scale)
        if bias is not None:
            t.CreateInput("bias", T.Float4).Set(bias)
        return t

    def diffuse_tex(self, file, st=None, alpha=False):
        t = self.texture("DiffuseTexture", file, "sRGB", st, fallback=(0.4, 0.4, 0.35, 1))
        self.pbr.CreateInput("diffuseColor", T.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        t.CreateOutput("rgb", T.Float3)
        if alpha:
            t.CreateOutput("a", T.Float)
            self.pbr.CreateInput("opacity", T.Float).ConnectToSource(t.ConnectableAPI(), "a")
            self.pbr.CreateInput("opacityThreshold", T.Float).Set(0.5)
        return t

    def normal_tex(self, file, st=None, name="NormalTexture"):
        t = self.texture(name, file, "raw", st, fallback=(0.5, 0.5, 1, 1), scale=NORMAL_SCALE, bias=NORMAL_BIAS)
        t.CreateOutput("rgb", T.Float3)
        self.pbr.CreateInput("normal", T.Normal3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        return t

    def roughness(self, v):
        self.pbr.CreateInput("roughness", T.Float).Set(float(v))

    def interface(self, name, type_name, default):
        i = self.mat.CreateInput(name, type_name)
        i.Set(default)
        return i


def write_library(cfg):
    """Write <out>/materials/library.usda. Returns its path."""
    path = os.path.join(cfg.out, atlas.MATERIAL_LIBRARY)
    stage = new_stage(path, "Materials", doc="Eldermark shared material library (UsdPreviewSurface).")
    UsdShade.Material  # noqa
    stage.DefinePrim("/Materials", "Scope")

    surfaces = {k: v for k, v in atlas.SURFACE_TEXTURES.items()}
    rough = {"bark_oak": 0.9, "bark_pine": 0.88, "bark_birch": 0.75, "bark_dead": 0.85, "rock": 0.82,
             "rock_cliff": 0.85, "stone_blocks": 0.86, "wood": 0.8, "plaster": 0.92, "thatch": 0.95,
             "roof_slate": 0.6, "timber": 0.85, "cloth_red": 0.9, "cloth_blue": 0.9, "canvas": 0.9, "crystal": 0.12, "bone": 0.7}
    for name in rough:
        b = _Builder(stage, name)
        c, n = surfaces[name]
        b.diffuse_tex(_tex(c))
        b.normal_tex(_tex(n))
        b.roughness(rough[name])
        if name == "crystal":
            # Faint inner light, so crystals and runes read at dusk and from afar.
            b.pbr.CreateInput("emissiveColor", T.Color3f).Set(Gf.Vec3f(0.12, 0.2, 0.55))

    foliage = dict(atlas.FOLIAGE_TEXTURES)
    foliage["grass"] = atlas.GRASS_ATLAS
    foliage["fern"] = atlas.FERN_TEXTURE
    for name in ("leaves_oak", "leaves_birch", "needles_pine", "needles_spruce", "leaves_bush", "grass", "fern"):
        b = _Builder(stage, name, doc="Alpha-tested foliage; bind to doubleSided meshes.")
        b.diffuse_tex(_tex(foliage[name]), alpha=True)
        b.roughness(0.7)

    for name, src in (("terrain_grass", "grass"), ("terrain_soil", "soil"), ("terrain_stone", "stone"),
                      ("terrain_mountain_rock", "mountain_rock"), ("terrain_snow", "snow")):
        b = _Builder(stage, name, doc=f"Tiled terrain source material ({atlas.TERRAIN_REPEAT_M} m per repeat in st).")
        b.diffuse_tex(_tex(atlas.TERRAIN_SOURCES[src]))
        b.normal_tex(_tex(atlas.TERRAIN_DETAIL_NORMAL))
        b.roughness(0.6 if src == "snow" else 0.9)

    # Terrain: per-region baked colour (interface input) + shared tiled detail normal and roughness.
    b = _Builder(stage, "terrain", doc="Terrain: override inputs:color_texture with the region bake; st spans the "
                                       "region 0..1; inputs:detail_scale = region size / detail tiling.")
    ctex = b.interface("color_texture", T.Asset, Sdf.AssetPath(_tex(atlas.TERRAIN_SOURCES["grass"])))
    scale = b.interface("detail_scale", T.Float2, Gf.Vec2f(166.6667, 166.6667))
    dst = b.tiled_st("DetailSt", scale)
    b.diffuse_tex(ctex)
    b.normal_tex(_tex(atlas.TERRAIN_DETAIL_NORMAL), st=dst)
    rt = b.texture("RoughnessTexture", _tex(atlas.TERRAIN_DETAIL_ROUGHNESS), "raw", dst, fallback=(0.9, 0.9, 0.9, 1))
    rt.CreateOutput("r", T.Float)
    b.pbr.CreateInput("roughness", T.Float).ConnectToSource(rt.ConnectableAPI(), "r")

    b = _Builder(stage, "terrain_far", doc="Region proxy terrain: override inputs:color_texture / normal_texture.")
    ctex = b.interface("color_texture", T.Asset, Sdf.AssetPath(_tex(atlas.TERRAIN_SOURCES["grass"])))
    ntex = b.interface("normal_texture", T.Asset, Sdf.AssetPath(_tex(atlas.TERRAIN_DETAIL_NORMAL)))
    b.diffuse_tex(ctex)
    b.normal_tex(ntex)
    b.roughness(0.92)

    missing = [m for m in atlas.LIBRARY_MATERIALS if not stage.GetPrimAtPath(f"/Materials/{m}")]
    assert not missing, missing
    stage.GetRootLayer().Save()
    return path
