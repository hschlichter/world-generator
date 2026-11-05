"""Texture names and atlas layouts shared by the texture generator, the material library and the asset geometry.

UV rects are (u0, v0, u1, v1) in USD texture space: st (0, 0) is the bottom-left of the image. In every foliage
cell the stem / base of the sprite sits at the bottom centre of its rect and the sprite grows upwards (+v).
"""

TEXTURE_DIR = "textures"          # relative to world/
MATERIAL_LIBRARY = "materials/library.usda"

# Foliage atlases (RGBA PNG, 1024 x 1024): 2 x 2 cells, each a different twig / leaf cluster.
FOLIAGE_TEXTURES = {
    "leaves_oak": "leaves_oak.png",
    "leaves_birch": "leaves_birch.png",
    "needles_pine": "needles_pine.png",
    "needles_spruce": "needles_spruce.png",
    "leaves_bush": "leaves_bush.png",
}


def foliage_cell(i):
    """UV rect of foliage atlas cell i (0..3): 0 bottom-left, 1 bottom-right, 2 top-left, 3 top-right."""
    u0 = (i % 2) * 0.5
    v0 = (i // 2) * 0.5
    return (u0, v0, u0 + 0.5, v0 + 0.5)


# Ground-cover atlas (RGBA PNG, 1024 x 1024). Bases at the bottom edge of each rect.
GRASS_ATLAS = "grass_atlas.png"
GRASS_RECTS = {
    "grass_a": (0.0, 0.5, 0.5, 1.0),        # dense green tuft
    "grass_b": (0.5, 0.5, 1.0, 1.0),        # taller, sparser, seed heads
    "flower_yellow": (0.0, 0.25, 0.25, 0.5),
    "flower_white": (0.25, 0.25, 0.5, 0.5),
    "flower_purple": (0.5, 0.25, 0.75, 0.5),
    "flower_red": (0.75, 0.25, 1.0, 0.5),
    "grass_dry": (0.0, 0.0, 0.5, 0.25),     # short dry/yellowed tuft, wide
    "clover": (0.5, 0.0, 0.75, 0.25),       # low broad-leaved weeds
    "reed": (0.75, 0.0, 1.0, 0.25),         # thin tall reeds
}
FERN_TEXTURE = "fern.png"                   # RGBA 512 x 1024, one frond, stem at bottom centre

# Tileable surface textures: (colour sRGB JPG, normal PNG raw). Bark: u wraps around the trunk, v along it;
# one texture repeat covers BARK_REPEAT_M metres of trunk length and wraps once around.
SURFACE_TEXTURES = {
    "bark_oak": ("bark_oak_color.jpg", "bark_oak_normal.png"),
    "bark_pine": ("bark_pine_color.jpg", "bark_pine_normal.png"),
    "bark_birch": ("bark_birch_color.jpg", "bark_birch_normal.png"),
    "bark_dead": ("bark_dead_color.jpg", "bark_dead_normal.png"),
    "rock": ("rock_color.jpg", "rock_normal.png"),
    "rock_cliff": ("rock_cliff_color.jpg", "rock_cliff_normal.png"),
    "stone_blocks": ("stone_blocks_color.jpg", "stone_blocks_normal.png"),
    "wood": ("wood_color.jpg", "wood_normal.png"),
    # Buildings and the uncanny (2 m per repeat; banner cloth: one repeat per banner).
    "plaster": ("plaster_color.jpg", "plaster_normal.png"),
    "thatch": ("thatch_color.jpg", "thatch_normal.png"),
    "roof_slate": ("roof_slate_color.jpg", "roof_slate_normal.png"),
    "timber": ("timber_color.jpg", "timber_normal.png"),
    "cloth_red": ("cloth_red_color.jpg", "cloth_red_normal.png"),
    "cloth_blue": ("cloth_blue_color.jpg", "cloth_blue_normal.png"),
    "canvas": ("canvas_color.jpg", "canvas_normal.png"),
    "crystal": ("crystal_color.jpg", "crystal_normal.png"),
    "bone": ("bone_color.jpg", "bone_normal.png"),
}
BARK_REPEAT_M = 2.0     # metres of trunk per texture repeat along v
ROCK_REPEAT_M = 4.0     # metres per repeat for rock / stone planar or spherical UVs (geometry divides by this)
STONE_REPEAT_M = 2.0

# Terrain source materials (tileable, sRGB JPG 512 x 512, TERRAIN_REPEAT_M metres per repeat). They are the inputs of
# the per-region terrain bake and also exist as library materials.
TERRAIN_SOURCES = {
    "grass": "terrain_grass.jpg",
    "dry_grass": "terrain_dry_grass.jpg",
    "soil": "terrain_soil.jpg",
    "forest_floor": "terrain_forest_floor.jpg",
    "stone": "terrain_stone.jpg",
    "mountain_rock": "terrain_mountain_rock.jpg",
    "snow": "terrain_snow.jpg",
    "path": "terrain_path.jpg",
    "crops": "terrain_crops.jpg",          # ripening grain (farm parcels)
    "ploughed": "terrain_ploughed.jpg",    # furrowed earth (farm parcels)
}
TERRAIN_REPEAT_M = 4.0
TERRAIN_DETAIL_NORMAL = "terrain_detail_normal.png"   # tileable 1024 x 1024, raw
TERRAIN_DETAIL_ROUGHNESS = "terrain_detail_roughness.png"  # tileable 512 x 512, raw, single channel
SKY_TEXTURE = "sky.exr"   # HDR lat-long 4096 x 2048, linear Rec.709 radiance, half float


def region_texture_paths(region_name):
    """Per-region baked terrain textures, relative to world/."""
    base = f"{TEXTURE_DIR}/regions/{region_name}"
    return {
        "color": f"{base}/terrain_color.jpg",          # region texture size (2048^2), whole region, st 0..1
        "color_far": f"{base}/terrain_color_far.jpg",  # proxy texture size (1024^2)
        "normal_far": f"{base}/terrain_normal_far.png",  # tangent-space macro normal for the proxy mesh
    }


# Material library prims (under /Materials in MATERIAL_LIBRARY).
LIBRARY_MATERIALS = [
    "bark_oak", "bark_pine", "bark_birch", "bark_dead",
    "leaves_oak", "leaves_birch", "needles_pine", "needles_spruce", "leaves_bush",
    "grass", "fern",
    "rock", "rock_cliff", "stone_blocks", "wood",
    "plaster", "thatch", "roof_slate", "timber", "cloth_red", "cloth_blue", "canvas", "crystal", "bone",
    "terrain_grass", "terrain_soil", "terrain_stone", "terrain_mountain_rock", "terrain_snow",
    "terrain",          # interface: inputs:color_texture (asset); tiled detail normal; inputs:detail_scale (float2)
    "terrain_far",      # interface: inputs:color_texture, inputs:normal_texture (assets)
]
