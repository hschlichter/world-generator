Here's a walk through the file. Only a few lines in it actually come from UsdLod. The rest is ordinary USD.

1. The UsdLod parts

There are three pieces.

The root: LODRootAPI. It's applied to Geom:

usda
def Xform "Geom" (
    prepend apiSchemas = ["LODRootAPI"]     # <- applied API schema: "this prim is a LOD root"
    ...
)
{
    int lod:default:index = 7               # <- LODRootAPI property
    rel lod:heuristics = [                  # <- LODRootAPI property: which heuristic(s) drive me
        </oak_lod8/LODHeuristics/imaging_screen>,
        </oak_lod8/LODHeuristics/imaging_distance>,
    ]

LODRootAPI is an applied API schema. It doesn't change what the prim is (it's still an Xform); it adds the lod: properties to it.
- lod:default:index says which level to use when nothing has evaluated a heuristic yet.
- lod:heuristics is a relationship. It points to the prims that decide which level to show.

The levels, which have no syntax of their own. The schema defines the levels as the root's direct children, in order, finest first. The names lod0…lod7 are my choice; UsdLod only looks at their order. That's why the file has no "level" keyword anywhere.

The heuristics: two typed schemas.

usda
def LODScreenSizeHeuristic "imaging_screen"
{
    uniform float[] thresholds = [0.25, 0.12, ...]       # 7 values -> 8 levels
    uniform float[] blendThresholds = [0.2125, ...]      # blend zone after each threshold
    float3[] extent = [...]                              # box whose projected size is measured
    uniform token projectionMethod = "projectedSphere"
    uniform token lod:domain = "imaging"                 # rendering, as opposed to e.g. "physics"
}

def LODDistanceHeuristic "imaging_distance"
{
    uniform float[] thresholds = [49.3, 70.8, ...]       # metres
    uniform float[] blendThresholds = [53.4, ...]
    point3f center = (...)                               # distance is measured to this point
    uniform token lod:domain = "imaging"
}

LODScreenSizeHeuristic and LODDistanceHeuristic are real prim types, like Mesh or Xform. N thresholds give N+1 levels. A renderer uses the matching query class (UsdLod.ScreenSizeHeuristicQuery, UsdLod.DistanceHeuristicQuery) to turn a camera into a level number. With blend thresholds that number can be fractional, for example 2.4, which means "blend 40% of the way from lod2 to lod3". Your engine can use either heuristic; they're set up to switch at the same points for the reference camera.

Not UsdLod:
- The LODHeuristics scope name. The heuristics can live anywhere, because the relationship finds them.
- visibility = "invisible" on lod0–lod6. That's a fallback for viewers that don't understand LOD, such as plain usdview or Blender, so they draw only lod7 instead of all 8 levels stacked on top of each other. A LOD-aware engine ignores it and does its own selection.

2. What over means

USD has three prim specifiers:

┌───────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Specifier │                                                     Meaning                                                     │
├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ def       │ "This prim exists, and here is its definition."                                                                 │
├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ over      │ "If this prim exists (defined somewhere else), add these opinions to it." It doesn't create the prim by itself. │
├───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ class     │ An abstract template that other prims inherit from.                                                             │
└───────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

The meshes lod0/wood, lod0/leaves and so on are defined in oak_lod8_geom.usdc, which Geom brings in with:

usda
prepend references = @./oak_lod8_geom.usdc@</oak_lod8_geom>

The interface file doesn't repeat the geometry. It only adds opinions on top of it:

usda
over "lod0"
{
    token visibility = "invisible"                     # add: hide this level
    over "wood" ( prepend apiSchemas = ["MaterialBindingAPI"] )
    {
        rel material:binding = </oak_lod8/Materials/bark_oak>   # add: bind bark material
    }
}

When the stage is composed, USD merges both files into one prim /oak_lod8/Geom/lod0/wood: the mesh comes from the .usdc, the visibility and material from the .usda. Where both files set the same property, the opinion in this file wins over the referenced one (local opinions are stronger than references).

The point of the split: the .usdc holds only plain geometry (heavy, binary, reusable), and the .usda is the small, human-readable "interface" where the asset's behaviour (LOD, materials, visibility) is set. If the reference were missing, the overs would describe prims that don't exist, and a normal traversal would skip them.

You can see the composed result with usdcat --flatten world/assets/lodtest/oak_lod8.usda.

3. The customData on the root Xform

usda
customData = {
    float[] switch_distances_reference_camera = [...]
    int[] triangles_per_level = [15998, 6085, ...]
}

customData is a free-form dictionary of metadata that any prim or property can carry. USD itself, renderers and UsdLod all ignore it, and it has no effect on what you see. My generator writes these two values for the scene manifest and for people reading the file.

- It's informational only. The switch distances that actually take effect are the thresholds on the heuristic prims. If someone edits those thresholds, the customData copy becomes stale, because nothing keeps the two in sync.
- The other metadata on that prim is standardised, unlike customData:
  - kind = "component" marks this prim as an asset in USD's model hierarchy.
  - assetInfo gives the asset's name and identifier.
  - GeomModelAPI with extentsHint gives a bounding box for the whole asset. A renderer can use it to cull the tree without loading or reading its geometry.

A correction to my earlier summary: I rebuilt the tree after writing the switch-distance table. The file's actual distances are 49.3, 70.8, 101.5, 145.7, 209.4, 301.5 and 429.4 m, a little further out than the 48 / 69 / 98 m … figures I gave. The screen-size thresholds themselves are unchanged. The final crown hulls made the tree's bounding box slightly bigger, so the same screen size is reached a bit further away.
