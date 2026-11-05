# Eldermark: world design and USD structure

Eldermark is a 10 km × 10 km medieval fantasy march (100 km², Z up, metres, centred on the origin, x east,
y north). It is built to exercise world streaming, LOD, culling, instancing and large-scene management. All 100
regions are generated at the full hierarchy depth; `generate` without `--regions` rewrites only R_4_5 (x −1000…0 m,
y 0…1000 m), the region with the village of Elderford. Map: [`previews/map.png`](previews/map.png). The LOD and
streaming model is described for engine implementers in [`LOD_STREAMING.html`](LOD_STREAMING.html).

## 1. World layout and biomes

- **Mountains.** Ridge polylines raise ridged-multifractal mountains: the *Greyspine* wall along the whole north,
  the *Eastern Teeth* down the east side, the high *Old Crown* massif in the north-west (the highest summit, about
  980 m), the *Dragonback* ridge in the south-west, and lower hill ranges in the lowlands (*Raven Hills* under the
  castle, *Fellmoor*, *Barrow Knolls*, the *Low Downs* in the south). Tilted, warped strata bands produce cliff
  steps. Droplet erosion (28 M droplets on a 2 m grid, run window by window) carves gullies and alluvial flats,
  and a thermal pass leaves scree slopes.
- **Valleys.** The *Elder Vale* runs from the Greyspine through a gorge, past Elderford, to the southern edge;
  *Thornvale* breaks through the Eastern Teeth in a gorge and falls south-west. Their carve depth includes most of
  the local mountain height, which gives complex visibility in the gorges and broad vales below.
- **Lowlands.** Swells and basins at the kilometre scale, rolling swells on three smaller scales, and ridged
  foothills fading out from the ranges.
- **Forest.** A noise-driven forest field (woodlands, groves at the edges, clearings), biased by moisture and
  the designed `[[biomes.areas]]` (*Mirkwood* in the south-west, *Thornwood* in the east), and limited by slope
  and the tree line. Woods are open: 300 trees per hectare at full forest density, and tree density varies
  strongly at the 40–100 m scale, so stands alternate with light-filled glades. Species follow the fields: oak
  and birch low down (birch in wet ground and near the valleys), pine and spruce with altitude, deadwood near the
  tree line and in open ground. Forest edges grow saplings and bushes.
- **Farmland.** `farm` areas around the settlements clear the woods and lay out medieval open fields in the
  terrain texture: furlongs of long strips (9–18 m wide), each strip ripening grain, green crop, ploughed earth or
  fallow, with grass balks between strips and headlands at furlong ends. Fields keep to workable slopes; ground
  cover, bushes and rocks are mostly cleared from them.
- **Ground cover.** Grass tufts, dry grass, flowers in meadow patches (colour by patch), clover, reeds in wet
  ground and near the valley floors (not in fields or villages), ferns under trees, pebbles on rocky ground.
- **Settlements** (`[[pois]]`, ground levelled under each): the villages *Elderford* (16 houses, a chapel and
  banners around a green with a well) and *Highmoor* (11 houses), the hamlets *Wyrmsby* and *Brackenhollow*,
  *Castle Ravencrest* on the Raven Hills (curtain walls with corner towers and bannered wall walks, a twin-towered
  gatehouse, a keep, halls and a well in the bailey), two windmills with the miller's house and haystacks, and
  *Saint Wenna's Yard* (a chapel, rows of gravestones, a tumbled churchyard wall). Houses face the green, each
  with a fenced plot behind it.
- **Ruins and landmarks**: the *Sunken Keep*, the *Ring of the Nine* stone circle, the *Hollow Watch* and
  *Greyfell Watch* towers, the *Wardens' Arch*, the *Moss Shrine*, the *Broken Colonnade* and the *Needles*.
- **The uncanny**: the *Spire of Aldric* (a leaning wizard's tower with a bent slate spire, glowing windows, a
  crystal floating above the tip and rock shards circling the upper tower, ringed by runestones), *Wyrm's Rest*
  (the half-buried skeleton of a 55 m dragon on the downs), *Shimmerdeep* (crystal clusters breaking out of the
  Greyspine foothills) and the *Stones of Varn* (a ring of runestones). Runes, crystal and the spire's windows
  use the `crystal` material, which glows faintly.
- **Roads.** The *King's Road* runs the length of the march from the southern edge through Elderford, past the
  castle and Highmoor, to a pass in the Greyspine; the *Westway*, *Eastway* and smaller tracks link every
  settlement and site. Roads are carved gently into the terrain, kept clear of vegetation and drawn into the
  terrain texture. Waystones (each with a glowing rune) and wayside shrines stand beside them every 420 m.

Transitions are continuous fields, never masks with hard edges: forest density, conifer share, rock exposure,
farmland and moisture all blend, and placement is probabilistic against them.

## 2. Streaming partition

Default: two levels of cell LOD roots plus asset roots, all configurable (`[partition]`). Every cell has three
heuristic domains, all distances from the camera to the cell's bounding box: **imaging** (which level is drawn),
**streaming** (which levels are resident: load radius per boundary, unload radius for hysteresis) and, on tiles,
**physics**.

| Root | Size | Levels (finest first) | Default | Imaging switch (blend to) | Streaming load / unload | Physics |
|---|---|---|---|---|---|---|
| Region `R_i_j` | 1000 m (100 in the world) | `full` (its 16 tiles, a reference), `medium`, `proxy` | proxy | 600 (680), 1200 (1300) m | 700 / 800, 1400 / 1600 m | |
| Tile `T_i_j` | 250 m (16 per region) | `full` (everything), `medium`, `simplified` | simplified | 220 (250), 420 (460) m | 300 / 360, 500 / 580 m | 120, 420 m |
| Ground chunk `G_i_j` | 50 m (25 per tile) | `fine` (0.5 m + micro-relief, a payload), `base` (1 m, inline) | base | 50 (58) m | 100 / 120 m | (base collides) |
| Asset (tree, rock, …) | n/a | `lod0` … `lod7` | coarsest | ScreenSize (projectedSphere), per class | streamed with their tile | rocks, structures, buildings |

```
world/root.usda                                      /World (assembly) /Regions (group)
  /World/Regions/R_4_5           LOD root, group, translate to region centre, extentsHint
     full    -> reference regions/R_4_5/R_4_5_full.usda   (structure only: 16 tile roots + their heuristics)
     medium  -> payload regions/R_4_5/R_4_5_medium.usdc   (terrain 5 m, point instances of the assets' levels)
     proxy   -> payload regions/R_4_5/R_4_5_proxy.usdc    (terrain 10 m, point instances: trees over 5 m, large assets)
  /World/Regions/LODHeuristics/R_4_5   imaging_distance, imaging_screen, streaming_distance, bounds (guide box)
  regions/R_4_5/R_4_5_full.usda
     /R_4_5_full/T_1_3           LOD root, group, translate to tile centre (relative to region), extentsHint
        full        -> payload tiles/R_4_5_T_1_3_full.usdc
        medium      -> payload tiles/R_4_5_T_1_3_medium.usdc
        simplified  -> payload tiles/R_4_5_T_1_3_simplified.usdc
     /R_4_5_full/LODHeuristics/T_1_3      imaging_distance, imaging_screen, physics_distance, streaming_distance,
                                          bounds
  tiles/R_4_5_T_1_3_full.usdc    G_0_0 … G_4_4 (ground chunks, LOD roots):
                                    fine  -> payload tiles/R_4_5_T_1_3_ground.usdc</R_4_5_T_1_3_ground/G_i_j>
                                    base     inline: 1 m ground + skirt (collision)
                                 LODHeuristics/G_i_j (imaging_distance, streaming_distance, bounds), Materials,
                                 Trees / Bushes / Rocks / Deadfall / Structures / Buildings / Props (groups of
                                 instanceable placements), GroundCover (component: 2 x 2 PointInstancer chunks)
  tiles/R_4_5_T_1_3_ground.usdc      one prim per chunk: 0.5 m ground + skirt + material on the tile's fine texture
  tiles/R_4_5_T_1_3_medium.usdc      Terrain (2 m), Instances (PointInstancer of asset levels), Materials
  tiles/R_4_5_T_1_3_simplified.usdc  Terrain (5 m), Instances, Materials
```

Data per cell level (world averages of the files on disk): ground chunk fine 0.35 MB (8.8 MB per tile ground file);
tile full 5.3 MB → medium 0.64 MB → simplified 0.19 MB; region as 16 simplified tiles 3.0 MB → medium 2.7 MB →
proxy 0.63 MB. The coarse levels are small because their assets are point instances of shared asset files; their
triangle counts are not (region medium about 0.96 M, proxy 0.34 M, mostly instanced lod7 trees).

Why this shape:
- **Bounded switch cost.** Going finer replaces one cell's level with its next level. A region switching from
  `medium` to `full` brings in 16 tiles at `simplified` (4.9 MB), prefetched from 700 m for the 600 m switch.
- **Drawing and residency are separate.** The streaming domain loads each level before its imaging switch
  (load radius ≥ imaging threshold) and releases it only beyond a larger unload radius, so a camera hovering at
  a threshold doesn't load and unload the same file repeatedly. See `lodselect.Streamer` and `stream-replay`.
- **Tile structure always composed.** A region's `full` level is a reference (`full_arc = "reference"`): 70 KB of
  tile roots and heuristics per region, 7 MB for the world (7.4 MB with `root.usda`). Tiles stream on their own heuristics and
  never wait for a parent payload.
- **Bounded far field.** With every root at its coarsest level, the world is its region proxies: 21k terrain
  triangles and about 8k lod7 asset instances (0.32 M triangles) per region, 33.7 M triangles for the whole 100 km²,
  nearly all of it GPU-instanced.
- **LOD roots near the camera only.** Asset roots and ground chunks exist only inside tile `full` levels.
- **Payload granularity.** Each content level of a cell is one file: 3 per tile, 2 per region plus the region's
  structure file.
- **Heuristics outside the roots.** A root's children are its levels, so the heuristic prims and their
  bounding-volume boxes live in a sibling `LODHeuristics` Xform (with the cell's transform). They stay available
  while the levels are unloaded, which is exactly when a streamer needs them.
- **Generic depth.** The writers recurse: a cell's `full` level holds its child cells, and the finest cells hold
  content. Adding a level to `[[partition.levels]]` adds a hierarchy level.

Coordinates: cell prims carry a translate to their centre and payload content is cell-local, so points stay
within ±500 m (regions) or ±125 m (tiles). They remain precise in single precision anywhere in the 10 km world.

## 3. LOD

**Defaults (`lod:default:index`).** Every root defaults to its **coarsest** level, and every non-default level is
authored `visibility = invisible`. A consumer that ignores UsdLod therefore draws exactly one level per root:
the cheapest complete representation. For the root stage that is the whole map as region proxies, the only
default whose cost is bounded for the full world. A LOD-aware consumer owns the visibility of LOD items: it shows
the selected level and hides the others. The same rule holds for physics through `physics:collisionEnabled`,
which is authored on every collider: `true` on the default level, `false` on the others (no level relies on
the schema fallback). This costs the LOD-aware consumer one visibility opinion per switch;
the alternative, defaulting to `full`, would make any non-LOD consumer draw the finest world.

**Hierarchical LOD.** Every coarse level of a cell is reduced terrain plus one PointInstancer whose prototypes are
the assets' own LOD levels (each prototype references `<asset>_geom.usdc</<asset>_geom/<variant>/lod<i>>` and binds
the asset's library materials). Each instance gets the level the asset itself would select at the distance where
the coarse level takes over (the asset's switch distances for the reference camera, times the instance's scale).
So a cell switching to a coarser level keeps every tree's and building's shape and material: a tree that showed
lod6 in the tile's `full` level is still lod6 in its `medium` level, and the region levels show trees at lod7.
What changes at a switch is the terrain resolution and which small assets are kept:

| Level | Takes over at | Terrain | Instances kept |
|---|---|---|---|
| tile `medium` | 220 m | 2 m | every tree and bush, other assets over 1.5 m |
| tile `simplified` | 420 m | 5 m | every tree, bushes over 0.6 m, other assets over 2.5 m |
| region `medium` | 600 m | 5 m, near material | every tree, bushes over 1 m, other assets over 4 m |
| region `proxy` | 1200 m | 10 m, 1024² bake + macro normal | trees over 5 m, other assets over 6 m (every building) |

The region `medium` uses the near terrain material, the same as tile `simplified`, so the 600 m switch changes
geometry density but not shading. Instanced asset levels never collide (`physics:collisionEnabled = false` on the
prototypes); physics uses the cells' own levels.

**Ground LOD.** Inside a tile's `full` level the terrain is 25 chunks of 50 m, each a LOD root (a child of the
level, so the cell walk nests into it like tiles inside a region):
- `fine` (within 50 m; loaded from 100 m, released beyond 120 m): a 0.5 m mesh. Heights are a Catmull-Rom
  interpolation of the 1 m heightfield plus synthesized micro-relief: clods and tussocks, forest-floor hummocks,
  sharp ridged detail on rock, medieval ridge-and-furrow on the farm strips (aligned with the painted strips),
  smoother trodden paths and settlement ground. It is coloured by a per-tile 1024² texture (0.24 m per texel)
  that samples the terrain source textures instead of their mean colours, and is the payload of one prim in the
  tile's `_ground.usdc` (one file per tile, one payload per chunk).
- `base` (the default; collision): the 1 m mesh, inline in the tile's `full` level.

Chunk skirts cover the worst gap between fine and base (and between base and every coarser terrain level).

**Assets.** Every level is genuinely different geometry:
- *Trees and bushes*: one skeleton per variant. lod0 has the trunk and two branch orders as tubes plus one
  alpha-tested card per twig cluster; lod1 has the trunk and primary branches at low tessellation, with leaf
  cards k-means-merged into fewer, larger cards; lod2 has a trunk prism and 2–5 convex crown hulls (Blender)
  carrying the foliage texture.
- *Rocks*: Blender quadric decimation of the fractured lod0.
- *Ruins*: lod0 has individual blocks, lod1 the merged stepped profile, lod2 a silhouette prism.
- *Buildings, props and the uncanny* (`assets/buildings.py`, `assets/fantasy.py`): built from a level kit in which
  every part carries the range of levels it appears in. Fine detail ends early (plinth blocks, window frames and
  shutters, timber braces, thatch eave rolls, portcullis bars, sail lattices, rune strokes, vertebrae), simplified
  stand-ins start late (one plinth band, a glowing rune band, the spine as a tube), and round forms lose sides
  level by level. lod7 is a box-and-gable house, a prism tower, a single crystal spike.

Buildings (houses, windmill, keep, gatehouse, the wizard's tower) have their own class, `[lod.assets.building]`,
whose screen thresholds are 2.5× smaller than other structures', about 1.6× the switch distance: a house switches
at 28, 40, 58, 83, 120, 172 and 248 m (was 18 … 157 m), so it keeps its timber framing to about 120 m.

Trees switch from lod0 to lod1 at about 15–50 m and reach lod7 at about 135–430 m (deadwood to oak) for the reference camera
(36 mm aperture, 24 mm focal length, 16:9); the per-asset values are in the manifest.

**Domains.** Imaging everywhere. Streaming on every cell (`streaming_distance`, a `LODDistanceHeuristic` with
`lod:domain = "streaming"`; its thresholds are load radii and the custom `streaming:unloadThresholds` attribute
holds the unload radii). Physics on tiles (1 m terrain within 120 m, 2 m to 420 m) and on rocks and structures
(collision uses decimated levels, never the render lod0).

**Thresholds.** All are in `config/world.toml`: distances in metres for cells, screen fractions for assets, with
blend thresholds. Selection is verified with UsdLod's own queries; see [VALIDATION.md](VALIDATION.md).

## 4. Assets, instancing, variants

31 assets with 68 variants, plus 11 ground-cover prototypes, in `world/assets/<category>/` (trees, bushes, rocks,
logs, structures, buildings, props, fantasy). Each asset is an interface layer `<name>.usda`: defaultPrim
`/<name>` (kind component), with a variant set (`shape` / `state` / `form` / `style` / `colour`), `Geom` (LODRootAPI), `LODHeuristics` and `Materials` (library materials referenced in and bound inside
the asset). Geometry is in `<name>_geom.usdc`, referenced per variant. Variation (tree shapes, rock forms, ruin
states, house styles, banner colours) is always a variant set; LOD is never a variant.

| Category | Assets (variants) |
|---|---|
| buildings | `house` (thatch, slate, long, chapel), `windmill`, `keep`, `gatehouse` |
| props | `well`, `fence` (intact, broken), `haystack`, `gravestone` (cross, slab, round), `banner` (red, blue), `wayside_shrine` |
| fantasy | `wizard_tower`, `runestone` (a, b, way), `crystal` (a, b), `dragon_bones` |
| structures | `standing_stone`, `wall` (intact, ruined, collapsed), `tower` (intact, ruined), `column`, `shrine`, `rock_arch` |
| trees, bushes, rocks, logs | oak, birch, pine, spruce, deadwood; shrub, juniper; boulder, cliff, pillar; deadfall |

- **Placed assets**: one `instanceable` prim per placement, referencing the asset with a variant selection, so
  each species × variant becomes one prototype. The densest woodland tile holds about 4.7k such instances.
- **Ground cover**: `UsdGeomPointInstancer`, 4 chunks per tile, with prototypes under the instancer
  referencing the scatter assets (kind subcomponent), plus positions, half-float orientations, scales, ids and
  an authored extent. That is 6k–110k points per tile (70k on average).

## 5. Terrain

One global 1 m heightfield (10001², 400 MB, cached and memory-mapped). Every terrain mesh samples it: tile full levels at 1 m, tile medium
at 2 m, tile simplified and region medium at 5 m, region proxies at 10 m. Shared edges at the same level are therefore bit-identical, and coarser levels
sample a subset of the same points. Meshes are quads with `primvars:normals`, computed by central differences
of the level's own grid (extended one sample past the cell, so neighbours agree), and `primvars:st`, which is
region-relative so the region's baked texture spans its tiles and proxy.

**Cracks are hidden with skirts.** Each terrain mesh has a vertical skirt. Its depth is the worst edge deviation
of its own level plus that of any level that can neighbour the edge, plus a margin. Validation measures the
real gaps between neighbouring meshes at every pair of levels (37k edge pairs): the worst gap is 94% of the skirt depth.

## 6. Materials and textures

There are 31 UsdPreviewSurface materials in `world/materials/library.usda`:
- bark ×4, foliage ×5 (opacity from alpha, `opacityThreshold` 0.5, double-sided cards), grass, fern;
- rock, cliff rock, stone blocks, wood;
- buildings and the uncanny: plaster, thatch, roof slate, timber, banner cloth ×2 (heraldic: a field, a border
  and a lozenge), sail canvas, bone, and crystal (with a faint emissive colour, for runes, crystals and the
  spire's windows);
- terrain source materials: grass, soil, stone, mountain rock, snow;
- `terrain` and `terrain_far`, whose textures are Material interface inputs that each tile, ground chunk and
  proxy overrides with its region's (or, for the fine ground, its tile's) bake.

Shaders use UsdUVTexture with an explicit colour space and UsdPrimvarReader_float2, normal maps use
scale/bias, and the terrain's tiled detail normal and roughness go through UsdTransform2d.

Textures are procedural: 54 shared, 3 per region and 1 per tile (the fine ground), PNG/JPG, power of two. Each region's terrain colour
(2048², about 0.5 m per texel) is baked from the source materials' colours with field-driven weights and
world-space noise, so there are no artificial biome edges and colours stay consistent across LOD levels. Farm
areas add their open-field strips to the bake, so fields read from every level, down to the region proxies.

## 7. Lights and viewpoints

`layers/environment.usda` holds a DistantLight sun and a DomeLight_1 sky (lat-long texture, `poleAxis = Z`).
The sky is `textures/sky.exr`: HDR, 4096×2048 half float, linear Rec.709 radiance (2 MB). It is generated for the
`[lighting]` sun: a Preetham sky (turbidity 3) for the sun's elevation, horizon haze, clouds lit from the sun
(forward-scattering, thinned within about 20° of it), a ground hemisphere lit by sun and sky, and a limb-darkened
sun disc at the DistantLight's direction, angular size and colour. The disc carries 1% of the DistantLight's
irradiance, so a renderer can use the dome for image-based lighting without counting the sun twice; the
DistantLight stays the sun's light and shadow source. The sky hemisphere averages 0.43 (×`sky_intensity` 0.9),
the level of the earlier 8-bit sky, and the sun's surroundings reach about 4× the zenith (the disc about 650), so
a viewer without tone mapping (Storm) clips the area around the sun to white. The dome's
`customData.sky` records the model, turbidity, sun direction and disc energy fraction. Lat-long convention
(UsdLux, Z up): row 0 is the zenith, column 0 is longitude +π, longitude 0 points to −Y and π/2 to +X.
`layers/viewpoints.usda` holds `/cameras/{overlook, dense_forest, valley, grassland, mountain_ridge, four_tiles,
four_regions, landmark, village, castle, spire}`, chosen automatically from the fields and points of interest
inside the generated regions, and `/cameras/flight_00…flight_58`: a 26.8 km route (`[flight]` in the config)
along the King's Road from the southern edge past Wyrm's Rest, through Elderford and the churchyard, up to the
castle and Highmoor and down to the Spire of Aldric, about 300 m apart; then a high, fast sweep (cameras 900 m
apart, marked `customData.fast`) across the south, the west and the Greyspine foothills.

## 8. Density and workload

Per tile at default density (the densest woodland, the most rocky mountain tile and Elderford's centre):

| | Forest (R_6_0/T_0_2) | Mountain (R_9_3/T_2_1) | Village (R_4_5/T_2_1) |
|---|---|---|---|
| Instances | 4.7k (1478 trees, 2141 bushes) | 2.1k (1213 rocks) | 0.9k (14 buildings, 48 props, 273 trees) |
| Point instances | 47k | 29k | 15k |
| Triangles (instances at default level) | 1.02 M | 0.97 M | 0.30 M |

World averages per tile: 3.0k instances, 70k ground-cover points, 0.94 M triangles at the asset default level.
The whole world holds 4.8 M placed instances (0.96 M trees, 9.6k per km²; the previous 4 km world had 33k per km²)
and 113 M ground-cover points. At lod0 everywhere a region is 300 M triangles (median, 514 M at most) and the world
30 G triangles and about 1.06 TB of expanded geometry. `--density light|heavy` (0.15 / 2.5) scales every category.

## 9. Known limitations of this run

- The plain-stage renders (`<camera>_default.png`, what a consumer that ignores UsdLod sees) are no longer made:
  usdrecord loads every payload, about 10 GB for this world. `preview` can still make them for a smaller world
  (`default_too`).
- Storm and usdview don't implement UsdLod. The renders and the usdview plugin apply the UsdLod selection as
  session-layer opinions. For assets this de-instances placements that need a non-default level; that matches
  what an engine would do internally, but costs memory in usdview.
- Coarse levels instance real asset levels, so the far field costs more than merged stand-ins would: a forest
  region's proxy is around 0.45 M instanced triangles (lod7 trees at 26–48 triangles each).
- The fine ground's micro-relief is synthesized; placed assets and ground cover sit on the 1 m surface, so they
  can float or sink by up to about 0.25 m on ridge-and-furrow and rocky ground.
- Hysteresis radii are a client policy authored as a custom attribute; UsdLod itself has no notion of residency.
