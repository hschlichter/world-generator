# Task: Generate a Large Fantasy Open World in USD

## This run

- **Work only in this directory.** It is a project of its own: the generator, its configuration and the generated world all live here. Don't
  read, search or depend on anything outside it, except the OpenUSD build and the system tools below. The world is for an open-world engine you
  don't need to know about: author it as best-practice USD that any USD-based engine or tool can load (usdview, Omniverse, Houdini, Blender).
- **Stop after section 8 step 4 and report**: one representative region at the full hierarchy depth, validated (section 10). Don't scale to
  the full world in this run; that comes after the region has been reviewed.
- **Layout**: the generator in `generator/` (Python; a README with how to run it and its parameters), configuration in `config/`, the generated
  world in `world/` with the root stage at `world/root.usda`. `world/` is generated output: git-ignored if this directory becomes a repository,
  and regenerated from the generator.
- **The machine**:
  - OpenUSD 26.08 with Python, imaging, usdview and the USD tools is built in `openusd_build/` here. Use it, not the system's OpenUSD (26.05,
    which has no UsdLod):

    ```sh
    export PYTHONPATH=$PWD/openusd_build/lib/python3.14/site-packages
    export PATH=$PWD/openusd_build/bin:$PATH
    python3 -c "from pxr import Usd, UsdLod; print(Usd.GetVersion())"
    ```

    `usdview`, `usdrecord`, `usdchecker` and `usdcat` are in its `bin`. `usdrecord` renders headless with Storm under
    `QT_QPA_PLATFORM=offscreen`. UsdLod's Python API is `UsdLod.RootAPI`, `UsdLod.ScreenSizeHeuristic`,
    `UsdLod.DistanceHeuristic`, `UsdLod.OverrideAPI` and the heuristic query classes.
  - Blender 5.2 (`/usr/bin/blender`) and numpy are installed; no `pip` or `uv`. Don't install system packages without asking.

## Objective

Create a complete, visually coherent, fantasy-inspired 3D environment for testing an open-world game engine.

Its main purpose is to exercise world streaming, level of detail, visibility culling, instancing and large-scale scene management under
realistic conditions.

It must be an actual generated environment, not a scene specification, a collection of placeholders or a technical demonstration.

The finished world should be visually interesting, and suitable for exploring freely on foot or with a flying camera.

## 1. World design

Minimum dimensions: 4 km × 4 km (16 km²).

Create a believable, continuous fantasy landscape with several distinct regions:

- **Open grasslands**: large rolling plains, meadows, tall grass, wildflowers and scattered trees, with dense ground vegetation in places.
- **Dense forests**: extensive woodland with several tree species, dense canopies, undergrowth, rocks, fallen trees and natural clearings.
- **Mountains**: substantial ranges with cliffs, rocky outcrops, steep slopes, ridgelines and distant peaks.
- **Transitional landscapes**: forest edges, foothills, valleys and natural terrain transitions.
- **Points of interest**: occasional fantasy landmarks such as ancient ruins, standing stones, abandoned structures or unusual rock formations.

Aim for a grounded, slightly stylized fantasy look, not exaggerated cartoon visuals.

The landscape should feel naturally formed. Avoid obvious procedural repetition, uniformly distributed vegetation and artificial biome
boundaries.

### World composition

Design the world to produce a range of runtime situations:

- Long, unobstructed views across open terrain.
- Dense forests with high object counts and substantial occlusion.
- Elevated viewpoints overlooking large parts of the map.
- Valleys with complex visibility.
- Rapid transitions between open and densely populated areas.
- Distant mountains that stay visible over several kilometres.

Include natural routes through the world, so it can be explored continuously.

## 2. USD scene architecture

The deliverable is a valid OpenUSD scene that uses the standard UsdLod schema and follows USD's best practices for large, streamable worlds
(section 9).

- A root stage assembles the complete environment.
- The world is partitioned into spatially organized cells, in a hierarchy (section 9.3).
- Payloads let cells load and unload independently; any cell loads without the rest of the world.
- Asset definitions are reusable and separate from scene placement.
- References, payloads, instancing, point instancing and variant sets are used where each is the right tool.
- Everything that can be unloaded has authored bounds; coordinate conventions are consistent.
- No unnecessarily large scene graphs or excessive individual prims.
- The partitioning is configurable: cell sizes, hierarchy depth and level counts are parameters, not constants.
- The structure stays readable and suitable for automated processing.

## 3. Level of detail

Use the standard UsdLod schemas (OpenUSD 26.08), not an invented LOD convention, and use them as they are meant to be used:

- `LODRootAPI` on cells and on significant assets; a root's levels are its direct children, finest first.
- Heuristics that fit each root: `LODScreenSizeHeuristic`, `LODDistanceHeuristic`, or both, with authored bounds, thresholds and, where they
  help, blend thresholds.
- A meaningful default level (`lod:default:index`), and correct results in a consumer that doesn't understand LOD.
- Domains where they apply: imaging at least; physics where collision geometry differs.

Provide multiple levels for:

- Terrain.
- Trees and other significant vegetation.
- Rocks and cliffs.
- Structures and landmarks.
- Dense vegetation clusters, where it makes sense.

Each level contains genuinely different geometry, not copies of the same mesh. Use hierarchical LOD: a cell's coarse level replaces everything
in the cell with merged, simplified geometry (terrain, forest canopy, landmark silhouettes), so the distant world is a few meshes per cell, not
thousands of far-away instances. Keep thresholds configurable, so they can be tuned against an engine.

## 4. Geometry and asset generation

The environment contains actual geometry and materials, not procedural shaders or external runtime systems. Use procedural generation wherever
practical.

### Terrain

- Continuous terrain across the entire world.
- No visible cracks between neighbouring cells, including when neighbours are at different levels.
- Different geometric resolutions per terrain level.
- Varied elevation and slope, erosion-like features and natural formations.

### Vegetation

- Several tree types, varying in height, shape, scale and orientation.
- Dense forests with large numbers of tree instances.
- Grass, bushes, shrubs and other small vegetation.
- Spatially coherent distribution that depends on the biome.
- Instancing to avoid duplicating assets (section 9.5).

### Materials

- Terrain materials for grass, soil, stone and mountain surfaces.
- Bark, foliage, rock and vegetation materials.
- Consistent art direction.
- Reusable materials and textures.

## 5. Performance and streaming test coverage

The world should deliberately expose a broad range of rendering workloads, including:

- Very high vegetation instance counts.
- Large numbers of individually cullable objects.
- Dense and sparse geometry distributions.
- Large differences between near and distant LOD complexity.
- Long-distance views spanning many cells.
- Traversal across many cell boundaries, including corners where four cells meet.
- Fast camera movement across the world.
- Significant overlap between visible streamed regions.

Don't optimize away the challenging cases: the environment must stay useful as a performance and scalability test. But don't create unnecessary
duplication or pathological USD structures just to add workload.

Make scene density and geometric complexity configurable (section 9.10 gives the default targets).

## 6. Generation workflow

Prefer a reproducible, procedural asset-generation pipeline over extensive manual modelling. You may use Blender, Python, OpenUSD, procedural
geometry libraries and mesh simplification tools. The generated files must not depend on those tools at runtime.

- Generation is deterministic for a given random seed.
- Individual cells and assets can be regenerated on their own, rewriting only their own files.
- Useful configuration parameters are exposed.
- Real, usable USD assets are written to disk.
- Generation tooling is kept separate from the scene data.
- No dependency on online services or externally hosted assets at runtime.
- Use existing open-source tooling rather than implementing complex geometry algorithms from scratch.

## 7. Deliverables

- **The complete USD world**: the root stage and every file it references, organized for streaming.
- **The generation pipeline**: scripts, configuration and instructions to regenerate the world.
- **LOD assets**: real alternative geometry per level, through UsdLod.
- **Materials**: every material and texture used.
- **Viewpoints**: named cameras for testing, and an ordered flight path across the world (section 9.9).
- **A scene manifest**: bounds, the cell hierarchy, and per cell and level its file, asset, instance and point-instance counts and triangle
  count. Totals for the far field (every root at its coarsest level) and for the world at its finest.
- **Validation tooling** (section 10).
- **Previews**: renders from the viewpoints (`usdrecord`), and screenshots from usdview.
- **Documentation** of the resulting structure and the important design decisions.

## 8. Implementation approach

1. Design the world layout, biomes and procedural generation approach.
2. Define the USD scene organization and partitioning strategy.
3. Implement a small representative area (one region: a few cells with terrain, vegetation and working LOD) at the full hierarchy depth.
4. Validate its USD structure, LOD levels and independently loadable payloads (section 10). Stop here and report (see "This run").
5. Scale the generation to the complete 4 km × 4 km world.
6. Validate the complete generated environment.

Don't generate the entire world before the USD structure is shown to work.

## 9. USD best practices

Follow established USD practice (Pixar's documentation, the AOUSD and ASWF asset structure guidelines, NVIDIA's guidance for large scenes).
Where a default below is a choice, it is a parameter of the generator, documented in the manifest.

### 9.1 Stage conventions

- `metersPerUnit = 1` and `upAxis = "Z"` on every layer; `defaultPrim` on every file that is referenced or payloaded.
- The world centred on the origin, so coordinates stay small enough for single-precision consumers.
- Relative asset paths only, resolving inside `world/`: no absolute paths, search paths, URLs or resolver configuration.
- Binary `.usdc` for heavy geometry; `.usda` where a human reads it (the root stage, cell structure).

### 9.2 Model hierarchy and assets

- `kind` throughout: the world and cells are `assembly` and `group`, placed assets are `component`, with `subcomponent` where useful.
- Assets follow a standard asset structure: one root prim (the `defaultPrim`), geometry and materials under it in their own scopes, materials
  bound inside the asset so it renders the same wherever it is referenced.
- An asset with variation (tree species shapes, rock forms, ruin states) uses **variant sets** for it; LOD is UsdLod, never a variant set.

### 9.3 Cells: the streaming hierarchy

Default: three levels of LOD roots, each switching its own scale of content.

- **Regions**, 1 km × 1 km (16 in the world): each a LOD root with two levels, each level a payload of its own file:
  - `full`: the region's tiles.
  - `proxy`: one merged, simplified representation of the whole region: low-resolution terrain, forest canopy as a few meshes, landmark
    silhouettes. A few thousand to some tens of thousands of triangles, a handful of meshes.
- **Tiles**, 250 m × 250 m (16 per region): each a LOD root inside its region's `full` level, with two levels, each a payload:
  - `full`: the tile's terrain at full resolution and everything placed in it.
  - `simplified`: terrain at reduced resolution and the large assets only, or merged clusters.
- **Assets** (trees, rocks, structures): a LOD root inside the asset, two or three levels. They are only loaded inside a tile's `full` level.

Why this shape:

- **Every level switch has a bounded cost.** Going finer replaces one cell's coarse level with its next level, never a whole region's detail at
  once: a region's `full` level holds tiles at their `simplified` level until the camera comes closer.
- **The far field is small.** With every root at its coarsest level, the world is its region proxies.
- **LOD roots exist only near the camera.** Asset roots live in tile `full` levels, so their number follows what is near.
- **Payload granularity.** Each level of a region or tile is one file. Avoid both extremes: a region as one file, or thousands of tiny payloads
  per cell.

### 9.4 UsdLod

- Levels as direct children of their root, finest first; each level of a cell a payload.
- Each heuristic bounds its root: an `extent` (or a `boundingVolume`) that is finite, non-empty and contains the root's content in every level.
- `thresholds` with one value fewer than the root has levels; defaults to start from, documented in the manifest.
- `lod:default:index` chosen so that a consumer that ignores the heuristics gets a sensible world (state which, and why).
- Verify the selection with UsdLod's own heuristic queries from several camera positions, and record the result (section 10).

### 9.5 Instancing

- **Placed assets** (trees, rocks, structures): native instancing, an `instanceable` prim per placement referencing the asset.
- **Scatter** (grass, flowers, small rocks, undergrowth, forest fill where it fits): `UsdGeomPointInstancer`, with prototypes under it and
  per-instance positions, orientations, scales and ids; bounds authored.
- Keep assets small and shared: a few dozen asset files make up the world, placed hundreds of thousands of times.

### 9.6 Terrain

- Triangle or quad meshes, one per tile per level and one per region proxy, with normals and texture coordinates (`primvars:normals`,
  `primvars:st`).
- Cracks between neighbouring levels hidden with skirts or matched edges; document which.

### 9.7 Materials and textures

- `UsdPreviewSurface` with the inputs that matter for the look: diffuse colour, roughness, metallic, normal maps, opacity and
  `opacityThreshold` for alpha-tested foliage. `UsdUVTexture` and `UsdPrimvarReader_float2` networks, colour spaces authored.
- Textures as PNG or JPG (EXR only where needed), power of two, shared and atlased where sensible; a modest count for the whole world.
- Bindings on asset roots or `GeomSubset`s, inside the asset.

### 9.8 Bounds

- `extent` on every gprim, `extentsHint` on models and on every payload prim, so a consumer can judge an unloaded cell.

### 9.9 Lights and viewpoints

- A `DistantLight` as the sun, and a `DomeLight` for the sky (a texture, or a constant colour).
- `UsdGeomCamera` prims under `/cameras`, named for what they show: an overlook of the whole map, inside dense forest, a valley, open
  grassland, a mountain ridge, a point where four tiles meet, a point where four regions meet, and a landmark.
- A flight path: `/cameras/flight_00`, `flight_01`, … in order, a few hundred metres apart, crossing biomes, region and tile boundaries, with at
  least one long fast traverse.

### 9.10 Density targets

Defaults, all configurable, for a world that fits in memory when streamed and doesn't when loaded at its finest:

- **The far field** (every region at `proxy`): under 1 M triangles.
- **A tile at `full`**: under 20,000 instances (point instances counted separately) and 2 M triangles at default density; dense forest tiles at
  the top of that range, grassland and mountain tiles well below it.
- **The world at its finest**: well over 8 GB of geometry.
- A density parameter scales instance counts per biome, so the same world can be generated light (for iteration) and heavy (for stress).

## 10. Validation

- `usdchecker` passes on the root stage and on every file.
- Every reference and payload resolves; every payload prim has `extentsHint`; every LOD heuristic has a finite, non-empty bound; every level
  of every root loads on its own.
- LOD selection from UsdLod's heuristic queries at the viewpoints matches what the manifest says each viewpoint should see.
- Terrain seams: neighbouring tiles' edges match (or are skirted) at every pair of levels.
- The manifest's counts recomputed from the stage.
- usdview opens the root stage, and loads a region and a tile in it on demand (payload loading), with LOD working.

## Success criteria

The result is a visually recognizable fantasy landscape that can be explored continuously across at least 16 km².

It has enough geometric diversity and density to meaningfully test large-world streaming and LOD.

Most importantly, the output is an actual, valid, self-contained USD environment that an engine can load and inspect, not a conceptual prototype
or a collection of disconnected example assets.
