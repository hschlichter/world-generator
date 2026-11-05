# Eldermark generator (`streamworld`)

Procedural generator for a 10 km × 10 km medieval fantasy landscape authored as streamable OpenUSD with UsdLod.
Design and USD structure: [`docs/DESIGN.md`](../docs/DESIGN.md). Validation: [`docs/VALIDATION.md`](../docs/VALIDATION.md).

## Requirements

Everything comes from this directory or the machine: the OpenUSD 26.08 build in `openusd_build/` (with UsdLod),
Python 3.14 with numpy, OpenImageIO (texture I/O) and PySide6 (usdview), and Blender 5.2 (`blender` on `PATH`,
override with `BLENDER=`). Blender only runs at generation time, for mesh decimation and convex hulls.
Nothing in `world/` depends on any of these tools.

## Running

```sh
generator/run.sh generate                     # regions from config world.generate_regions (R_4_5, Elderford)
generator/run.sh generate --regions all       # the whole world
generator/run.sh generate --regions R_4_5 R_6_6
generator/run.sh generate --cells R_4_5/T_2_1 # regenerate one tile; it rewrites only that tile's own files
generator/run.sh generate --density light     # light | default | heavy | <scale>
generator/run.sh generate --force-assets      # rebuild the asset library (assets are otherwise reused)
generator/run.sh manifest                     # rebuild world/manifest.json
generator/run.sh validate [--quick]           # every check; writes world/validation.json (--quick skips usdchecker)
generator/run.sh stream-replay [--speed 30] [--bandwidth 50] [--latency 0.05]
                                              # fly the flight path, simulate payload streaming -> world/stream_replay.json
generator/run.sh preview [--cameras a,b]      # LOD-aware usdrecord renders -> docs/previews/
generator/run.sh map                          # overview map -> docs/previews/map.png
generator/run.sh lodtree [--species oak] [--render]   # 8-level UsdLod test tree -> world/assets/lodtest/
generator/usdview.sh [world/root.usda] [--unloaded]   # usdview with the UsdLod plugin (menu "UsdLod")
generator/usdview_check.sh                    # headless usdview run: payload loading + LOD, screenshots
```

`generator/env.sh` sets `PYTHONPATH`/`PATH` for interactive use (`. generator/env.sh`).

Timings on this machine (24 cores): the terrain is built once per terrain configuration and cached in
`build/cache/` (the eroded macro grid about 9 min, keyed by `[terrain]` only, without `[terrain.ground_lod]`; then the 1 m heightfield with levelled
settlement ground and roads about 3 min, and the biome fields about 3 min). Assets take about 20 s, one region (16
tiles with all levels and their fine ground and textures, proxy, bake and structure files) about 35 s, and the whole
100-region world about 50 min plus about 11 min for the root and manifest (24 GB, of which 14 GB is the fine ground).
Run it with no more than about 16 GB of memory to spare beside other applications: 8 worker processes peak around
1–1.5 GB each. Validation of the whole world takes about 45 min, `stream-replay` about 8 min.

## Determinism and partial regeneration

Every random choice is a hash of the seed and a world-space integer coordinate, or an RNG seeded from the seed
and a stable name, so output is identical for a given seed and configuration. Placement is world-anchored, not
per cell (see `scatter.py`), so a tile's content doesn't depend on its neighbours or on the partition. Each
cell's bounds come from the same deterministic functions, so regenerating one cell never requires rewriting its
parent. `--cells` writes only that cell's level files and its `world/stats/` entries; run `manifest` afterwards
to refresh the totals.

## Configuration (`config/world.toml`)

| Section | What it controls |
|---|---|
| `[world]` | name, seed, size (world spans ±size/2), output and cache directories, regions to generate |
| `[partition]` | `default_level` (coarsest/finest: `lod:default:index` of cells); `[[partition.levels]]`: the cell hierarchy, coarsest first. Each level has a size (it must divide the parent), `coarse` levels (name, terrain quads, terrain material near/far, the instance filters `trees_min_height` / `bushes_min_height` / `min_size`, texture size), distance and screen-size thresholds and blend thresholds, streaming load and unload radii, physics thresholds, and `full_arc` (payload or reference). Depth = number of levels; LOD level count = 1 + number of coarse entries. |
| `[terrain]` | heightfield spacing, full-level quads per tile, skirt margin/min, lowland swell height, erosion grid/droplets per km²/steps, baked texture size, detail tiling; `[terrain.ground_lod]` ground chunks (chunk size, fine spacing, fine texture size per tile, micro-relief scale, imaging and streaming thresholds); `[[terrain.ranges]]` mountain ridge polylines (height, width), `[[terrain.valleys]]` |
| `[[biomes.areas]]` | designed open land, farmland (open-field strips, `angle`) and woodland on top of the noise-driven forest field |
| `[[pois]]` | points of interest: kind, position, radius (cleared of trees; settlements and built sites also level the ground). Settlements: village / hamlet (`houses`), castle, windmill, graveyard. Ruins: ruin_keep, stone_circle, tower, rock_arch, shrine, colonnade, rock_pillars. The uncanny: wizard_tower, dragon_bones, crystal_hollow, runestones |
| `[[paths]]`, `[markers]` | roads and footpaths (carved into the terrain, kept free of vegetation, drawn into the terrain texture); waystones and wayside shrines beside them every `spacing` metres |
| `[density]` | `scale`, per-hectare densities by category, ground cover per m², point-instancer chunks per tile edge, presets |
| `[lod]` | reference camera; asset default level; per asset class (tree, bush, rock, log, structure, building) screen-size thresholds/blend and physics distance thresholds; `[lod.lodtest]` the 7 screen thresholds and blend thresholds of the 8-level test tree |
| `[flight]` | the test flight: waypoints `[x, y, height above ground, fast]` |
| `[lighting]`, `[preview]` | sun direction/intensity/colour, sky intensity; preview image width |

## Modules

| Module | Role |
|---|---|
| `config.py` | TOML loading, the cell partition (`Cell`, `CellLevel`) |
| `noise.py` | hash-based vectorized noise (perlin, fBm, ridged, domain warp) |
| `terrain.py` | macro layout, windowed droplet + thermal erosion, cubic upsampling, detail, levelled settlement ground, path carving; biome fields (incl. farmland) |
| `textures.py`, `materials.py`, `atlas.py` | procedural textures (incl. plaster, thatch, slate, timber, heraldic cloth, crystal, bone), the per-region terrain bake (with open-field farm strips), the UsdPreviewSurface library |
| `assets/lodtree.py` | the 8-level UsdLod test tree (one world species skeleton, hierarchical card merging, crown hulls), its lineup stage and switch renders with a pop metric |
| `assets/` | trees and bushes (`trees.py`), rocks (`rocks.py`), ruins (`structures.py`), houses, windmill, keep, gatehouse and village props (`buildings.py`), the wizard's spire, runestones, crystals and dragon bones (`fantasy.py`), their level kit (`kit.py`), ground cover and deadfall (`plants.py`), the asset writer (`library.py`) |
| `blender.py`, `blender_ops.py` | batched, cached Blender jobs: quadric decimation, convex hulls |
| `scatter.py`, `pois.py` | biome-driven placement; layouts of settlements, the castle and every other point of interest, path markers |
| `cells.py` | terrain meshes with skirts, tile / region levels (coarse levels as point instances of asset levels), ground chunk roots and the tiles' ground files, LOD roots, heuristics, root stage |
| `ground.py` | ground LOD: chunk cells, Catmull-Rom fine heights with synthesized micro-relief, the per-tile fine texture bake |
| `viewpoints.py` | sun and sky dome (the HDR sky itself is `textures.sky`), named cameras, flight path |
| `lodselect.py` | LOD selection with UsdLod queries, the streaming policy (`Streamer`), session-layer application (standalone, no generator dependency) |
| `streamreplay.py` | flight-path streaming simulation: draw-only vs streaming-domain residency, late draws, churn, bandwidth |
| `manifest.py`, `validate.py`, `preview.py`, `mapview.py` | manifest, validation, renders, map |
| `usdview_plugin/` | usdview plugin (menu "UsdLod") and the headless usdview check |
