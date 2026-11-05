# Validation: the whole march (100 regions, 1600 tiles) at full hierarchy depth

`generator/run.sh validate` runs every check below and writes `world/validation.json` (full details, including
per-viewpoint selections). It runs in its own memory-capped scope (`STREAMWORLD_MEM_MAX`, default 16G). Result of
this run: **all checks pass** (about 45 min, plus a re-run of `terrain_seams` and of everything but usdchecker; see
below).

| Check | Result | What was checked |
|---|---|---|
| usdchecker | PASS, 6780 files (incl. 1600 tile ground files) | `usdchecker` on every layer. Asset files are checked with all variant combinations; cell files use `--skipVariants`, because their instances' selections are placement data and every variant is already covered in its asset file. `root.usda` and the regions' `full` files are aggregates; they get the same validators (UsdValidation) on the stage with payloads unloaded, because fully loaded they need gigabytes each, and their payloaded content is checked file by file. |
| conventions | PASS | Every layer has `metersPerUnit = 1` and `upAxis = Z`; the 6777 referenced or payloaded files have a `defaultPrim`; every sublayer, reference, payload and texture path is relative and resolves inside `world/` |
| composition | PASS | 75840 references, 6600 payload arcs, 2 sublayers and 45157 asset paths resolve; zero composition errors with each region loaded in full |
| payload_bounds | PASS, 45000 payload prims (incl. 40000 fine ground levels) | Every payload prim has an `extentsHint`, and its loaded content fits inside it |
| gprim_extents | PASS, 64.4 M gprims (incl. instance proxies) | Every gprim has an authored `extent`; models in cell files carry an `extentsHint` |
| lod_structure | PASS, 1700 cell roots + 40000 ground chunks + 66 asset prototypes, 86810 heuristics | Levels are the root's children; `thresholds` has levels − 1 values in the right order; blend thresholds match in count; default index is in range; exactly the default level is visible; every heuristic bound is finite, non-empty and contains every level's content; domains are imaging, physics, audio or streaming |
| streaming | PASS, 1700 cell roots, 40000 ground chunks | Every cell root and ground chunk has a `streaming` distance heuristic with load and unload radii per boundary, and imaging threshold ≤ load ≤ unload; every level is authored with the arc the config asks for (region `full` a reference, a chunk's `base` inline, everything else a payload); at every viewpoint and flight camera, every drawn payload level is wanted by `lodselect.Streamer` |
| levels_standalone | PASS, 5100 cell levels, 80000 ground levels, 544 asset levels | Each cell level composes alone in the root stage (only that payload, the enclosing `full` levels being references) and as a file by itself; in every tile `full` level, each chunk's `fine` payload loads on its own with a bound ground and skirt, its `base` is inline, and the tile's ground file opens by itself; every asset variant's 8 levels compose with geometry and bound materials |
| collision | PASS, 41761 default colliders on, 1786 off | Every collider (`PhysicsCollisionAPI`) of every cell level, ground chunk level and asset level authors `physics:collisionEnabled`: `true` on its root's default level, `false` elsewhere; point-instanced stand-ins `false` |
| lod_viewpoints | PASS, 11 viewpoints + 59 flight cameras | The cell and ground chunk levels UsdLod's `DistanceHeuristicQuery` selects (nested roots evaluated in turn) equal the manifest's expected levels, computed from the generator's own box distances with the schema's blend semantics |
| terrain_seams | PASS, 37440 cell edge pairs, 318400 chunk edge pairs | Shared vertices of neighbouring cells at the same level are identical (max difference 0.0 m); across levels (every combination of tile full/medium/simplified and region medium/proxy that can be adjacent) the worst gap is 94% of the skirt depth. Ground chunks against their right and upper neighbours, inside a tile and across tile borders, at fine/fine, fine/base and base/base: shared vertices identical, worst gap 28% of the skirt |
| manifest_counts | PASS, 5100 levels | Instances, point instances (incl. the coarse levels' asset instancers), meshes, triangles and asset lists recomputed from each level file match the manifest |

At 100 regions, the loaded-stage checks (composition, payload bounds, gprim extents, LOD structure) no longer open the
whole world at once. The stage opens with payloads unloaded; everything outside the regions is checked as composed,
then each region is loaded with all its payloads, checked and unloaded again, so memory stays at about one region
(under 1 GB) instead of the whole world. Asset LOD roots are checked once per asset and variant across all regions.

The first full run of this version failed `terrain_seams` at 8 tile edges (a full tile against a coarser neighbour).
The check, not the terrain, was wrong: it took the shallowest skirt of a tile's 25 chunks for every edge, while each
edge belongs to one chunk whose skirt was computed for that edge. With each point's own chunk skirt the check passes
with the same 94 % worst ratio as before the chunks existed; `terrain_seams` was re-run on its own for that. Then the
ground chunks' streaming radii were raised from 75/95 m to 100/120 m in the tile files (after the replay showed late
fine-ground loads), and every check except usdchecker was run again on the patched files.

An earlier world failed `payload_bounds` and `lod_structure` in 2 of the 1600 tiles: ground-cover ferns
(up to 1.75× scale) reached 1.1 m past the tile bound, which allowed 1 m around the cell. Cell bounds now extend by
the largest ground-cover prototype's radius at the largest scale (3.0 m), and the run above includes that fix.

## LOD selection at the viewpoints

Distance heuristics (primary). The last column shows how often the screen-size heuristics, calibrated to the same
switch distances for the reference camera, pick the same level. A projected-sphere heuristic depends on each cell's
height range, and the thresholds are calibrated on one representative extent, so switch distances vary by cell;
the distance form is the reference. Most disagreements are near cells in the mountains, whose bounds are tall.

| Viewpoint | Regions as tiles | Tiles at `full` | Ground chunks at `fine` | Assets evaluated, lod0 / lod1–6 / lod7 | Screen-size agreement |
|---|---|---|---|---|---|
| overlook | R_0_8, R_0_9, R_1_8, R_1_9 | 4 | 0 | 0 / 0 / 4587 | 174/182 |
| dense_forest | R_7_8, R_7_9, R_8_8, R_8_9 | 8 | 7 | 19 / 652 / 22288 | 247/257 |
| valley | R_4_9, R_5_9, R_6_9 | 5 | 6 | 0 / 694 / 20134 | 199/207 |
| grassland | R_0_0, R_0_1, R_0_2, R_1_0, R_1_1, R_1_2 | 6 | 10 | 1 / 83 / 17374 | 206/223 |
| mountain_ridge | R_0_8, R_0_9, R_1_8, R_1_9 | 5 | 5 | 0 / 6 / 6174 | 215/248 |
| four_tiles | R_0_0, R_0_1, R_1_0 | 6 | 8 | 37 / 1275 / 26159 | 241/248 |
| four_regions | R_4_4, R_4_5, R_5_4, R_5_5 | 5 | 8 | 0 / 103 / 9574 | 253/264 |
| landmark (Elderford) | R_3_4, R_3_5, R_4_4, R_4_5, R_5_5 | 9 | 0 | 0 / 16 / 8741 | 217/232 |
| village (Elderford) | R_3_4, R_3_5, R_4_4, R_4_5, R_5_4, R_5_5 | 8 | 9 | 0 / 67 / 11491 | 221/232 |
| castle (Ravencrest) | R_5_6, R_5_7, R_6_5, R_6_6, R_6_7 | 8 | 9 | 1 / 122 / 26584 | 272/289 |
| spire (Aldric) | R_7_3, R_7_4, R_7_5, R_8_3, R_8_4, R_8_5 | 7 | 8 | 8 / 867 / 31780 | 244/257 |

The screen-size column counts cell roots and ground chunks; chunks have only distance heuristics, so the
screen-size selection falls back to them and always agrees.

## Streaming replay

`generator/run.sh stream-replay` (not part of `validate`) flies the 26.8 km flight path and simulates asynchronous
loading. The replay composes a `full` level once it is resident, so the ground chunks nested in a tile's full level
take part like any other root. At 30 m/s and 50 MB/s the streaming domain has 5 late draws (0.1 root-seconds in a
15-minute flight, all fine ground chunks; draw-only: 5667), a median lead time of 2.0 s, and cuts reloads from 1157
to 200. At 20 MB/s 8 roots are briefly late (1.2 root-seconds); at 10 MB/s 121 (48.5 root-seconds); at 5 MB/s
the fine ground (about 600 MB over the flight) saturates the channel. Late draws fall back to coarser levels (the
1 m base for a chunk), never holes.
Details: `world/stream_replay.json` and [LOD_STREAMING.html](LOD_STREAMING.html), section 11.

## usdview

`generator/usdview_check.sh` runs usdview headless (`testusdview`, Qt offscreen) on `world/root.usda`, started
with `--unloaded`, with the UsdLod plugin in streaming mode (payload residency from the streaming domain, drawing
with coarser fallback). Results are in `previews/usdview_check.json` and screenshots in `previews/usdview_*.png`.

| Step | Payloads resident | Result |
|---|---|---|
| open unloaded | 0 | root structure and every cell root (region `full` levels are references) |
| overlook | 186 | 4 regions as tiles; the 100 region proxies, prefetched and coarser fallback levels resident |
| four_regions | 212 | 4 regions as tiles |
| four_tiles | 182 | 3 regions as tiles |
| dense_forest | 205 | 4 regions as tiles, asset LOD applied |

Interactively: `generator/usdview.sh --unloaded`, then **UsdLod → Apply LOD from view** (Ctrl+L), or
**Auto-update LOD** to re-select whenever the camera moves.

## Renders

`generator/run.sh preview` renders every viewpoint LOD-aware (`docs/previews/<camera>.png`), including the new
`village`, `castle` and `spire` showcases. The plain-stage renders (`<camera>_default.png`) are no longer made,
because usdrecord loads every payload of the 10 GB world for them.
