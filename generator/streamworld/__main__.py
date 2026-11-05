"""Eldermark world generator.

    generator/run.sh generate [--regions R_2_2 ...|all] [--cells R_2_2/T_1_3 ...] [--density light|default|heavy|<x>]
                              [--force-assets] [--force-textures] [--jobs N]
    generator/run.sh manifest            rebuild world/manifest.json from the stage and the per-cell stats
    generator/run.sh validate            run every check (docs/VALIDATION.md); writes world/validation.json
    generator/run.sh preview [--cameras overlook,valley,...]   LOD-aware renders of the viewpoints (usdrecord)
    generator/run.sh map                 top-down overview map (docs/previews/map.png)
    generator/run.sh stream-replay [--speed 30] [--bandwidth 50] [--latency 0.05]
                                         fly the flight path, simulate payload streaming (world/stream_replay.json)
    generator/run.sh lodtree [--species oak] [--variant 0] [--render]
                                         the 8-level UsdLod test tree (world/assets/lodtest/)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from .config import Config
from .jobs import _gen_leaf, _save_stats


def _cfg(args):
    over = {}
    if getattr(args, "density", None):
        cfg0 = Config(args.config)
        presets = cfg0.c.density.presets
        over["density.scale"] = float(presets[args.density]) if args.density in presets else float(args.density)
    return Config(args.config, over)


def cmd_generate(args):
    from . import materials, textures
    from .assets import library
    from .cells import Generator
    from .terrain import World
    from .viewpoints import write_environment, write_viewpoints
    cfg = _cfg(args)
    overrides = {"density.scale": cfg.density_scale()}
    t0 = time.time()
    world = World(cfg)
    textures.build_shared(cfg, force=args.force_textures)
    materials.write_library(cfg)
    library.write_all(cfg, force=args.force_assets)
    gen = Generator(cfg, world)
    names = args.regions or cfg.c.world.generate_regions
    tops = cfg.top_cells()
    regions = tops if names in ("all", ["all"]) else [c for c in tops if c.name in names]
    only = set(args.cells or [])

    def wanted(cell):
        return not only or any(cell.id == o or cell.id.startswith(o + "/") for o in only)

    for r in regions:
        textures.bake_region(cfg, world, r, force=args.force_textures)
        leaves = [l for l in cfg.leaf_cells(r) if wanted(l)]
        if args.jobs > 1 and len(leaves) > 1:
            with ProcessPoolExecutor(args.jobs) as ex:
                for cid in ex.map(_gen_leaf, [cfg.path] * len(leaves), [overrides] * len(leaves),
                                  [l.id for l in leaves]):
                    print(f"[generate] {cid} done", flush=True)
        else:
            for l in leaves:
                _gen_leaf(cfg.path, overrides, l.id)

        # Coarser cells, finest first (generic in the hierarchy depth).
        def upper(cell):
            kids = cfg.children(cell)
            if not kids:
                return
            for k in kids:
                upper(k)
            if wanted(cell) or not only:
                _save_stats(cfg, cell, "full", gen.write_cell_full(cell))
                for spec in cell.level.coarse:
                    _save_stats(cfg, cell, spec["name"], gen.write_coarse(cell, spec))
        upper(r)
    if only:
        print(f"[generate] cells {sorted(only)} done in {time.time() - t0:.0f}s (root, layers and manifest untouched; "
              f"run `manifest` to refresh it)", flush=True)
        return
    # The root lists every region whose level files exist.
    from .cells import cell_level_path
    present = [c for c in tops if all(os.path.exists(cell_level_path(cfg, c, ln)) for ln in c.level.lod_names)]
    write_environment(cfg)
    write_viewpoints(cfg, world, present)
    gen.write_root(present)
    from .manifest import build_manifest
    build_manifest(cfg, world)
    print(f"[generate] done in {time.time() - t0:.0f}s", flush=True)


def cmd_manifest(args):
    from .manifest import build_manifest
    from .terrain import World
    cfg = _cfg(args)
    build_manifest(cfg, World(cfg))


def cmd_validate(args):
    from .validate import run_all
    cfg = _cfg(args)
    ok = run_all(cfg, quick=args.quick)
    sys.exit(0 if ok else 1)


def cmd_preview(args):
    from .preview import render_previews
    cfg = _cfg(args)
    render_previews(cfg, args.cameras.split(",") if args.cameras else None, args.width)


def cmd_map(args):
    from .mapview import render_map
    from .terrain import World
    cfg = _cfg(args)
    out = os.path.join(cfg.root, "docs", "previews", "map.png")
    render_map(cfg, World(cfg), out, 1024)
    print(out)


def cmd_stream_replay(args):
    from .streamreplay import run_replay
    run_replay(_cfg(args), speed=args.speed, bandwidth=args.bandwidth, latency=args.latency, dt=args.dt)


def cmd_lodtree(args):
    from . import materials, textures
    from .assets import lodtree
    cfg = _cfg(args)
    textures.build_shared(cfg)
    materials.write_library(cfg)
    lodtree.write(cfg, args.species, args.variant)
    if args.render:
        lodtree.render(cfg, args.species, args.width)


def main():
    ap = argparse.ArgumentParser(prog="streamworld", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "config", "world.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--regions", nargs="*")
    g.add_argument("--cells", nargs="*", help="regenerate only these cells (e.g. R_2_2/T_1_3), their files only")
    g.add_argument("--density")
    g.add_argument("--force-assets", action="store_true")
    g.add_argument("--force-textures", action="store_true")
    g.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 3)))
    g.set_defaults(fn=cmd_generate)
    m = sub.add_parser("manifest")
    m.set_defaults(fn=cmd_manifest)
    v = sub.add_parser("validate")
    v.add_argument("--quick", action="store_true", help="skip usdchecker on every file")
    v.set_defaults(fn=cmd_validate)
    p = sub.add_parser("preview")
    p.add_argument("--cameras")
    p.add_argument("--width", type=int, default=None)
    p.set_defaults(fn=cmd_preview)
    mp = sub.add_parser("map")
    mp.set_defaults(fn=cmd_map)
    sr = sub.add_parser("stream-replay")
    sr.add_argument("--speed", type=float, default=30.0, help="camera speed, m/s")
    sr.add_argument("--bandwidth", type=float, default=50.0, help="simulated read bandwidth, MB/s")
    sr.add_argument("--latency", type=float, default=0.05, help="simulated latency per request, s")
    sr.add_argument("--dt", type=float, default=0.1, help="simulation step, s")
    sr.set_defaults(fn=cmd_stream_replay)
    lt = sub.add_parser("lodtree")
    lt.add_argument("--species", default="oak")
    lt.add_argument("--variant", type=int, default=0)
    lt.add_argument("--render", action="store_true", help="usdrecord the lineup and each switch -> docs/previews/lodtest/")
    lt.add_argument("--width", type=int, default=1280)
    lt.set_defaults(fn=cmd_lodtree)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
