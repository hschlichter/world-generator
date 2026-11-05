"""Generation jobs that run in worker processes (kept out of __main__ so they can be pickled)."""

import json
import os

from .config import Config


def _stats_path(cfg, cell, level):
    return os.path.join(cfg.out, "stats", f"{cell.file_stem}_{level}.json")


def _save_stats(cfg, cell, level, stats):
    p = _stats_path(cfg, cell, level)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    stats = dict(stats, cell=cell.id, level=level)
    with open(p, "w") as f:
        json.dump(stats, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def _gen_leaf(config_path, overrides, cell_id):
    """Worker: write both levels of one finest cell."""
    from .cells import Generator
    from .terrain import World
    cfg = Config(config_path, overrides)
    world = World(cfg)
    gen = Generator(cfg, world)
    cell = cfg.cell_by_id(cell_id)
    _save_stats(cfg, cell, "full", gen.write_leaf_full(cell))
    for spec in cell.level.coarse:
        _save_stats(cfg, cell, spec["name"], gen.write_coarse(cell, spec))
    return cell_id
