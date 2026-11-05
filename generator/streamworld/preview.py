"""Preview renders of the viewpoints with usdrecord (Storm), LOD-aware.

Storm does not implement UsdLod, so for each camera the LOD selection is computed with UsdLod's heuristic queries
(lodselect) and written to a session layer that usdrecord applies (--sessionLayer): selected levels visible,
unselected cell levels deactivated (not composed, as if streamed out), assets that need a non-default level
de-instanced. `<camera>_default.png` (optional) renders the plain stage the way a consumer that ignores LOD sees it.
"""

import os
import subprocess
import time

from pxr import Sdf, Usd

from . import lodselect as ls
from .cells import root_path

DEFAULT_CAMERAS = ["overlook", "dense_forest", "valley", "grassland", "mountain_ridge", "four_tiles",
                   "four_regions", "landmark", "village", "castle", "spire"]


def _record(root, camera, out_png, width, session=None):
    cmd = ["usdrecord", "--camera", camera, "--imageWidth", str(width), "--enableDomeLightVisibility",
           "--complexity", "high"]
    if session:
        cmd += ["--sessionLayer", session]
    cmd += [root, out_png]
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(out_png):
        raise RuntimeError(f"usdrecord failed for {camera}:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def render_previews(cfg, cameras=None, width=None, default_too=()):
    """LOD-aware renders of the cameras. `default_too` also renders those cameras on the plain stage; usdrecord loads
    every payload for that, which no longer fits in memory for the 100-region world, so it is off by default."""
    width = width or int(cfg.c.preview.image_width)
    root = root_path(cfg)
    out_dir = os.path.join(cfg.root, "docs", "previews")
    ses_dir = os.path.join(cfg.root, "build", "preview_sessions")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ses_dir, exist_ok=True)
    cams = cameras or DEFAULT_CAMERAS
    report = {}
    for cam in cams:
        t0 = time.time()
        st = Usd.Stage.Open(root, Usd.Stage.LoadNone)
        cpath = cam if cam.startswith("/") else f"/cameras/{cam}"
        if not st.GetPrimAtPath(cpath):
            print(f"[preview] {cam}: no such camera (its point of interest is outside the generated regions)")
            continue
        fr, _ = ls.camera_frustum(st, cpath)
        sel = ls.Selector(st).select(fr, load=True, assets=True)
        layer = Sdf.Layer.CreateAnonymous(".usda")
        ls.apply_to_session(st, sel, layer=layer, deactivate=True)
        name = cpath.split("/")[-1]
        ses = os.path.join(ses_dir, f"{name}.usda")
        layer.Export(ses)
        png = os.path.join(out_dir, f"{name}.png")
        _record(root, cpath, png, width, ses)
        full = [p.split("/")[-1] for p, c in sel.cells.items() if c.index == 0 and c.levels != ["fine", "base"]]
        fine = sum(1 for c in sel.cells.values() if c.index == 0 and c.levels == ["fine", "base"])
        report[name] = dict(full_cells=full, fine_ground_chunks=fine, assets=sel.asset_histogram(),
                            seconds=round(time.time() - t0, 1))
        print(f"[preview] {name}: full {full}, {fine} fine ground chunks, asset levels {sel.asset_histogram()} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if name in default_too:
            _record(root, cpath, os.path.join(out_dir, f"{name}_default.png"), width)
    return report
