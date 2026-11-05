"""testusdview script: usdview opens the root stage unloaded, then loads a region and its tiles on demand through
the UsdLod plugin from several cameras, saving viewport and window screenshots and a JSON report.

    generator/usdview_check.sh            (writes docs/previews/usdview_*.png and world/usdview_check.json)
"""

import json
import os

from pxr import Usd

OUT = os.environ.get("SW_USDVIEW_OUT", ".")


def _settle(app_controller, n=6):
    from pxr.Usdviewq.qt import QtWidgets
    for _ in range(n):
        QtWidgets.QApplication.processEvents()
        if app_controller._stageView:
            app_controller._stageView.updateView(resetCam=False, forceComputeBBox=False)
        QtWidgets.QApplication.processEvents()


def _shot(api, name, window=False):
    img = api.GrabWindowShot() if window else api.GrabViewportShot()
    path = os.path.join(OUT, f"usdview_{name}.png")
    if img is not None:
        img.save(path, "PNG")
    return path


def testUsdviewInputFunction(appController):
    import streamworldUsdview as plug
    api = appController._usdviewApi
    stage = api.stage
    report = dict(steps=[])

    def step(name, camera, apply=True):
        appController._dataModel.viewSettings.cameraPrim = stage.GetPrimAtPath(camera)
        _settle(appController)
        sel = plug.apply_lod(api, quiet=True) if apply else None
        _settle(appController, 10)
        loaded = sorted(str(p) for p in stage.GetLoadSet())
        rec = dict(name=name, camera=camera, loaded_payloads=loaded,
                   full_cells=sorted(p.split("/")[-1] for p, c in sel.cells.items() if c.index == 0) if sel else [],
                   asset_levels=sel.asset_histogram() if sel else {}, viewport=_shot(api, name))
        report["steps"].append(rec)

    report["initial_load_set"] = sorted(str(p) for p in stage.GetLoadSet())
    step("unloaded", "/cameras/overlook", apply=False)
    step("overlook", "/cameras/overlook")
    step("four_regions", "/cameras/four_regions")
    step("four_tiles", "/cameras/four_tiles")
    step("dense_forest", "/cameras/dense_forest")
    _shot(api, "window", window=True)
    plug.reset(api)
    with open(os.path.join(OUT, "usdview_check.json"), "w") as f:
        json.dump(report, f, indent=1)
