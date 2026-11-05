"""usdview plugin: UsdLod selection and payload streaming driven by the current camera.

Storm does not implement UsdLod. This plugin evaluates every LOD root's heuristics with UsdLod's own queries
from the viewport camera and writes session-layer visibility so exactly the selected level of each root is drawn.
With streaming on, payload residency follows the cells' `streaming` heuristics (load radii with unload hysteresis,
lodselect.Streamer) and drawing falls back to the nearest coarser resident level; with streaming off, the drawn
levels are loaded on demand and nothing is unloaded. Asset
LOD roots inside instances are switched by de-instancing only the placements that need a non-default level.

Menu "UsdLod":
  Apply LOD from view          (Ctrl+L)  one selection pass from the current camera
  Auto-update LOD                         re-run whenever the camera moves
  Stream (streaming domain)               load / unload payloads by the streaming heuristics
  Include asset LOD                       also switch tree / rock / structure LOD inside loaded tiles
  Report LOD selection                    print the current selection (also in the status bar)
  Reset to defaults                       drop the plugin's session opinions (lod:default:index again)
"""

from pxr import Sdf, Tf, Usd
from pxr.Usdviewq.plugin import PluginContainer
from pxr.Usdviewq.qt import QtCore

from streamworld import lodselect as ls


class _State:
    auto = False
    stream = True
    assets = True
    timer = None
    last_cam = None
    last = None
    streamer = None


S = _State()


def _clear_session(stage):
    layer = stage.GetSessionLayer()
    spec = layer.GetPrimAtPath("/World")
    if spec:
        del layer.pseudoRoot.nameChildren["World"]


def apply_lod(usdviewApi, quiet=False):
    stage = usdviewApi.stage
    cam = usdviewApi.currentGfCamera
    if stage is None or cam is None:
        return None
    w, h = usdviewApi.viewportSize
    if w and h:
        cam.verticalAperture = cam.horizontalAperture * h / float(w)
    fr = cam.frustum
    with Sdf.ChangeBlock():
        _clear_session(stage)
    if S.stream:
        if S.streamer is None or S.streamer.stage is not stage:
            S.streamer = ls.Streamer(stage)
        loads, unloads, _ = S.streamer.update(fr.GetPosition())
        S.streamer.apply(loads, unloads)
        sel = ls.Selector(stage).select(fr, load=False, assets=S.assets)
    else:
        sel = ls.Selector(stage).select(fr, load=True, assets=S.assets)
    ls.apply_to_session(stage, sel)
    S.last = sel
    S.last_cam = cam.transform
    msg = _summary(sel)
    usdviewApi.PrintStatus(msg)
    if not quiet:
        print("[UsdLod]", msg)
    usdviewApi.UpdateGUI()
    return sel


def _summary(sel):
    chunks = [c for c in sel.cells.values() if c.levels == ["fine", "base"]]
    cells = [(p, c) for p, c in sel.cells.items() if c.levels != ["fine", "base"]]
    full = sorted(p.split("/")[-1] for p, c in cells if c.index == 0)
    coarse = sum(1 for _, c in cells if c.index > 0)
    s = f"{len(cells)} cell roots: full {full}, {coarse} coarse; ground chunks fine {sum(c.index == 0 for c in chunks)}" \
        f"/{len(chunks)}"
    if sel.assets:
        s += f"; asset levels {sel.asset_histogram()}"
    if sel.late:
        s += f"; {len(sel.late)} drawn coarser (not resident)"
    return s


def report(usdviewApi):
    sel = S.last or apply_lod(usdviewApi, quiet=True)
    if sel is None:
        return
    print("[UsdLod] selection from the current camera:")
    for p, c in sorted(sel.cells.items()):
        print(f"  {p:55s} -> {c.levels[c.index]:10s} ({c.heuristic} {c.metric:.1f}, lod {c.value:.2f})")
    if sel.assets:
        print(f"  assets evaluated: {len(sel.assets)}, by level {sel.asset_histogram()}")
    print("  loaded payloads:", sorted(str(p) for p in usdviewApi.stage.GetLoadSet()))
    usdviewApi.PrintStatus(_summary(sel))


def reset(usdviewApi):
    _clear_session(usdviewApi.stage)
    S.last = None
    S.streamer = None
    usdviewApi.PrintStatus("UsdLod: session opinions cleared (lod:default:index applies)")
    usdviewApi.UpdateGUI()


def _tick(usdviewApi):
    cam = usdviewApi.currentGfCamera
    if cam is not None and (S.last_cam is None or cam.transform != S.last_cam):
        apply_lod(usdviewApi, quiet=True)


def toggle_auto(usdviewApi):
    S.auto = not S.auto
    if S.auto:
        if S.timer is None:
            S.timer = QtCore.QTimer()
            S.timer.timeout.connect(lambda: _tick(usdviewApi))
        S.timer.start(600)
    elif S.timer is not None:
        S.timer.stop()
    usdviewApi.PrintStatus(f"UsdLod: auto-update {'on' if S.auto else 'off'}")


def toggle_stream(usdviewApi):
    S.stream = not S.stream
    S.streamer = None
    usdviewApi.PrintStatus(f"UsdLod: streaming domain {'on' if S.stream else 'off'}")


def toggle_assets(usdviewApi):
    S.assets = not S.assets
    usdviewApi.PrintStatus(f"UsdLod: asset LOD {'on' if S.assets else 'off'}")


class LodPluginContainer(PluginContainer):
    def registerPlugins(self, plugRegistry, usdviewApi):
        self._apply = plugRegistry.registerCommandPlugin(
            "UsdLod.apply", "Apply LOD from view", apply_lod, "Select LOD levels from the current camera")
        self._auto = plugRegistry.registerCommandPlugin(
            "UsdLod.auto", "Auto-update LOD (toggle)", toggle_auto, "Re-select whenever the camera moves")
        self._stream = plugRegistry.registerCommandPlugin(
            "UsdLod.stream", "Stream: streaming domain (toggle)", toggle_stream,
            "Load and unload payloads by the streaming heuristics")
        self._assets = plugRegistry.registerCommandPlugin(
            "UsdLod.assets", "Include asset LOD (toggle)", toggle_assets, "Switch asset LOD inside tiles")
        self._report = plugRegistry.registerCommandPlugin(
            "UsdLod.report", "Report LOD selection", report, "Print the selection")
        self._reset = plugRegistry.registerCommandPlugin(
            "UsdLod.reset", "Reset to defaults", reset, "Remove the plugin's session opinions")

    def configureView(self, plugRegistry, plugUIBuilder):
        m = plugUIBuilder.findOrCreateMenu("UsdLod")
        m.addItem(self._apply, "Ctrl+L")
        m.addItem(self._auto)
        m.addItem(self._stream)
        m.addItem(self._assets)
        m.addSeparator()
        m.addItem(self._report)
        m.addItem(self._reset)


Tf.Type.Define(LodPluginContainer)
