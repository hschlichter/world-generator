"""Lights (sun + sky dome) and test viewpoints: named cameras and an ordered flight path."""

import os

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdLux

from . import atlas, textures
from . import usdutil as uu


def look_at(eye, target):
    """Camera-to-world matrix (USD row-vector convention) for a camera looking from eye to target, Z up."""
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, f)
    m = np.eye(4)
    m[0, :3] = right
    m[1, :3] = up
    m[2, :3] = -f
    m[3, :3] = eye
    return Gf.Matrix4d(*m.ravel().tolist())


def _search(world, cfg, regions, score_fn, step=10.0, margin=30.0):
    """Best-scoring point on a grid over the given regions (or the world)."""
    boxes = [c.bounds2d for c in regions] if regions else [(-cfg.world_size / 2, -cfg.world_size / 2,
                                                              cfg.world_size / 2, cfg.world_size / 2)]
    best, bp = -np.inf, None
    for x0, y0, x1, y1 in boxes:
        xs = np.arange(x0 + margin, x1 - margin, step)
        ys = np.arange(y0 + margin, y1 - margin, step)
        X, Y = np.meshgrid(xs, ys)
        s = score_fn(X, Y)
        i = np.unravel_index(np.argmax(s), s.shape)
        if s[i] > best:
            best, bp = s[i], (float(X[i]), float(Y[i]))
    return bp


def choose_viewpoints(cfg, world, regions):
    """Named viewpoints. Searches inside the generated regions where possible, so they show generated content."""
    f = world.field
    h = world.height
    eye = 1.7
    vps = {}
    poi = {p["name"]: p for p in cfg.c.pois}
    lv0 = cfg.levels[0]
    lv1 = cfg.levels[-1]

    def inside(x, y):
        return any(c.x0 <= x < c.x0 + c.level.size and c.y0 <= y < c.y0 + c.level.size for c in regions)

    # Overlook: the highest summit in the world, looking at the world centre (or the generated area).
    k = 8
    Hc = np.asarray(world.H[::k, ::k])
    j, i = np.unravel_index(np.argmax(Hc), Hc.shape)
    px, py = world.x0 + i * k * world.spacing, world.x0 + j * k * world.spacing
    tgt = np.mean([c.center for c in regions], axis=0) if regions else (0.0, 0.0)
    vps["overlook"] = dict(eye=(px, py, h(px, py) + 60.0), target=(tgt[0], tgt[1], h(*tgt)),
                           note="highest summit, looking across the map")
    # Dense forest: deepest forest, eye height, looking along the slope.
    x, y = _search(world, cfg, regions, lambda X, Y: f("forest", X, Y) * (1 - f("slope", X, Y) / 60) -
                   0.02 * np.maximum(0, 30 - f("path", X, Y)))
    vps["dense_forest"] = dict(eye=(x, y, h(x, y) + eye), target=(x + 40, y + 25, h(x + 40, y + 25) + 3),
                               note="inside dense forest at eye height")
    # Valley: on the valley floor, looking up the valley.
    vpts = np.asarray(cfg.c.terrain.valleys[0]["points"])
    seg = None
    for a, b in zip(vpts[:-1], vpts[1:]):
        for t in np.linspace(0, 1, 21):
            p = a + (b - a) * t
            if inside(*p) or not regions:
                seg = (p, b - a)
                break
        if seg is not None:
            break
    if seg is None:
        seg = (vpts[len(vpts) // 2], vpts[len(vpts) // 2 + 1] - vpts[len(vpts) // 2])
    p, d = seg
    d = d / np.linalg.norm(d)
    vps["valley"] = dict(eye=(p[0], p[1], h(*p) + 38.0), target=(p[0] + d[0] * 300, p[1] + d[1] * 300,
                                                                 h(p[0] + d[0] * 300, p[1] + d[1] * 300) + 30),
                         note="valley floor looking up the valley")
    # Open grassland.
    x, y = _search(world, cfg, regions, lambda X, Y: (1 - f("forest", X, Y)) * (1 - f("rock", X, Y)) *
                   (1 - f("slope", X, Y) / 30) + 0.3 * f("meadow", X, Y))
    vps["grassland"] = dict(eye=(x, y, h(x, y) + eye), target=(x - 200, y - 120, h(x - 200, y - 120) + 5),
                            note="open grassland at eye height, long view")
    # Mountain ridge: high, steep surroundings.
    x, y = _search(world, cfg, regions, lambda X, Y: h(X, Y) + 2.0 * f("slope", X, Y))
    vps["mountain_ridge"] = dict(eye=(x, y, h(x, y) + 4.0), target=(x - 300, y - 600, h(x - 300, y - 600)),
                                 note="highest ground of the generated area, looking out over it")
    # Four tiles meet: a tile corner that is not a region corner.
    reg = regions[0] if regions else None
    if reg is not None:
        tx, ty = reg.x0 + lv1.size * 2, reg.y0 + lv1.size * 2
    else:
        tx, ty = lv1.size, lv1.size
    vps["four_tiles"] = dict(eye=(tx - 6, ty - 6, h(tx, ty) + eye), target=(tx + 30, ty + 30, h(tx + 30, ty + 30) + 1),
                             note="point where four tiles meet")
    # Four regions meet: the region corner closest to the generated area (the world origin by default).
    corners = [(cx, cy) for cx in np.arange(-cfg.world_size / 2 + lv0.size, cfg.world_size / 2, lv0.size)
               for cy in np.arange(-cfg.world_size / 2 + lv0.size, cfg.world_size / 2, lv0.size)]
    c = min(corners, key=lambda q: np.hypot(q[0] - tgt[0], q[1] - tgt[1]))
    vps["four_regions"] = dict(eye=(c[0] - 8, c[1] - 8, h(*c) + 3.0), target=(c[0] + 60, c[1] + 60, h(c[0] + 60, c[1] + 60)),
                               note="point where four regions meet")
    # Landmark: the first point of interest inside the generated area.
    lm = next((p for p in cfg.c.pois if inside(*p["position"])), cfg.c.pois[0])
    lx, ly = lm["position"]
    r = float(lm["radius"])
    vps["landmark"] = dict(eye=(lx - r * 1.6, ly - r * 1.9, h(lx, ly) + r * 0.6), target=(lx, ly, h(lx, ly) + 4),
                           note=f"landmark: {lm['name']}")
    # Showcases: the first village, the castle and the wizard's spire inside the generated area, each seen from a
    # walker's height on its approach (from the south-west, a little above the ground).
    for name, kind, dist, up, look_up in (("village", "village", 1.25, 6.0, 2.0), ("castle", "castle", 1.7, 24.0, 6.0),
                                           ("spire", "wizard_tower", 2.4, 3.0, 18.0)):
        p = next((q for q in cfg.c.pois if q["kind"] == kind and inside(*q["position"])), None)
        if p is None:
            continue
        px, py = p["position"]
        d = float(p["radius"]) * dist
        ex, ey = px - d * 0.6, py - d * 0.8
        vps[name] = dict(eye=(ex, ey, max(h(ex, ey), h(px, py) - 20) + up), target=(px, py, h(px, py) + look_up),
                         note=f"{kind.replace('_', ' ')}: {p['name']}")
    return vps


def flight_path(cfg, world, spacing=300.0, fast_spacing=900.0):
    """Ordered flight cameras along the configured waypoints ([flight] waypoints = [[x, y, height above ground,
    fast], ...]): one camera every `spacing` metres on slow legs, every `fast_spacing` on fast ones."""
    wps = [(float(x), float(y), float(a), bool(f)) for x, y, a, f in cfg.c.flight.waypoints]
    wp = np.array([[x, y] for x, y, _, _ in wps])
    agl = np.array([a for _, _, a, _ in wps])
    fast = [f for *_, f in wps]
    out = []
    for k in range(len(wp) - 1):
        a, b = wp[k], wp[k + 1]
        L = np.linalg.norm(b - a)
        sp = fast_spacing if fast[k + 1] else spacing
        n = max(1, int(round(L / sp)))
        for i in range(n):
            t = i / n
            p = a + (b - a) * t
            al = agl[k] + (agl[k + 1] - agl[k]) * t
            out.append((p, al, fast[k + 1]))
    out.append((wp[-1], agl[-1], True))
    cams = []
    for i, (p, al, f) in enumerate(out):
        nxt = out[min(i + 1, len(out) - 1)][0] if i + 1 < len(out) else p + (p - out[i - 1][0])
        z = world.height(*p) + al
        tz = world.height(*nxt) + out[min(i + 1, len(out) - 1)][1] - 10
        cams.append(dict(eye=(float(p[0]), float(p[1]), float(z)), target=(float(nxt[0]), float(nxt[1]), float(tz)),
                         fast=bool(f)))
    return cams


def _camera(stage, path, eye, target, note=None):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(24.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(20.25)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.3, 12000.0))
    cam.AddTransformOp().Set(look_at(eye, target))
    if note:
        cam.GetPrim().SetDocumentation(note)
    return cam


def write_viewpoints(cfg, world, regions):
    path = os.path.join(cfg.out, "layers", "viewpoints.usda")
    st = uu.new_stage(path, "cameras", doc="Named test cameras and the ordered flight path (flight_00, flight_01, ...).")
    st.DefinePrim("/cameras", "Scope")
    vps = choose_viewpoints(cfg, world, regions)
    for name, v in vps.items():
        _camera(st, f"/cameras/{name}", v["eye"], v["target"], v["note"])
    fl = flight_path(cfg, world)
    for i, c in enumerate(fl):
        cam = _camera(st, f"/cameras/flight_{i:02d}", c["eye"], c["target"])
        cam.GetPrim().SetCustomDataByKey("fast", c["fast"])
    st.GetRootLayer().Save()
    return path, vps, fl


def write_environment(cfg):
    path = os.path.join(cfg.out, "layers", "environment.usda")
    st = uu.new_stage(path, "Environment", doc="Lighting: a DistantLight sun and a DomeLight HDR sky (the same sun).")
    env = UsdGeom.Xform.Define(st, "/Environment")
    lc = cfg.c.lighting
    sun = UsdLux.DistantLight.Define(st, "/Environment/Sun")
    d = np.asarray(lc.sun_direction, float)
    d /= np.linalg.norm(d)
    # The light shines along its local -Z: aim -Z away from the sun.
    sun.AddTransformOp().Set(look_at((0, 0, 0), -d))
    sun.CreateIntensityAttr(float(lc.sun_intensity))
    sun.CreateColorAttr(Gf.Vec3f(*lc.sun_color))
    sun.CreateAngleAttr(float(lc.sun_angle))
    dome = UsdLux.DomeLight_1.Define(st, "/Environment/Sky")
    dome.CreateTextureFileAttr(Sdf.AssetPath(f"../{atlas.TEXTURE_DIR}/{atlas.SKY_TEXTURE}"))
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreatePoleAxisAttr("Z")
    dome.CreateIntensityAttr(float(lc.sky_intensity))
    # The HDR sky is built for this sun (see textures.sky): its disc sits at the DistantLight's direction and
    # carries only a small fraction of its energy, so the dome can light the scene without counting the sun twice.
    dome.GetPrim().SetCustomDataByKey("sky", {"model": "Preetham", "turbidity": textures.SKY_TURBIDITY,
                                              "sunDirection": Gf.Vec3d(*map(float, d)),
                                              "sunDiscEnergyFraction": textures.SUN_DISC_ENERGY})
    st.GetRootLayer().Save()
    return path
