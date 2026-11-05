"""Top-down overview map of the world (hillshade + biome tint + cell grid), for documentation and review."""

import numpy as np

from .imgutil import write_image


def render_map(cfg, world, path, px=1000):
    k = max(1, int(round(world.n / px)))
    H = np.asarray(world.H[::k, ::k], dtype=np.float64)[::-1]  # north up
    sp = world.spacing * k
    gy, gx = np.gradient(H, sp)
    gy = -gy  # rows were flipped
    n = np.stack([-gx, -gy, np.ones_like(H)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    sun = np.array([-0.5, 0.6, 0.62])
    sun /= np.linalg.norm(sun)
    shade = np.clip(n @ sun, 0, 1)
    ny, nx = H.shape
    xs = world.x0 + np.arange(nx) * sp
    ys = world.x0 + np.arange(ny)[::-1] * sp
    X, Y = np.meshgrid(xs, ys)
    f = {name: world.field(name, X, Y) for name in ("forest", "rock", "snow", "conifer", "meadow", "path", "wet",
                                                    "farm")}
    col = np.zeros(H.shape + (3,))
    grass = np.array([0.45, 0.58, 0.25])
    col[:] = grass
    col = col * (1 - f["meadow"][..., None] * 0.3) + np.array([0.6, 0.62, 0.3]) * f["meadow"][..., None] * 0.3
    farm = f["farm"][..., None]
    strips = 0.5 + 0.5 * np.sin(X * 0.21 + Y * 0.07)
    fcol = np.array([0.72, 0.62, 0.32]) * (0.85 + 0.25 * strips[..., None])
    col = col * (1 - farm) + fcol * farm
    forest = f["forest"][..., None]
    fcol = np.array([0.18, 0.36, 0.14]) * (1 - f["conifer"][..., None]) + np.array([0.1, 0.25, 0.15]) * f["conifer"][..., None]
    col = col * (1 - forest) + fcol * forest
    rock = f["rock"][..., None]
    col = col * (1 - rock) + np.array([0.5, 0.48, 0.45]) * rock
    snow = f["snow"][..., None]
    col = col * (1 - snow) + np.array([0.95, 0.95, 0.97]) * snow
    on_path = (f["path"] < 0.6 * sp)[..., None]   # at least a pixel wide
    col = np.where(on_path, np.array([0.55, 0.42, 0.28]), col)
    img = col * (0.35 + 0.75 * shade[..., None])
    # Cell grid lines.
    for lv, alpha in zip(cfg.levels, (0.9, 0.35)):
        step = int(round(lv.size / sp))
        for i in range(0, nx, step):
            img[:, i] = img[:, i] * (1 - alpha) + alpha
        for j in range(0, ny, step):
            img[j, :] = img[j, :] * (1 - alpha) + alpha
    # Points of interest: settlements brown, ruins and landmarks red, the uncanny violet; path markers as dots.
    kinds = {"village": (0.45, 0.2, 0.05), "hamlet": (0.45, 0.2, 0.05), "castle": (0.45, 0.2, 0.05),
             "windmill": (0.45, 0.2, 0.05), "graveyard": (0.45, 0.2, 0.05), "wizard_tower": (0.55, 0.2, 0.9),
             "dragon_bones": (0.55, 0.2, 0.9), "crystal_hollow": (0.55, 0.2, 0.9), "runestones": (0.55, 0.2, 0.9)}
    for p in cfg.c.pois:
        cx = int((p["position"][0] - world.x0) / sp)
        cy = int((world.x0 + world.size - p["position"][1]) / sp)
        r = max(3, int(float(p["radius"]) / sp * 0.6))
        img[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1] = kinds.get(p["kind"], (0.9, 0.1, 0.1))
    from .pois import path_markers
    for mx, my, _, _ in path_markers(cfg):
        cx = int((mx - world.x0) / sp)
        cy = int((world.x0 + world.size - my) / sp)
        img[max(cy - 1, 0):cy + 2, max(cx - 1, 0):cx + 2] = (0.95, 0.9, 0.6)
    write_image(path, np.clip(img, 0, 1))
