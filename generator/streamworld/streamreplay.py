"""Streaming replay: fly the manifest's flight path through the cell hierarchy and simulate payload streaming.

No rendering and almost no USD loading: the stage is opened unloaded (cell heuristics live outside the payloads),
and only a `full` level that becomes resident is composed (its nested roots, the ground chunks of a tile, are
inside that payload, as they would be for an engine); the
residency policy is lodselect.Streamer, and loads are simulated as asynchronous requests on one I/O channel
(latency per request + size / bandwidth, sizes are the level files on disk). Every step the imaging heuristics pick
the level to draw; a level that is not resident yet falls back to the nearest coarser resident level (a "late"
draw). Two policies are compared:

    draw-only   residency = the drawn level only (imaging thresholds, no hysteresis): load what is drawn
    streaming   residency = streaming domain (load radii, unload hysteresis), coarser levels kept as fallback

Writes world/stream_replay.json and prints a summary.
"""

import json
import os
import time

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdLod

from . import lodselect as ls
from .cells import cell_level_path, root_path


class _Sim:
    def __init__(self, cfg, stage, policy, bandwidth, latency):
        self.cfg, self.stage = cfg, stage
        self.streamer = ls.Streamer(stage, domain=ls.STREAMING, keep_coarser=True) if policy == "streaming" else \
            ls.Streamer(stage, domain=UsdLod.Tokens.imaging, keep_coarser=False)
        self.selector = ls.Selector(stage, cell_heuristic="distance")
        self.bw, self.lat = bandwidth * 1e6, latency
        self.resident = {}        # level path -> time it became resident
        self.pending = {}         # level path -> completion time
        self.channel_free = 0.0
        self.sizes = {}
        self.ev = dict(loads=0, unloads=0, bytes=0, reloads=0, late_events=0, late_root_seconds=0.0,
                       cancelled=0)
        self.loaded_once = set()
        self.late_now = set()
        self.leads = []
        self.drawn = {}
        self.peak_resident = 0
        self.peak_queue_s = 0.0
        self.log = []

    def size(self, level_path):
        if level_path not in self.sizes:
            prim = self.stage.GetPrimAtPath(level_path)
            nbytes = prim.GetCustomDataByKey("bytes")
            if nbytes is not None:
                # A ground chunk's fine level: one prim of the tile's ground file (its geometry bytes).
                self.sizes[level_path] = int(nbytes)
            else:
                cell = self.cfg.cell_by_id(prim.GetParent().GetCustomDataByKey("cell"))
                self.sizes[level_path] = os.path.getsize(cell_level_path(self.cfg, cell, prim.GetName()))
        return self.sizes[level_path]

    def request(self, t, p):
        start = max(t, self.channel_free)
        done = start + self.lat + self.size(p) / self.bw
        self.channel_free = done
        self.pending[p] = done
        self.peak_queue_s = max(self.peak_queue_s, done - t)
        self.ev["loads"] += 1
        self.ev["bytes"] += self.size(p)
        if p in self.loaded_once:
            self.ev["reloads"] += 1
        self.loaded_once.add(p)
        self.log.append((round(t, 2), "LOAD", p, self.size(p)))

    def _compose(self, p, load):
        """Compose (or release) a resident `full` level, so the roots nested in its payload become visible."""
        if not p.endswith("/full"):
            return
        if load:
            self.stage.Load(p, Usd.LoadWithoutDescendants)
        else:
            self.stage.Unload(p)
        self._xc.Clear()

    def step(self, t, eye, warm=False):
        loads, unloads, _ = self.streamer.update(eye)
        for p in unloads:
            if p in self.pending:
                del self.pending[p]
                self.ev["cancelled"] += 1
            if self.resident.pop(p, None) is not None:
                self._compose(p, False)
            self.ev["unloads"] += 1
            self.log.append((round(t, 2), "UNLOAD", p, self.size(p)))
        for p in loads:
            if warm:
                self.resident[p] = t
                self.loaded_once.add(p)
                self._compose(p, True)
            else:
                self.request(t, p)
        for p, done in list(self.pending.items()):
            if done <= t:
                self.resident[p] = done
                del self.pending[p]
                self._compose(p, True)
        self.peak_resident = max(self.peak_resident, sum(self.size(p) for p in self.resident))
        self.draw(t, eye)

    def draw(self, t, eye, dt=0.0):
        xc = self._xc
        eye3 = Gf.Vec3d(*eye)
        top = self.stage.GetPrimAtPath("/World/Regions")
        queue = [p for p in top.GetChildren() if p.HasAPI(UsdLod.RootAPI)]
        late = set()
        while queue:
            root = queue.pop(0)
            idx, *_ = self.selector.evaluate(root, xc, eye3, None, "distance")
            levels = ls.lod_levels(root)
            res = [not lv.HasAuthoredPayloads() or str(lv.GetPath()) in self.resident for lv in levels]
            d = idx
            if not res[idx]:
                late.add(str(root.GetPath()))
                coarser = [i for i in range(idx + 1, len(levels)) if res[i]]
                d = coarser[0] if coarser else None
            key = str(root.GetPath())
            prev = self.drawn.get(key)
            if d is not None and prev is not None and d < prev and levels[d].HasAuthoredPayloads():
                self.leads.append(t - self.resident[str(levels[d].GetPath())])
            self.drawn[key] = d
            if d == 0:
                queue += [c for c in levels[0].GetChildren() if c.HasAPI(UsdLod.RootAPI)]
        self.ev["late_events"] += len(late - self.late_now)
        self.late_now = late
        return late


def _path_points(man, speed, dt):
    eyes = np.array([f["eye"] for f in man["flight_path"]], float)
    seg = np.linalg.norm(np.diff(eyes, axis=0), axis=1)
    total = float(seg.sum())
    cum = np.r_[0, np.cumsum(seg)]
    s = np.arange(0, total, speed * dt)
    return np.stack([np.interp(s, cum, eyes[:, k]) for k in range(3)], 1), total


def run_replay(cfg, speed=30.0, bandwidth=50.0, latency=0.05, dt=0.1, policies=("draw-only", "streaming")):
    man = json.load(open(os.path.join(cfg.out, "manifest.json")))
    pts, total = _path_points(man, speed, dt)
    out = dict(flight_m=round(total, 1), speed_mps=speed, bandwidth_MBps=bandwidth, latency_s=latency, dt_s=dt,
               steps=len(pts), policies={})
    for pol in policies:
        t0 = time.time()
        st = Usd.Stage.Open(root_path(cfg), Usd.Stage.LoadNone)
        sim = _Sim(cfg, st, pol, bandwidth, latency)
        sim._xc = UsdGeom.XformCache()
        late_seconds = 0.0
        mbps = []
        last_bytes = 0
        for i, eye in enumerate(pts):
            t = i * dt
            sim.step(t, eye, warm=(i == 0))
            # Warm start: repeat until residency settles, since nested roots (ground chunks) only appear once their
            # tile's full level is composed.
            for _ in range(4 if i == 0 else 0):
                sim.step(t, eye, warm=True)
            late_seconds += len(sim.late_now) * dt
            if i == 0:
                warm_bytes = sum(sim.size(p) for p in sim.resident)
                sim.ev["loads"] = sim.ev["bytes"] = 0
                last_bytes = 0
            if i % int(round(1 / dt)) == 0:
                mbps.append((sim.ev["bytes"] - last_bytes) / 1e6)
                last_bytes = sim.ev["bytes"]
        leads = np.array(sim.leads) if sim.leads else np.array([0.0])
        r = dict(loads=sim.ev["loads"], unloads=sim.ev["unloads"], reloads=sim.ev["reloads"],
                 cancelled=sim.ev["cancelled"], loaded_MB=round(sim.ev["bytes"] / 1e6, 1),
                 warm_start_MB=round(warm_bytes / 1e6, 1), peak_resident_MB=round(sim.peak_resident / 1e6, 1),
                 peak_requested_MB_per_s=round(max(mbps), 1), peak_queue_s=round(sim.peak_queue_s, 2),
                 late_events=sim.ev["late_events"], late_root_seconds=round(late_seconds, 1),
                 lead_time_s=dict(min=round(float(leads.min()), 2), median=round(float(np.median(leads)), 2)),
                 seconds=round(time.time() - t0, 1))
        out["policies"][pol] = r
        out.setdefault("log_head", {})[pol] = sim.log[:40]
    path = os.path.join(cfg.out, "stream_replay.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[stream-replay] {total:.0f} m at {speed} m/s, {bandwidth} MB/s, {latency * 1000:.0f} ms/request "
          f"({len(pts)} steps)")
    keys = ["loads", "unloads", "reloads", "loaded_MB", "warm_start_MB", "peak_resident_MB",
            "peak_requested_MB_per_s", "peak_queue_s", "late_events", "late_root_seconds"]
    print(f"  {'':24s}" + "".join(f"{p:>14s}" for p in out["policies"]))
    for k in keys + ["lead_time_s"]:
        vals = [out["policies"][p][k] for p in out["policies"]]
        fmt = [f"{v['min']}/{v['median']}" if isinstance(v, dict) else str(v) for v in vals]
        print(f"  {k:24s}" + "".join(f"{v:>14s}" for v in fmt))
    print(f"  -> {path}")
    return out
