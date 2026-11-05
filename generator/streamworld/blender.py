"""Mesh simplification and convex hulls through Blender (run headless, batched, results cached on disk).

Blender is a generation-time tool only; nothing written to world/ depends on it.
"""

import hashlib
import json
import os
import subprocess
import tempfile

import numpy as np

from .meshutil import Mesh

BLENDER = os.environ.get("BLENDER", "blender")
_HERE = os.path.dirname(os.path.abspath(__file__))


class Job:
    def __init__(self, op, mesh=None, points=None, target_tris=None, planar_angle=None):
        self.op = op
        self.mesh = mesh
        self.points = points
        self.target_tris = target_tris
        self.planar_angle = planar_angle
        self.result = None

    def key(self):
        h = hashlib.sha1()
        h.update(f"{self.op}:{self.target_tris}:{self.planar_angle}:v2".encode())
        if self.mesh is not None:
            for a in (self.mesh.points, self.mesh.counts, self.mesh.indices, self.mesh.st):
                if a is not None:
                    h.update(np.ascontiguousarray(a).tobytes())
        if self.points is not None:
            h.update(np.ascontiguousarray(self.points).tobytes())
        return h.hexdigest()[:20]


def _from_corners(d, weld_tol=1e-6):
    """Rebuild a vertex-indexed mesh from per-corner data, splitting vertices only where st or normals differ."""
    cp = d["corner_points"].astype(np.float64)
    cn = d["corner_normals"].astype(np.float64)
    counts = d["counts"]
    cst = d["corner_st"].astype(np.float64) if "corner_st" in d.files else None
    vid = d["corner_vertex"]
    keyparts = [vid[:, None].astype(np.float64), np.round(cn, 3)]
    if cst is not None:
        keyparts.append(np.round(cst, 5))
    key = np.concatenate(keyparts, axis=1)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(uniq)
    pts = np.zeros((n, 3))
    nrm = np.zeros((n, 3))
    pts[inv] = cp
    nrm[inv] = cn
    st = None
    if cst is not None:
        st = np.zeros((n, 2))
        st[inv] = cst
    return Mesh(pts, counts, inv.astype(np.int32), nrm, st)


def run(jobs, cache_dir):
    """Execute jobs (in place: sets job.result) using the cache where possible."""
    cache_dir = os.path.join(cache_dir, "blender")
    os.makedirs(cache_dir, exist_ok=True)
    todo = []
    for j in jobs:
        out = os.path.join(cache_dir, j.key() + ".npz")
        if os.path.exists(out):
            j.result = _from_corners(np.load(out))
        else:
            todo.append((j, out))
    if not todo:
        return jobs
    with tempfile.TemporaryDirectory(dir=cache_dir) as tmp:
        spec = []
        for i, (j, out) in enumerate(todo):
            inp = os.path.join(tmp, f"in{i}.npz")
            if j.op == "hull":
                np.savez(inp, points=np.asarray(j.points, dtype=np.float64))
            else:
                arrs = dict(points=j.mesh.points, counts=j.mesh.counts, indices=j.mesh.indices)
                if j.mesh.st is not None:
                    arrs["st"] = j.mesh.st
                np.savez(inp, **arrs)
            spec.append(dict(op=j.op, **{"in": inp}, out=out, target_tris=j.target_tris, planar_angle=j.planar_angle))
        jp = os.path.join(tmp, "jobs.json")
        json.dump(spec, open(jp, "w"))
        r = subprocess.run([BLENDER, "-b", "--factory-startup", "--python", os.path.join(_HERE, "blender_ops.py"),
                            "--", jp], capture_output=True, text=True)
        missing = [out for _, out in todo if not os.path.exists(out)]
        if missing:
            raise RuntimeError(f"blender failed ({len(missing)} jobs):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    for j, out in todo:
        j.result = _from_corners(np.load(out))
    return jobs


def decimate(mesh, target_tris, cache_dir):
    return run([Job("decimate", mesh=mesh, target_tris=target_tris)], cache_dir)[0].result


def hull(points, target_tris, cache_dir):
    return run([Job("hull", points=points, target_tris=target_tris)], cache_dir)[0].result
