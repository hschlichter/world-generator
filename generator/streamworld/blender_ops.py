"""Runs inside Blender (blender -b --python blender_ops.py -- jobs.json): mesh decimation and convex hulls.

Each job reads <in>.npz (points, counts, indices[, st]) and writes <out>.npz with per-corner data:
corner_points, corner_normals, corner_st (if st given), counts.
"""

import json
import sys

import bmesh
import bpy
import numpy as np


def build(points, counts, indices, st=None):
    me = bpy.data.meshes.new("m")
    nv = len(points)
    nf = len(counts)
    me.vertices.add(nv)
    me.vertices.foreach_set("co", points.astype(np.float32).ravel())
    me.loops.add(len(indices))
    me.loops.foreach_set("vertex_index", indices.astype(np.int32))
    me.polygons.add(nf)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int32)
    me.polygons.foreach_set("loop_start", starts)
    if st is not None:
        uv = me.uv_layers.new(name="st")
        uv.data.foreach_set("uv", st[indices].astype(np.float32).ravel())
    me.update(calc_edges=True)
    me.validate()
    ob = bpy.data.objects.new("o", me)
    bpy.context.scene.collection.objects.link(ob)
    return ob


def export(ob, path, has_st):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = ob.evaluated_get(dg)
    me = ev.to_mesh()
    for p in me.polygons:
        p.use_smooth = True
    nl = len(me.loops)
    vi = np.empty(nl, dtype=np.int32)
    me.loops.foreach_get("vertex_index", vi)
    co = np.empty(len(me.vertices) * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    counts = np.empty(len(me.polygons), dtype=np.int32)
    me.polygons.foreach_get("loop_total", counts)
    cn = np.empty(nl * 3, dtype=np.float32)
    me.corner_normals.foreach_get("vector", cn) if hasattr(me, "corner_normals") else None
    out = dict(corner_points=co[vi], corner_vertex=vi, corner_normals=cn.reshape(-1, 3), counts=counts)
    if has_st and me.uv_layers:
        uv = np.empty(nl * 2, dtype=np.float32)
        me.uv_layers[0].data.foreach_get("uv", uv)
        out["corner_st"] = uv.reshape(-1, 2)
    np.savez(path, **out)
    ev.to_mesh_clear()


def main():
    jobs = json.load(open(sys.argv[sys.argv.index("--") + 1]))
    for job in jobs:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        d = np.load(job["in"])
        st = d["st"] if "st" in d.files else None
        if job["op"] == "hull":
            me = bpy.data.meshes.new("h")
            bm = bmesh.new()
            for p in d["points"]:
                bm.verts.new(p)
            bmesh.ops.convex_hull(bm, input=bm.verts[:])
            bm.to_mesh(me)
            bm.free()
            ob = bpy.data.objects.new("o", me)
            bpy.context.scene.collection.objects.link(ob)
            st = None
        else:
            ob = build(d["points"], d["counts"], d["indices"], st)
        target = job.get("target_tris")
        if target:
            ntri = sum(len(p.vertices) - 2 for p in ob.data.polygons)
            if ntri > target:
                m = ob.modifiers.new("dec", "DECIMATE")
                m.decimate_type = "COLLAPSE"
                m.ratio = max(target / ntri, 0.001)
                m.use_collapse_triangulate = True
        if job.get("planar_angle"):
            m = ob.modifiers.new("pl", "DECIMATE")
            m.decimate_type = "DISSOLVE"
            m.angle_limit = job["planar_angle"]
        export(ob, job["out"], st is not None)


main()
