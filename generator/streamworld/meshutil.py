"""Polygon meshes as numpy arrays, and procedural primitives."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    points: np.ndarray                 # (N, 3)
    counts: np.ndarray                 # (F,) vertices per face
    indices: np.ndarray                # (sum(counts),)
    normals: np.ndarray | None = None  # (N, 3) per vertex
    st: np.ndarray | None = None       # (N, 2) per vertex
    color: np.ndarray | None = None    # (N, 3) per vertex displayColor
    double_sided: bool = False

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 3)
        self.counts = np.asarray(self.counts, dtype=np.int32)
        self.indices = np.asarray(self.indices, dtype=np.int32)

    @property
    def n_points(self):
        return len(self.points)

    @property
    def n_tris(self):
        return int(np.sum(self.counts - 2))

    def extent(self):
        return self.points.min(axis=0), self.points.max(axis=0)

    def copy(self):
        return Mesh(self.points.copy(), self.counts.copy(), self.indices.copy(),
                    None if self.normals is None else self.normals.copy(),
                    None if self.st is None else self.st.copy(),
                    None if self.color is None else self.color.copy(), self.double_sided)

    def transformed(self, m4):
        """Apply a 4x4 row-vector matrix (p' = p @ M)."""
        out = self.copy()
        p = np.c_[self.points, np.ones(len(self.points))] @ m4
        out.points = p[:, :3]
        if self.normals is not None:
            nm = np.linalg.inv(m4[:3, :3]).T
            n = self.normals @ nm
            out.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        return out

    def triangles(self):
        """Fan-triangulated (T, 3) index array."""
        tris = []
        starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        for c in np.unique(self.counts):
            sel = starts[self.counts == c]
            for k in range(1, c - 1):
                tris.append(np.stack([self.indices[sel], self.indices[sel + k], self.indices[sel + k + 1]], axis=1))
        return np.concatenate(tris, axis=0) if tris else np.zeros((0, 3), dtype=np.int32)

    def compute_normals(self):
        t = self.triangles()
        p = self.points
        fn = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
        n = np.zeros_like(p)
        for k in range(3):
            np.add.at(n, t[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-12, n / np.maximum(ln, 1e-12), np.array([0.0, 0.0, 1.0]))
        self.normals = n
        return self


def merge(meshes):
    meshes = [m for m in meshes if m is not None and m.n_points > 0]
    if not meshes:
        return None
    pts, cnt, idx, nrm, st, col = [], [], [], [], [], []
    off = 0
    has_n = all(m.normals is not None for m in meshes)
    has_st = all(m.st is not None for m in meshes)
    has_c = any(m.color is not None for m in meshes)
    for m in meshes:
        pts.append(m.points)
        cnt.append(m.counts)
        idx.append(m.indices + off)
        off += m.n_points
        if has_n:
            nrm.append(m.normals)
        if has_st:
            st.append(m.st)
        if has_c:
            col.append(m.color if m.color is not None else np.ones((m.n_points, 3)))
    return Mesh(np.concatenate(pts), np.concatenate(cnt), np.concatenate(idx),
                np.concatenate(nrm) if has_n else None, np.concatenate(st) if has_st else None,
                np.concatenate(col) if has_c else None, any(m.double_sided for m in meshes))


# ---------------------------------------------------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------------------------------------------------

def mat_trs(t=(0, 0, 0), rz=0.0, s=1.0, rx=0.0, ry=0.0):
    """Row-vector matrix: scale, then rotate X, Y, Z (radians), then translate."""
    s = np.broadcast_to(np.asarray(s, dtype=np.float64), (3,))
    S = np.diag([s[0], s[1], s[2], 1.0])

    def R(axis, a):
        c, sn = np.cos(a), np.sin(a)
        m = np.eye(4)
        i, j = [(1, 2), (0, 2), (0, 1)][axis]
        m[i, i] = c
        m[j, j] = c
        m[i, j] = sn if axis != 1 else -sn
        m[j, i] = -sn if axis != 1 else sn
        return m
    T = np.eye(4)
    T[3, :3] = t
    return S @ R(0, rx) @ R(1, ry) @ R(2, rz) @ T


def rotation_to(v_from, v_to):
    """3x3 matrix (row-vector) rotating v_from onto v_to."""
    a = np.asarray(v_from, float) / np.linalg.norm(v_from)
    b = np.asarray(v_to, float) / np.linalg.norm(v_to)
    v = np.cross(a, b)
    c = np.dot(a, b)
    if c < -0.999999:
        axis = np.cross(a, [1, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1, 0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return (np.eye(3) + 2 * K @ K).T
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return (np.eye(3) + K + K @ K / (1 + c)).T


# ---------------------------------------------------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------------------------------------------------

def grid_mesh(heights, x0, y0, step, st_fn=None):
    """Quad grid over a (n+1, n+1) height array (rows = y)."""
    ny, nx = heights.shape
    xs = x0 + np.arange(nx) * step
    ys = y0 + np.arange(ny) * step
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel(), heights.ravel()], axis=1)
    i = np.arange((ny - 1) * (nx - 1))
    r, c = i // (nx - 1), i % (nx - 1)
    v0 = r * nx + c
    quads = np.stack([v0, v0 + 1, v0 + nx + 1, v0 + nx], axis=1).ravel()
    m = Mesh(pts, np.full(len(i), 4), quads)
    if st_fn is not None:
        m.st = st_fn(pts[:, 0], pts[:, 1])
    return m


def tube(path, radii, sides, v_per_m=1.0, cap_end=False, u_repeat=1.0, twist=0.0, flare=None):
    """Generalized cylinder along a polyline with parallel-transport frames."""
    path = np.asarray(path, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = len(path)
    tang = np.gradient(path, axis=0)
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(tang[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    nrm = np.cross(tang[0], ref)
    nrm /= np.linalg.norm(nrm)
    frames = []
    for i in range(n):
        if i > 0:
            nrm = nrm - np.dot(nrm, tang[i]) * tang[i]
            nrm /= max(np.linalg.norm(nrm), 1e-12)
        frames.append((nrm.copy(), np.cross(tang[i], nrm)))
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    seglen = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    pts, nrms, st = [], [], []
    for i in range(n):
        a, b = frames[i]
        ca = np.cos(ang + twist * i)[:, None]
        sa = np.sin(ang + twist * i)[:, None]
        dirs = ca * a + sa * b
        r = radii[i]
        if flare is not None:
            r = r * flare(i, ang)[:, None] if callable(flare) else r
        pts.append(path[i] + dirs * (r if np.ndim(r) else r))
        nrms.append(dirs)
        st.append(np.stack([np.linspace(0, u_repeat, sides + 1), np.full(sides + 1, seglen[i] * v_per_m)], 1))
    pts = np.concatenate(pts)
    nrms = np.concatenate(nrms)
    st = np.concatenate(st)
    k = sides + 1
    quads = []
    for i in range(n - 1):
        for j in range(sides):
            a0 = i * k + j
            quads.append([a0, a0 + 1, a0 + k + 1, a0 + k])
    quads = np.asarray(quads).ravel()
    counts = np.full(len(quads) // 4, 4)
    m = Mesh(pts, counts, quads, nrms, st)
    if cap_end:
        c = len(m.points)
        tip = path[-1] + tang[-1] * radii[-1] * 0.6
        m.points = np.vstack([m.points, tip])
        m.normals = np.vstack([m.normals, tang[-1]])
        m.st = np.vstack([m.st, [0.5, seglen[-1] * v_per_m + radii[-1]]])
        base = (n - 1) * k
        tri = [[base + j, c, base + j + 1] for j in range(sides)]
        m.counts = np.r_[m.counts, np.full(sides, 3)]
        m.indices = np.r_[m.indices, np.asarray(tri).ravel()]
    return m


def card(center, right, up, uv_rect=(0, 0, 1, 1), normal=None):
    """A single quad centred at `center` spanning +-right and +-up. uv_rect = (u0, v0, u1, v1)."""
    c = np.asarray(center, float)
    r = np.asarray(right, float)
    u = np.asarray(up, float)
    pts = np.array([c - r - u, c + r - u, c + r + u, c - r + u])
    u0, v0, u1, v1 = uv_rect
    st = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    n = np.cross(r, u) if normal is None else np.asarray(normal, float)
    n = n / max(np.linalg.norm(n), 1e-12)
    return Mesh(pts, [4], [0, 1, 2, 3], np.tile(n, (4, 1)), st, double_sided=True)


def cards(centers, rights, ups, uv_rects, normals):
    """Many quads at once (vectorized `card`)."""
    c = np.asarray(centers, float)
    r = np.asarray(rights, float)
    u = np.asarray(ups, float)
    k = len(c)
    pts = np.stack([c - r - u, c + r - u, c + r + u, c - r + u], axis=1).reshape(-1, 3)
    uv = np.asarray(uv_rects, float).reshape(-1, 4)
    if len(uv) == 1:
        uv = np.repeat(uv, k, 0)
    st = np.stack([uv[:, [0, 1]], uv[:, [2, 1]], uv[:, [2, 3]], uv[:, [0, 3]]], axis=1).reshape(-1, 2)
    n = np.asarray(normals, float)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    nrm = np.repeat(n, 4, axis=0)
    return Mesh(pts, np.full(k, 4), np.arange(4 * k), nrm, st, double_sided=True)


def icosphere(subdiv):
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2],
                  [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11],
                  [6, 2, 10], [8, 6, 7], [9, 8, 1]])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(subdiv):
        cache = {}
        verts = list(v)

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                p = (verts[a] + verts[b]) / 2
                verts.append(p / np.linalg.norm(p))
                cache[key] = len(verts) - 1
            return cache[key]
        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        v = np.array(verts)
        f = np.array(nf)
    return Mesh(v, np.full(len(f), 3), f.ravel(), v.copy())


def box(lo, hi, uv_scale=1.0):
    """Axis-aligned box with per-face vertices (flat normals, planar UVs in metres * uv_scale)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    faces = []
    for axis in range(3):
        for sgn in (-1, 1):
            n = np.zeros(3)
            n[axis] = sgn
            a1, a2 = [a for a in range(3) if a != axis]
            if sgn < 0:
                a1, a2 = a2, a1
            corners = []
            for c1, c2 in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = np.zeros(3)
                p[axis] = hi[axis] if sgn > 0 else lo[axis]
                p[a1] = (lo, hi)[c1][a1]
                p[a2] = (lo, hi)[c2][a2]
                corners.append(p)
            corners = np.array(corners)
            st = np.stack([corners[:, a1], corners[:, a2]], 1) * uv_scale
            if axis == 2:
                st = corners[:, :2] * uv_scale
            else:
                hz = corners[:, 2]
                hor = corners[:, a1] if a1 != 2 else corners[:, a2]
                st = np.stack([hor, hz], 1) * uv_scale
            faces.append(Mesh(corners, [4], [0, 1, 2, 3], np.tile(n, (4, 1)), st))
    return merge(faces)


def prism(radius, height, sides, z0=0.0, top_scale=1.0, cap=True, uv_scale=1.0):
    """Faceted vertical prism / truncated cone with flat side normals."""
    ang = np.linspace(0, 2 * np.pi, sides + 1)
    parts = []
    for i in range(sides):
        a0, a1 = ang[i], ang[i + 1]
        p0 = np.array([np.cos(a0), np.sin(a0), 0]) * radius
        p1 = np.array([np.cos(a1), np.sin(a1), 0]) * radius
        q0 = p0 * top_scale
        q1 = p1 * top_scale
        pts = np.array([p0 + [0, 0, z0], p1 + [0, 0, z0], q1 + [0, 0, z0 + height], q0 + [0, 0, z0 + height]])
        n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
        u = np.array([a0, a1, a1, a0]) * radius * uv_scale
        v = np.array([z0, z0, z0 + height, z0 + height]) * uv_scale
        parts.append(Mesh(pts, [4], [0, 1, 2, 3], np.tile(n / np.linalg.norm(n), (4, 1)), np.stack([u, v], 1)))
    if cap:
        top = np.array([[np.cos(a) * radius * top_scale, np.sin(a) * radius * top_scale, z0 + height] for a in ang[:-1]])
        parts.append(Mesh(top, [sides], np.arange(sides), np.tile([0, 0, 1.0], (sides, 1)), top[:, :2] * uv_scale))
    return merge(parts)


def skirt(edge_pts, edge_normals, edge_st, depth):
    """A strip hanging `depth` below an ordered edge polyline (closed loop), facing outward."""
    n = len(edge_pts)
    lower = edge_pts - np.array([0.0, 0.0, 1.0]) * depth
    pts = np.vstack([edge_pts, lower])
    idx = []
    for i in range(n):
        j = (i + 1) % n
        idx.append([i, i + n, j + n, j])
    return Mesh(pts, np.full(n, 4), np.asarray(idx).ravel(), np.vstack([edge_normals, edge_normals]),
                np.vstack([edge_st, edge_st]))
