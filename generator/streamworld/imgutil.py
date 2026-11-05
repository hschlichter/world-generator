"""Image writing (OpenImageIO) and small raster helpers."""

import os

import numpy as np
import OpenImageIO as oiio


def write_image(path, img, quality=92):
    """Write a float [0,1] or uint8 image (H, W, C). Rows are top to bottom."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".exr"):
        return write_exr(path, img)
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, c = a.shape
    spec = oiio.ImageSpec(w, h, c, oiio.UINT8)
    if path.lower().endswith((".jpg", ".jpeg")):
        spec.attribute("Compression", f"jpeg:{quality}")
    elif path.lower().endswith(".png"):
        spec.attribute("png:compressionLevel", 9)
    tmp = path + ".tmp" + os.path.splitext(path)[1]
    out = oiio.ImageOutput.create(tmp)
    if out is None:
        raise RuntimeError(oiio.geterror())
    out.open(tmp, spec)
    out.write_image(np.ascontiguousarray(a))
    out.close()
    os.replace(tmp, path)


def write_exr(path, img):
    """Write a linear float image (H, W, C) as half-float OpenEXR (zip compression). Rows are top to bottom."""
    a = np.ascontiguousarray(np.asarray(img, dtype=np.float32))
    if a.ndim == 2:
        a = a[:, :, None]
    h, w, c = a.shape
    spec = oiio.ImageSpec(w, h, c, oiio.HALF)
    spec.attribute("compression", "zip")
    tmp = path + ".tmp.exr"
    out = oiio.ImageOutput.create(tmp)
    if out is None:
        raise RuntimeError(oiio.geterror())
    out.open(tmp, spec)
    out.write_image(a)
    out.close()
    os.replace(tmp, path)


def read_image(path):
    inp = oiio.ImageInput.open(path)
    if inp is None:
        raise RuntimeError(oiio.geterror())
    a = inp.read_image(format=oiio.FLOAT)
    inp.close()
    return np.asarray(a)


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def downsample(img, factor):
    """Box downsample by an integer factor."""
    h, w = img.shape[:2]
    a = img[: h // factor * factor, : w // factor * factor]
    sh = (h // factor, factor, w // factor, factor) + a.shape[2:]
    return a.reshape(sh).mean(axis=(1, 3))


def resize_bilinear(img, out_h, out_w):
    h, w = img.shape[:2]
    ys = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    ty = np.clip(ys - y0, 0, 1)[:, None]
    tx = np.clip(xs - x0, 0, 1)[None, :]
    if img.ndim == 3:
        ty = ty[..., None]
        tx = tx[..., None]
    a = img[y0][:, x0] * (1 - tx) + img[y0][:, x1] * tx
    b = img[y1][:, x0] * (1 - tx) + img[y1][:, x1] * tx
    return a * (1 - ty) + b * ty


def height_to_normal(h, spacing, strength=1.0):
    """Tangent-space normal map (encoded 0..1) from a height image, rows top-to-bottom = +v down."""
    gy, gx = np.gradient(h * strength, spacing)
    # Image rows run top to bottom, texture v runs bottom to top: flip the y gradient.
    n = np.stack([-gx, gy, np.ones_like(h)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def tileable_noise(size, seed, periods, octaves=4, gain=0.5):
    """Seamlessly tileable fBm image of shape (size, size) in roughly [-1, 1]."""
    from .noise import hash_float, _fade
    total = np.zeros((size, size))
    amp = 1.0
    norm = 0.0
    p = periods
    for o in range(octaves):
        u = np.arange(size) * p / size
        X, Y = np.meshgrid(u, u)
        xi = np.floor(X).astype(np.int64)
        yi = np.floor(Y).astype(np.int64)
        xf = X - xi
        yf = Y - yi

        def g(ix, iy, dx, dy):
            a = hash_float(ix % p, iy % p, seed + o * 97) * 2 * np.pi
            return np.cos(a) * dx + np.sin(a) * dy
        n00 = g(xi, yi, xf, yf)
        n10 = g(xi + 1, yi, xf - 1, yf)
        n01 = g(xi, yi + 1, xf, yf - 1)
        n11 = g(xi + 1, yi + 1, xf - 1, yf - 1)
        uu = _fade(xf)
        vv = _fade(yf)
        total += amp * 1.414 * ((n00 + uu * (n10 - n00)) * (1 - vv) + (n01 + uu * (n11 - n01)) * vv)
        norm += amp
        amp *= gain
        p *= 2
    return total / norm
