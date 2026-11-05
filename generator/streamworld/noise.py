"""Deterministic, vectorized 2D noise built on integer hashing.

Every function is a pure function of its coordinates and seed, so any sub-grid of the world can be evaluated on its
own and gives exactly the values the whole-world evaluation gives. All arithmetic is float64.
"""

import numpy as np

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)


def hash_u64(*keys):
    """SplitMix64-style hash of integer arrays (broadcast together). Returns uint64."""
    with np.errstate(over="ignore"):
        h = np.uint64(0x2545F4914F6CDD1D)
        for k in keys:
            k = np.asarray(k).astype(np.int64).astype(np.uint64)
            h = (h ^ k) * _M1
            h = (h ^ (h >> np.uint64(30))) * _M2
            h = (h ^ (h >> np.uint64(27))) * _M3
            h = h ^ (h >> np.uint64(31))
        return h


def hash_float(*keys):
    """Uniform floats in [0, 1) from integer keys."""
    return (hash_u64(*keys) >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def perlin(x, y, seed):
    """Gradient noise in roughly [-1, 1] with unit lattice spacing."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.floor(x)
    yi = np.floor(y)
    xf = x - xi
    yf = y - yi
    xi = xi.astype(np.int64)
    yi = yi.astype(np.int64)

    def grad(ix, iy, dx, dy):
        a = hash_float(ix, iy, seed) * (2.0 * np.pi)
        return np.cos(a) * dx + np.sin(a) * dy

    n00 = grad(xi, yi, xf, yf)
    n10 = grad(xi + 1, yi, xf - 1.0, yf)
    n01 = grad(xi, yi + 1, xf, yf - 1.0)
    n11 = grad(xi + 1, yi + 1, xf - 1.0, yf - 1.0)
    u = _fade(xf)
    v = _fade(yf)
    nx0 = n00 + u * (n10 - n00)
    nx1 = n01 + u * (n11 - n01)
    return (nx0 + v * (nx1 - nx0)) * 1.41421356


def fbm(x, y, seed, wavelength, octaves=5, lacunarity=2.0, gain=0.5):
    """Fractal sum of perlin octaves, normalised to roughly [-1, 1]."""
    total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
    amp = 1.0
    norm = 0.0
    freq = 1.0 / wavelength
    for o in range(octaves):
        # Rotate each octave a little to break lattice alignment.
        c, s = np.cos(0.5 + o * 1.7), np.sin(0.5 + o * 1.7)
        xr = (x * c - y * s) * freq
        yr = (x * s + y * c) * freq
        total += amp * perlin(xr, yr, seed + o * 1013)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def ridged(x, y, seed, wavelength, octaves=5, lacunarity=2.05, gain=0.5, sharpness=2.0):
    """Ridged multifractal in [0, 1]: sharp crests, rounded valleys."""
    total = np.zeros(np.broadcast(np.asarray(x), np.asarray(y)).shape)
    amp = 1.0
    norm = 0.0
    freq = 1.0 / wavelength
    weight = 1.0
    for o in range(octaves):
        c, s = np.cos(1.1 + o * 2.3), np.sin(1.1 + o * 2.3)
        xr = (x * c - y * s) * freq
        yr = (x * s + y * c) * freq
        n = 1.0 - np.abs(perlin(xr, yr, seed + o * 7919))
        n = n ** sharpness
        n = n * weight
        weight = np.clip(n * 1.6, 0.0, 1.0)
        total += amp * n
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def warp(x, y, seed, wavelength, amount):
    """Domain warp: returns displaced coordinates."""
    wx = fbm(x, y, seed + 17, wavelength, octaves=3)
    wy = fbm(x, y, seed + 31, wavelength, octaves=3)
    return x + wx * amount, y + wy * amount


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
