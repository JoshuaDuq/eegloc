"""The scalp as a flat map: an azimuthal equidistant projection about a centre in the head.

A direction from the centre becomes a pixel. Its angle from the image's superior axis is the
pixel's distance from the middle of the map, and its azimuth is the pixel's bearing: the vertex is
at the middle, the nose is up and the participant's right is on the right.
"""

import numpy as np
from numpy.typing import NDArray

from eegloc.config import MAP_EXTENT, MAP_SIZE

Array = NDArray[np.float64]


def to_directions(x: Array, y: Array) -> Array:
    """Unit vectors (..., 3) of map pixels."""
    gx, gy = x / (MAP_SIZE - 1) * 2 - 1, y / (MAP_SIZE - 1) * 2 - 1
    polar, azimuth = np.hypot(gx, gy) * MAP_EXTENT, np.arctan2(gy, gx)
    return np.stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], axis=-1
    )


def to_pixels(directions: Array) -> tuple[Array, Array]:
    """Map pixels (x, y) of unit vectors (..., 3)."""
    polar = np.arccos(np.clip(directions[..., 2], -1, 1))
    azimuth = np.arctan2(directions[..., 1], directions[..., 0])
    return (
        (polar / MAP_EXTENT * np.cos(azimuth) + 1) / 2 * (MAP_SIZE - 1),
        (polar / MAP_EXTENT * np.sin(azimuth) + 1) / 2 * (MAP_SIZE - 1),
    )


def _grid() -> tuple[Array, Array]:
    u = np.linspace(-1, 1, MAP_SIZE)
    gx, gy = np.meshgrid(u, u)
    return gx, gy


def map_polar() -> tuple[Array, Array]:
    """Every pixel's angle from the superior axis and its azimuth."""
    gx, gy = _grid()
    return np.hypot(gx, gy) * MAP_EXTENT, np.arctan2(gy, gx)


def map_directions() -> Array:
    """Every pixel's unit vector, (MAP_SIZE, MAP_SIZE, 3)."""
    polar, azimuth = map_polar()
    return np.stack(
        [np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)], axis=-1
    )


def map_disc() -> NDArray[np.bool_]:
    """The pixels inside the map's extent."""
    return np.hypot(*_grid()) <= 1


def nearest_pixels(points: Array) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Rows and columns of the map pixels nearest to points about the centre."""
    x, y = to_pixels(points / np.linalg.norm(points, axis=-1, keepdims=True))
    rows = np.clip(np.rint(y).astype(int), 0, MAP_SIZE - 1)
    cols = np.clip(np.rint(x).astype(int), 0, MAP_SIZE - 1)
    return rows, cols
