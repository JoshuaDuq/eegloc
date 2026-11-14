"""The head's outer boundary in the T1 acquired with the cap on (after Bhutada et al., 2020).

Along each ray from a centre in the head, the outer boundary is the outermost sample above a small
fraction of scalp fat: skin, or gel where there is any. Unlike a skin threshold, this does not
depend on how bright the skin itself is. The boundary's smooth floor is the scalp the cap lies on,
and its height above that floor shows the gel bumps and the dents of the electrode holders.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from eegloc import geometry
from eegloc.config import (
    BOUNDARY_LEVEL,
    CENTRE_DEPTH_MM,
    FAT_PERCENTILE,
    FLOOR_PERCENTILE,
    FLOOR_SMOOTHING_PIXELS,
    FLOOR_WINDOW_PIXELS,
    HEAD_LEVEL,
    MAP_SIZE,
    RAY_RADII_MM,
    SLAB_MM,
    T1_SMOOTHING_MM,
)
from eegloc.geometry import Array


@dataclass(frozen=True)
class Head:
    """The scalp of one participant as maps about a centre in the head.

    Points "about the centre" are in mm along the T1's own axes (right, anterior, superior after
    reorientation), with the centre as origin.
    """

    affine: Array  # voxel to scanner coordinates of the reoriented T1
    zooms: Array  # voxel size, mm
    centre: Array  # mm along the T1's own axes
    outer: Array  # radius of the outer boundary, mm (0 where a ray meets none)
    floor: Array  # radius of the scalp floor, mm
    bump: Array  # height of the outer boundary above the floor, mm (NaN where there is none)

    def to_scanner(self, points: Array) -> Array:
        """Scanner coordinates (mm) of points about the centre."""
        voxel = (self.centre + points) / self.zooms
        return voxel @ self.affine[:3, :3].T + self.affine[:3, 3]

    def from_scanner(self, points: Array) -> Array:
        """Points about the centre from scanner coordinates (mm)."""
        inverse = np.linalg.inv(self.affine)
        return (points @ inverse[:3, :3].T + inverse[:3, 3]) * self.zooms - self.centre

    def floor_radius(self, directions: Array) -> Array:
        """Radius of the scalp floor along unit vectors (..., 3)."""
        x, y = geometry.to_pixels(directions)
        radius = ndimage.map_coordinates(
            self.floor, [np.ravel(y), np.ravel(x)], order=1, mode="nearest"
        )
        return radius.reshape(np.shape(x))


def read_head(t1: str | Path) -> Head:
    """The outer-boundary maps of a T1 acquired with the cap on."""
    image = nib.as_closest_canonical(nib.load(t1))
    if image.ndim != 3:
        raise ValueError(f"{t1}: expected a 3D image, got {image.ndim}D")
    zooms = np.array(image.header.get_zooms(), dtype=float)
    data = ndimage.gaussian_filter(image.get_fdata(), T1_SMOOTHING_MM / zooms)
    fat = np.percentile(data, FAT_PERCENTILE)

    # The centre: the centre of mass of the top of the head, a fixed depth below its top.
    bright = data > HEAD_LEVEL * fat
    top = np.nonzero(bright.any(axis=(0, 1)))[0].max()
    slab = np.zeros_like(bright)
    bottom = max(top - round(SLAB_MM / zooms[2]), 0)
    slab[:, :, bottom : top + 1] = bright[:, :, bottom : top + 1]
    centre = np.array(ndimage.center_of_mass(slab)) * zooms
    centre[2] = top * zooms[2] - CENTRE_DEPTH_MM

    # Walk inward from far outside: the first radius where the signal rises above the level.
    directions = geometry.map_directions()
    outer = np.zeros((MAP_SIZE, MAP_SIZE))
    found = np.zeros((MAP_SIZE, MAP_SIZE), bool)
    for radius in RAY_RADII_MM[::-1]:
        voxel = (centre + directions * radius) / zooms
        value = ndimage.map_coordinates(data, voxel.reshape(-1, 3).T, order=1, cval=0)
        hit = (value.reshape(MAP_SIZE, MAP_SIZE) > BOUNDARY_LEVEL * fat) & ~found
        outer[hit] = radius
        found |= hit

    floor = ndimage.gaussian_filter(
        ndimage.percentile_filter(outer, FLOOR_PERCENTILE, size=FLOOR_WINDOW_PIXELS),
        FLOOR_SMOOTHING_PIXELS,
    )
    bump = np.where(geometry.map_disc() & found, outer - floor, np.nan)
    return Head(image.affine, zooms, centre, outer, floor, bump)
