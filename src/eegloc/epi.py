"""Electrodes as signal voids in the scalp layer of the mean EPI.

The EPI is fat-suppressed and T2*-weighted: each electrode (silver ring and series resistor) leaves
a hole of about 15 mm in the scalp signal. Each run's reference image is sampled in a shell about
the T1's scalp floor, through the run's transform to the T1. Runs are averaged: the head sits a
little differently in each, which smooths the EPI's coarse voxel grid. The layer is divided by its
own broad local mean, and holes are the local minima of that ratio.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

from eegloc import geometry
from eegloc.config import (
    COVERAGE_LEVEL,
    COVERAGE_MARGIN_MM,
    COVERAGE_SMOOTHING_MM,
    EPI_PERCENTILE,
    HOLE_DEPTH,
    HOLE_SMOOTHING_MM,
    HOLE_SPACING_MM,
    LAYER_OFFSETS_MM,
    MAP_SIZE,
    MM_PER_PIXEL,
)
from eegloc.geometry import Array
from eegloc.head import Head

LPS = np.array([-1.0, -1.0, 1.0])  # ITK's axes against the scanner's right, anterior, superior


@dataclass(frozen=True)
class Holes:
    """The holes of a scalp layer, on the map."""

    x: Array  # pixel coordinates of each hole
    y: Array
    depth: Array  # signal lost relative to the local mean, over the map
    covered: NDArray[np.bool_]  # where the EPI has scalp signal at all


@dataclass(frozen=True)
class ItkAffine:
    """An affine transform as ITK writes it, in ITK's axes: p' = R (p - c) + c + t."""

    rotation: Array
    translation: Array
    centre: Array

    @classmethod
    def read(cls, path: str | Path) -> "ItkAffine":
        values = {
            key: np.array(line.split()[1:], dtype=float)
            for line in Path(path).read_text().splitlines()
            for key in ("Parameters:", "FixedParameters:")
            if line.startswith(key)
        }
        if set(values) != {"Parameters:", "FixedParameters:"} or len(values["Parameters:"]) != 12:
            raise ValueError(f"{path}: not an ITK affine transform text file")
        parameters = values["Parameters:"]
        return cls(parameters[:9].reshape(3, 3), parameters[9:], values["FixedParameters:"])

    def apply(self, points: Array) -> Array:
        """The transform of scanner coordinates (right, anterior, superior; mm)."""
        moved = (points * LPS - self.centre) @ self.rotation.T + self.centre + self.translation
        return moved * LPS


def run_shell(head: Head, epi: str | Path, transform: str | Path) -> Array:
    """One run's reference image on the map, at each depth about the scalp floor.

    The transform is the run's EPI-to-T1 image transform (for example fMRIPrep's
    from-boldref_to-T1w text file), which maps T1 points to EPI points.
    """
    image = nib.load(epi)
    if image.ndim != 3:
        raise ValueError(f"{epi}: expected a 3D reference image, got {image.ndim}D")
    data = image.get_fdata()
    data = data / np.percentile(data, EPI_PERCENTILE)
    to_run = ItkAffine.read(transform)
    inverse = np.linalg.inv(image.affine)
    directions = geometry.map_directions()
    shell = np.empty((len(LAYER_OFFSETS_MM), MAP_SIZE, MAP_SIZE))
    for k, offset in enumerate(LAYER_OFFSETS_MM):
        in_t1 = head.to_scanner(directions * (head.floor + offset)[..., None]).reshape(-1, 3)
        voxel = to_run.apply(in_t1) @ inverse[:3, :3].T + inverse[:3, 3]
        sampled = ndimage.map_coordinates(data, voxel.T, order=1, cval=0)
        shell[k] = sampled.reshape(MAP_SIZE, MAP_SIZE)
    return shell


def scalp_layer(shells: list[Array]) -> Array:
    """The scalp layer: the mean over runs, then over depths."""
    total = np.zeros_like(shells[0])
    for shell in shells:
        total += shell
    return (total / len(shells)).mean(axis=0)


def find_holes(layer: Array) -> Holes:
    """The holes of a scalp layer and the part of the map the EPI covers."""
    disc = geometry.map_disc()
    # The EPI does not reach the face or the far occiput.
    broad = ndimage.gaussian_filter(layer, COVERAGE_SMOOTHING_MM / MM_PER_PIXEL)
    covered = disc & (broad > COVERAGE_LEVEL * np.percentile(broad[disc], 90))
    covered = ndimage.binary_erosion(covered, iterations=int(COVERAGE_MARGIN_MM / MM_PER_PIXEL))
    smooth = ndimage.gaussian_filter(layer, HOLE_SMOOTHING_MM / MM_PER_PIXEL)
    depth = 1 - smooth / np.maximum(broad, 1e-6)
    deepest = depth == ndimage.maximum_filter(depth, size=int(HOLE_SPACING_MM / MM_PER_PIXEL))
    rows, cols = np.nonzero(deepest & (depth > HOLE_DEPTH) & covered)
    return Holes(cols.astype(float), rows.astype(float), depth, covered)
