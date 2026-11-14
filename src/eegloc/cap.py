"""The whole cap fitted to the holes, by arc length over the participant's own scalp.

A cap is a fabric: what it fixes is each site's distance from Cz along the scalp and its bearing,
not its angle about some centre of the head. Each site is laid out from Cz along its bearing, over
the scalp floor, until the arc length reaches the layout's (the site's angle from Cz on the
montage's sphere, times a nominal radius) times a stretch.

A pose is where Cz sits (pitch, roll, in degrees from the starting positions), the cap's twist
(degrees) and its stretch along the left-right and front-back bearings. The score of a pose is the
agreement between the sites and the holes, counted both ways: sites that have a hole nearby, and
holes that have a site nearby.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import mne
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from eegloc import geometry
from eegloc.config import (
    MATCH_SIGMA_MM,
    MINIMUM_SITES_IN_VIEW,
    NOMINAL_RADIUS_MM,
    SEARCH,
    SECTION_ANGLES,
)
from eegloc.epi import Holes
from eegloc.geometry import Array
from eegloc.head import Head

Bounds = Sequence[tuple[float, float]]


@dataclass(frozen=True)
class Layout:
    """Where each site sits on the cap, relative to Cz."""

    names: tuple[str, ...]
    angle: Array  # angle from Cz on the montage's sphere, radians
    bearing: Array  # radians: 0 toward the right ear, pi / 2 toward the nose


def montage_layout(names: Sequence[str], montage: str) -> Layout:
    """The layout of the named sites on one of MNE's standard montages."""
    positions = mne.channels.make_standard_montage(montage).get_positions()["ch_pos"]
    lookup = {name.lower(): position for name, position in positions.items()}
    missing = [name for name in (*names, "Cz", "C3", "C4") if name.lower() not in lookup]
    if missing:
        raise ValueError(f"montage {montage} has no site named {missing}")
    points = np.array([lookup[name.lower()] for name in names])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    cz, right, front = _frame(lookup["cz"], lookup["c3"], lookup["c4"])
    angle = np.arccos(np.clip(points @ cz, -1, 1))
    return Layout(tuple(names), angle, np.arctan2(points @ front, points @ right))


def _frame(cz: Array, c3: Array, c4: Array) -> tuple[Array, Array, Array]:
    """Unit vectors toward Cz, toward the right ear and toward the nose."""
    cz = cz / np.linalg.norm(cz)
    right = c4 - c3
    right = right - (right @ cz) * cz
    right = right / np.linalg.norm(right)
    return cz, right, np.cross(cz, right)


class Cap:
    """A cap layout on one participant's scalp, and the holes it is fitted to."""

    def __init__(self, head: Head, holes: Holes, layout: Layout, start: dict[str, Array]):
        """`start` gives approximate positions about the head's centre of Cz, C3 and C4."""
        if len(holes.x) == 0:
            raise ValueError("the EPI scalp layer has no holes")
        self.head = head
        self.names = list(layout.names)
        self.angle, self.bearing = layout.angle, layout.bearing
        toward = {name: start[name] / np.linalg.norm(start[name]) for name in ("Cz", "C3", "C4")}
        self.cz, self.right, self.front = _frame(toward["Cz"], toward["C3"], toward["C4"])
        self.covered = holes.covered
        hole_directions = geometry.to_directions(holes.x, holes.y)
        self.hole_points = hole_directions * head.floor_radius(hole_directions)[:, None]
        self.tree = cKDTree(self.hole_points)

    def frame(self, pitch: float, roll: float) -> tuple[Array, Array, Array]:
        """Cz, right and front once Cz has moved by a pitch and a roll (degrees)."""
        rotation = Rotation.from_rotvec(
            np.deg2rad(pitch) * self.right + np.deg2rad(roll) * self.front
        )
        return rotation.apply(self.cz), rotation.apply(self.right), rotation.apply(self.front)

    def arc_lengths(self, cz: Array, tangents: Array) -> Array:
        """Arc length (mm) from Cz along the scalp section through each tangent.

        One row per tangent, one column per angle of SECTION_ANGLES from Cz.
        """
        section = (
            np.cos(SECTION_ANGLES)[None, :, None] * cz
            + np.sin(SECTION_ANGLES)[None, :, None] * tangents[:, None, :]
        )
        radius = self.head.floor_radius(section)
        steps = np.hypot(
            (radius[:, 1:] + radius[:, :-1]) / 2 * np.diff(SECTION_ANGLES),
            np.diff(radius, axis=1),
        )
        return np.concatenate([np.zeros((len(tangents), 1)), np.cumsum(steps, axis=1)], axis=1)

    def placed(self, pose: Sequence[float]) -> Array:
        """The sites' positions on the scalp floor, mm about the head's centre."""
        pitch, roll, twist, stretch_lr, stretch_ap = pose
        cz, right, front = self.frame(pitch, roll)
        bearing = self.bearing + np.deg2rad(twist)
        tangents = np.cos(bearing)[:, None] * right + np.sin(bearing)[:, None] * front
        arcs = self.arc_lengths(cz, tangents)
        stretch = stretch_lr * np.cos(self.bearing) ** 2 + stretch_ap * np.sin(self.bearing) ** 2
        target = NOMINAL_RADIUS_MM * self.angle * stretch
        angle = np.array(
            [np.interp(t, arc, SECTION_ANGLES) for t, arc in zip(target, arcs, strict=True)]
        )
        directions = np.cos(angle)[:, None] * cz + np.sin(angle)[:, None] * tangents
        return directions * self.head.floor_radius(directions)[:, None]

    def in_view(self, points: Array) -> NDArray[np.bool_]:
        """Whether points about the centre fall where the EPI has scalp signal."""
        return self.covered[geometry.nearest_pixels(points)]

    def matched(self, sites: Array, within: float) -> tuple[NDArray[np.bool_], NDArray[np.intp]]:
        """Which sites have a hole of their own, and each site's nearest hole.

        A site has a hole when the hole is within `within` mm, is nearer to this site than to any
        other, and the site is in the EPI's view.
        """
        distance, nearest = self.tree.query(sites)
        back = cKDTree(sites).query(self.hole_points)[1]
        mutual = back[nearest] == np.arange(len(sites))
        return (distance < within) & mutual & self.in_view(sites), nearest

    def score(self, pose: Sequence[float], subset: NDArray[np.intp] | None = None) -> float:
        """Agreement between the sites (or a subset of them) and the holes, from 0 to 1."""
        sites = self.placed(pose)
        if subset is not None:
            sites = sites[subset]
        seen = self.in_view(sites)
        if seen.sum() < MINIMUM_SITES_IN_VIEW:
            return 0.0
        to_hole = np.exp(-0.5 * (self.tree.query(sites[seen])[0] / MATCH_SIGMA_MM) ** 2)
        to_site = np.exp(-0.5 * (cKDTree(sites).query(self.hole_points)[0] / MATCH_SIGMA_MM) ** 2)
        return float((to_hole.sum() + to_site.sum()) / (len(sites) + len(self.hole_points)))


def fit(
    cap: Cap,
    bounds: Bounds,
    seeds: Sequence[int],
    subset: NDArray[np.intp] | None = None,
) -> tuple[Array, float]:
    """The pose with the highest score within the bounds, and that score.

    The search (differential evolution) is run once per seed and the best result is kept.
    """
    results = [
        differential_evolution(lambda pose: -cap.score(pose, subset), bounds, seed=seed, **SEARCH)
        for seed in seeds
    ]
    best = min(results, key=lambda result: result.fun)
    return best.x, -best.fun
